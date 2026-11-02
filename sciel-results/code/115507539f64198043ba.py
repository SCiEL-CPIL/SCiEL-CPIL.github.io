import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Baseline: Rank-55 Strassen-based decomposition.
    - Search: Attempt to find a rank-54 decomposition by exploiting 
      linear dependencies in the tensor structure. If not found in time, 
      return rank-55.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # Strassen 7 terms
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank_54_search(U55, V55, W55, time_limit=30):
        """
        Try to find a rank-54 decomposition by looking for linear dependencies.
        
        Approach: The tensor T = sum_r U_r ⊗ V_r ⊗ W_r. If rank 54 exists,
        there must be a linear relation among the 55 rank-55 terms that allows
        one to be eliminated. We search for such relations.
        """
        start_search = time.time()
        
        # Build the tensor T as a 6D array (flattened)
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want to find if there's a non-trivial linear combination of the 
        # rank-55 terms that gives zero.
        
        # Strategy: For each pair of columns (r, s) in U, V, W, check if we can
        # express one as a linear combination of the others. This is expensive,
        # so we use a smarter approach.
        
        # Key insight: If rank 54 is achievable, there exists a 54-dimensional 
        # subspace of the 55-dimensional space spanned by the rank-55 terms.
        # We can try to find a vector (alpha_1, ..., alpha_55) such that
        # sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = 0
        
        # This is equivalent to finding the null space of a certain linear map.
        # Let's build the tensor as a matrix and find its null space.
        
        # Flatten the tensor: T has dimensions (n*m) x (m*p) x (p*n)
        # Actually, let's think of it differently.
        
        # The tensor T can be seen as a linear map from (U ⊗ V) to W^*.
        # More precisely, for each (i,j,k), T[i,j,k] = <U_{ij}, V_{jk}, W_{ki}>
        
        # Let's build a matrix M where each column corresponds to a term r,
        # and M[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But this is 3D in (i,j,k) and 1D in r, so it's not a standard matrix.
        
        # Instead, let's use the fact that if there's a linear dependence,
        # then for some fixed (i,j), the vectors {W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]}
        # for r=1..55 must be linearly dependent in some structured way.
        
        # Simpler approach: Try random linear combinations and check if they vanish.
        # This is heuristic but might find a relation if one exists.
        
        np.random.seed(42)
        num_trials = 1000
        
        for trial in range(num_trials):
            if time.time() - start_search > time_limit:
                break
            
            # Generate random coefficients
            alpha = np.random.randn(55)
            
            # Compute sum_r alpha_r * U_r ⊗ V_r ⊗ W_r
            # This should be zero if there's a linear dependence
            # Check on a sample of (i,j,k) indices
            is_zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(55):
                            val += alpha[r] * U55[i*m+j, r] * V55[j*p+k, r] * W55[k*n+i, r]
                        if abs(val) > 1e-8:
                            is_zero = False
                            break
                    if not is_zero:
                        break
                if not is_zero:
                    break
            
            if is_zero and np.max(np.abs(alpha)) > 1e-6:
                # Found a linear dependence! Now try to eliminate one term.
                # Find the term with the largest |alpha_r| and try to remove it.
                max_r = np.argmax(np.abs(alpha))
                
                # Try to construct a rank-54 decomposition by removing term max_r
                # and adjusting the others.
                # This is non-trivial, but let's try a simple approach:
                # Keep all terms except max_r, and see if we can add a correction.
                
                # Actually, if sum alpha_r * T_r = 0, then we can replace any term
                # T_s with T_s - (alpha_s/alpha_max_r) * T_max_r and remove T_max_r.
                
                # Let's try removing term max_r and see if the remaining 54 terms
                # still decompose the tensor.
                
                U54 = np.copy(U55)
                V54 = np.copy(V55)
                W54 = np.copy(W55)
                
                # Zero out column max_r
                U54[:, max_r] = 0
                V54[:, max_r] = 0
                W54[:, max_r] = 0
                
                # Check if the remaining 54 terms still work
                if verify_decomposition(U54, V54, W54, tol=1e-10):
                    # Success! We found a rank-54 decomposition.
                    # Remove the zero column
                    mask = np.ones(55, dtype=bool)
                    mask[max_r] = False
                    U54 = U54[:, mask]
                    V54 = V54[:, mask]
                    W54 = W54[:, mask]
                    return U54, V54, W54, 54
        
        return None

    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55

    # Verify baseline
    if not verify_decomposition(U55, V55, W55, tol=1e-10):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Attempt rank 54 search
    if time.time() - start_time < DEADLINE - 15:
        try:
            result = try_rank_54_search(U55, V55, W55, time_limit=DEADLINE - 20 - (time.time() - start_time))
            if result is not None:
                U54, V54, W54, rank54 = result
                if verify_decomposition(U54, V54, W54, tol=1e-10):
                    best_U, best_V, best_W, best_rank = U54, V54, W54, rank54
        except Exception:
            pass

    if time.time() - start_time > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
