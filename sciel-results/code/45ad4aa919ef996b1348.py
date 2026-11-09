<<<<<<< SEARCH
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] should equal 1 if the term contributes to C[i,k], else 0
    # Actually, the standard embedding: T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k} for the "correct" terms?
    
    # Let me reconsider. The matrix multiplication tensor T is defined as:
    # T[i,j,k] = 1 if this term A[i,j]*B[j,k] appears in C[i,k]
    # For standard matrix mult, every (i,j,k) triple contributes exactly once to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k in the valid ranges.
    
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need this to equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_limit=30):
    """
    Try to reduce the rank by finding linear combinations that allow elimination.
    Uses the structure of prime dimensions to constrain search.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to find if any rank-1 term can be expressed as a linear combination of others
    # with integer/half-integer coefficients
    
    # For each pair of terms, check if one can be eliminated
    # This is expensive, so we use a greedy approach with time limit
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try removing one term at a time and checking if we can adjust others
    for c_remove in range(rank):
        if time.time() - start_time > time_limit:
            break
        
        # Skip the term to be removed
        remaining = [c for c in range(rank) if c != c_remove]
        
        # Check if term c_remove can be approximated by linear combination of remaining
        # with small integer/half-integer coefficients
        
        # Get the vectors for the term to remove
        u_vec = np.array([U_arr[i*m+j, c_remove] for i in range(n) for j in range(m)])
        v_vec = np.array([V_arr[j*p+k, c_remove] for j in range(m) for k in range(p)])
        w_vec = np.array([W_arr[k*n+i, c_remove] for k in range(p) for i in range(n)])
        
        # Try to find coefficients alpha_c for remaining terms such that:
        # sum_c alpha_c * (u_c, v_c, w_c) approximates (u_vec, v_vec, w_vec)
        # This is a linear system. We'll use a simple approach: try small coefficients
        
        # For efficiency, only try coefficients in {-1, -0.5, 0, 0.5, 1}
        # This is very limited but fast
        
        # Actually, let's try a different approach: look for structural patterns
        # Given prime dimensions 3,5,7, there might be symmetry-based reductions
        
        pass
    
    return best_U, best_V, best_W, best_rank
=======
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
    """Verify using NumPy for speed."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k]) * W_arr[k*n+i]
                # Actually need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    return _verify_decomposition_fast(U, V, W, n, m, p, tol)

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_limit=30):
    """
    Try to reduce the rank by finding linear combinations that allow elimination.
    Uses a greedy approach: try removing each term and see if the remaining terms
    can be adjusted (with small rational coefficients) to still satisfy the tensor equation.
    """
    start_time = time.time()
    rank = len(U[0])
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each term to potentially remove, check if it's redundant
    for c_remove in range(rank):
        if time.time() - start_time > time_limit:
            break
        
        remaining = [c for c in range(rank) if c != c_remove]
        if not remaining:
            break
        
        # Extract the "contribution pattern" of term c_remove
        # Term c contributes: U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We want to see if we can find coefficients alpha_c for c in remaining
        # such that sum_c alpha_c * (U[:,c] ⊗ V[:,c] ⊗ W[:,c]) ≈ U[:,c_remove] ⊗ V[:,c_remove] ⊗ W[:,c_remove]
        
        # This is a trilinear equation, hard to solve directly.
        # Alternative: check if the outer product of U[:,c_remove], V[:,c_remove], W[:,c_remove]
        # is in the span of the other outer products.
        
        # Simplified check: if all three vectors have zeros in the same positions as some other term,
        # or if there's a simple linear relationship.
        
        # For now, just check if removing this term and zeroing it out still gives a valid decomposition
        # (this only works if the term was already zero, which it isn't in schoolbook)
        
        # Better: try to find if any term is a scalar multiple of another in all three components
        found_duplicate = False
        for c_other in remaining:
            # Check if U[:,c_remove] = a * U[:,c_other], V[:,c_remove] = a * V[:,c_other], W[:,c_remove] = a * W[:,c_other]
            u_rm = U_arr[:, c_remove]
            u_ot = U_arr[:, c_other]
            v_rm = V_arr[:, c_remove]
            v_ot = V_arr[:, c_other]
            w_rm = W_arr[:, c_remove]
            w_ot = W_arr[:, c_other]
            
            # Find scale factor from U
            nonzero_u = u_ot != 0
            if np.any(nonzero_u):
                a_candidates = u_rm[nonzero_u] / u_ot[nonzero_u]
                if len(np.unique(np.round(a_candidates, 10))) == 1:
                    a = a_candidates[0]
                    # Check V and W
                    if np.allclose(v_rm, a * v_ot, atol=tol) and np.allclose(w_rm, a * w_ot, atol=tol):
                        # Terms are proportional! We can merge them.
                        # Remove c_remove and adjust c_other
                        found_duplicate = True
                        break
        
        if found_duplicate:
            # Build new matrices without c_remove
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            
            # Zero out column c_remove
            for i in range(len(new_U)):
                new_U[i][c_remove] = 0
            for i in range(len(new_V)):
                new_V[i][c_remove] = 0
            for i in range(len(new_W)):
                new_W[i][c_remove] = 0
            
            new_rank = best_rank - 1
            if new_rank < best_rank:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                rank = new_rank  # Update for subsequent iterations
                # Re-convert arrays
                U_arr = np.array(best_U, dtype=float)
                V_arr = np.array(best_V, dtype=float)
                W_arr = np.array(best_W, dtype=float)
    
    return best_U, best_V, best_W, best_rank
>>>>>>> REPLACE

