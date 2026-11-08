"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.linalg import null_space

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)  
    W: shape (p*n, rank)
    
    For the tensor T_{(i,j),(j,k),(k,i)} = delta_{...}, we need:
    T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check that rank is consistent
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i,j) pair and (k) pair, verify the sum
    # T[i*m+j, j*p+k, k*n+i] should be 1 if the indices are "compatible" in the tensor sense
    # Actually for matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    # Otherwise T[a,b,c] = 0
    
    # Let's verify by checking the bilinear form
    # For fixed i,k: sum_j T[i*m+j, j*p+k, k*n+i] should give the correct structure
    # Actually, the correct check is:
    # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For incompatible indices (where the middle index doesn't match), it should be 0
    
    # But actually the tensor is defined such that:
    # T_{(i,j),(j,k),(k,i)} = 1 only when the three indices share consistent j,k
    # Let me just verify the full tensor contraction
    
    # Reshape approach:
    # U reshaped: U_reshaped[i, j, c] = U[i*m+j, c]
    # V reshaped: V_reshaped[j, k, c] = V[j*p+k, c]  
    # W reshaped: W_reshaped[k, i, c] = W[k*n+i, c]
    
    # Then sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c] should equal 1 for all i,j,k
    # This is because T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    # Also check that incompatible entries are zero
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i' with j != j' or k != k' should be 0
    # Let's check a few random incompatible entries
    rng = np.random.RandomState(42)
    for _ in range(1000):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        j2 = rng.randint(m)
        k2 = rng.randint(p)
        # Make sure they're incompatible
        if j == j2 and k == k2:
            continue
        a = i * m + j
        b = j2 * p + k
        c = k2 * n + i
        val = np.sum(U[a,:] * V[b,:] * W[c,:])
        if abs(val) > max_err:
            max_err = abs(val)
    
    return max_err < tol

def _is_valid_integer_half(U, V, W, tol=1e-9):
    """Check if all entries are integers or half-integers."""
    for mat in [U, V, W]:
        mat = np.array(mat, dtype=float)
        # 2*x should be an integer
        twice = 2.0 * mat
        rounded = np.round(twice)
        if np.any(np.abs(twice - rounded) > tol):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 60
    
    # Start with the trivial schoolbook decomposition as fallback
    U0 = np.zeros((n * m, trivial_rank))
    V0 = np.zeros((m * p, trivial_rank))
    W0 = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = trivial_rank
    
    # Try to find a lower rank decomposition using structured approaches
    # Strategy: exploit the shared middle dimension structure
    
    # Approach 1: Try rank = n*p + m*p + n*m - some savings
    # Actually, let's try a more systematic approach using the structure
    
    # The key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as: for each j, define intermediate products
    # But that's just the trivial decomposition.
    
    # Alternative: use the fact that we can share computations across i or k
    # For fixed j: A[i,j] * B[j,k] for all i,k is a rank-1 matrix (column i of A) ⊗ (row k of B)
    # So C = sum_j (A[:,j] ⊗ B[j,:])
    # Each term A[:,j] ⊗ B[j,:] is an outer product of an n-vector and p-vector
    # To express A[:,j] * B[j,k] as a sum of rank-1 terms in the tensor sense:
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j] * B[j,k]
    
    # This is equivalent to finding a CP decomposition of each "slice"
    # But the standard trick is to note that if we can diagonalize the middle operations...
    
    # Let's try: for each j, the sub-tensor T_j has T_j[i,k] = A[i,j]*B[j,k]
    # In our tensor notation: T[i*m+j, j*p+k, k*n+i] = A[i,j]*B[j,k]
    # For this to work with our factorization: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = A[i,j]*B[j,k]
    # 
    # If we set for term c: U[i*m+j,c] = u_i, V[j*p+k,c] = v_k, W[k*n+i,c] = w
    # Then sum_c u_i * v_k * w = A[i,j]*B[j,k]
    # This means the matrix (u_i * v_k) * w summed over c should equal A[i,j]*B[j,k]
    # i.e., sum_c w_c * u_c[i] * v_c[k] = A[i,j] * B[j,k]
    # This is a rank-r CP decomposition of the matrix A[:,j] * B[j,:]^T
    # But A[:,j] * B[j,:]^T is already rank 1! So we only need 1 term per j.
    # With U[i*m+j, c_j] = A[i,j], V[j*p+k, c_j] = B[j,k], W[k*n+i, c_j] = 1
    # But wait, this doesn't work because U depends on the input A, not fixed.
    
    # Right, the decomposition must be FIXED (independent of input matrices).
    # So we need fixed coefficients such that for ANY input A, B:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] (with appropriate indexing)
    
    # This is a bilinear identity. The standard result is that the minimal rank
    # for m×n×p matrix multiplication is at least max(m*n, n*p, m*p) and at most m*n*p.
    # For 3×4×5: lower bound max(12,20,15)=20, upper bound 60.
    # Known best: I don't think we can do much better than 60 for such small dimensions
    # without specialized constructions.
    
    # Let's try a random search with integer/half-integer constraint for ranks between 20 and 59
    
    # Actually, let's try a more structured approach first
    # Group by the middle index j:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # For each j, we need to express the "product" A[i,j]*B[j,k] using shared terms.
    # If we use the same set of terms for all j, we can potentially save rank.
    
    # Let's try: assume U[i*m+j, c] = f_c(i) for all j (independent of j)
    # And V[j*p+k, c] = g_c(k) for all j (independent of j)  
    # And W[k*n+i, c] = h_c(i,k)
    # Then: sum_c f_c(i) * g_c(k) * h_c(i,k) = sum_j A[i,j] B[j,k]
    # This seems unlikely to work with fixed coefficients.
    
    # Let's just do a careful random search for low ranks
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound up to trivial_rank - 1
    # Lower bound: max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # But realistic lower bound for 3x4x5 might be higher
    
    for target_rank in range(25, min(trivial_rank, 50)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random half-integer matrices
            scale = 1.0
            U_rand = (np.random.randint(0, 3, size=(n*m, target_rank)) - 1) * 0.5  # -0.5, 0, 0.5
            V_rand = (np.random.randint(0, 3, size=(m*p, target_rank)) - 1) * 0.5
            W_rand = (np.random.randint(0, 3, size=(p*n, target_rank)) - 1) * 0.5
            
            # Try a few iterations of alternating optimization with rounding
            U_curr = U_rand.copy()
            V_curr = V_rand.copy()
            W_curr = W_rand.copy()
            
            for it in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Update W: for each (k,i), W[k*n+i,:] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # This is a linear system: for fixed (i,k), 
                # Let M be the (m x target_rank) matrix with M[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # Then we need M @ W[k*n+i,:].T = 1 (vector of m ones)
                # If M is square and invertible, W[k*n+i,:].T = M^{-1} @ ones
                # Otherwise, use least squares
                
                improved = False
                W_new = W_curr.copy()
                for i in range(n):
                    for k in range(p):
                        M = np.outer(
                            U_curr[i*m:(i+1)*m, :].T,  # (target_rank, m) -> transpose to (m, target_rank)? 
                            V_curr[k*p:(k+1)*p, :]      # Wait, V is indexed by j*p+k
                        )
                        # V[j*p+k, c] for fixed k: indices are j*p+k for j=0..m-1
                        # So V_curr[j*p+k, :] for j in range(m)
                        V_fixed_k = np.array([V_curr[j*p+k, :] for j in range(m)])  # (m, target_rank)
                        U_fixed_i = U_curr[i*m:(i+1)*m, :]  # (m, target_rank)
                        M = U_fixed_i * V_fixed_k  # element-wise, (m, target_rank)
                        
                        # Solve M @ w_vec = ones(m)
                        try:
                            if target_rank >= m:
                                w_vec, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            else:
                                # Underdetermined - use pseudo-inverse
                                w_vec = np.linalg.pinv(M) @ np.ones(m)
                        except:
                            continue
                        
                        W_new[k*n+i, :] = w_vec
                        if np.linalg.norm(w_vec - W_curr[k*n+i, :]) > 1e-10:
                            improved = True
                
                W_curr = W_new
                
                # Update U similarly
                U_new = U_curr.copy()
                for i in range(n):
                    for j in range(m):
                        # For fixed (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        # V[j*p+k, c] * W[k*n+i, c] for k=0..p-1, c=0..target_rank-1
                        # Let N[k,c] = V[j*p+k,c] * W[k*n+i,c], shape (p, target_rank)
                        # Need U[i*m+j,:] @ N.T = ones(p)
                        N = np.array([V_curr[j*p+k, :] * W_curr[k*n+i, :] for k in range(p)])  # (p, target_rank)
                        try:
                            if target_rank >= p:
                                u_vec, residuals, rank_n, sv = np.linalg.lstsq(N, np.ones(p), rcond=None)
                            else:
                                u_vec = np.linalg.pinv(N) @ np.ones(p)
                        except:
                            continue
                        U_new[i*m+j, :] = u_vec
                        if np.linalg.norm(u_vec - U_curr[i*m+j, :]) > 1e-10:
                            improved = True
                U_curr = U_new
                
                # Update V
                V_new = V_curr.copy()
                for j in range(m):
                    for k in range(p):
                        # For fixed (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let O[i,c] = U[i*m+j,c] * W[k*n+i,c], shape (n, target_rank)
                        # Need V[j*p+k,:] @ O.T = ones(n)
                        O = np.array([U_curr[i*m+j, :] * W_curr[k*n+i, :] for i in range(n)])  # (n, target_rank)
                        try:
                            if target_rank >= n:
                                v_vec, residuals, rank_o, sv = np.linalg.lstsq(O, np.ones(n), rcond=None)
                            else:
                                v_vec = np.linalg.pinv(O) @ np.ones(n)
                        except:
                            continue
                        V_new[j*p+k, :] = v_vec
                        if np.linalg.norm(v_vec - V_curr[j*p+k, :]) > 1e-10:
                            improved = True
                V_curr = V_new
                
                if not improved:
                    break
            
            # Round to half-integers
            U_rounded = np.round(2 * U_curr) / 2.0
            V_rounded = np.round(2 * V_curr) / 2.0
            W_rounded = np.round(2 * W_curr) / 2.0
            
            # Verify
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(), 
                        "W": W_rounded.tolist()
                    }
                    print(f"Found valid decomposition with rank {target_rank}")
                    # Don't break - try to find even lower rank
                break  # Found valid for this rank, move to next
    
    return best_result
# EVOLVE-BLOCK-END
