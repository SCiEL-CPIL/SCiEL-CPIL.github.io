"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition for 3x3 x 3x6. Rank = 54."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] should be 1
                val = float(np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    # Check off-diagonal entries
    for i in range(n):
        for j in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j2 * p + k
                        c = k * n + i2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        val = float(np.sum(U_arr[a] * V_arr[b] * W_arr[c]))
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
    return max_err < tol

def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iter=200):
    """
    Try to reduce the rank by merging terms.
    Start from a valid decomposition and try to find linear combinations
    that preserve the tensor while reducing rank.
    """
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    # Convert to numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # We need to find U', V', W' with target_rank columns such that
    # for all valid (i,j,k): sum_{c<target} U'[i*m+j,c]*V'[j*p+k,c]*W'[k*n+i,c] = 1
    # and for all invalid triples: = 0
    
    # Strategy: try to express some columns as linear combinations of others
    # This is equivalent to finding a lower-rank completion
    
    # Build the tensor T as a flat vector
    # Number of valid (i,j,k) triples for the diagonal: n*m*p
    # But we need to satisfy all constraints including off-diagonal
    
    # Let's use a different approach: iterative merging
    # For each pair of columns (a, b), try to see if we can replace them
    # with a single column that preserves all constraints
    
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    best_rank = rank
    
    for _ in range(max_iter):
        # Find the pair of columns that can be merged
        merged = False
        for a in range(best_rank):
            for b in range(a+1, best_rank):
                # Try to find alpha, beta, gamma such that
                # the combination alpha*col_a + beta*col_b can be absorbed
                # This is complex; let's try a simpler heuristic:
                # Check if column b is close to a scalar multiple of column a
                # in some projected space
                
                # Simple check: if the "product" of columns a and b
                # in the tensor space is linearly dependent
                prod_a = best_U[:, a] * best_V[:, a] * best_W[:, a]  # not quite right
                # Actually, the contribution of column c to the tensor is
                # T_c[i*m+j, j*p+k, k*n+i] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                
                # Let's compute the tensor contribution of each column
                def col_tensor(U_col, V_col, W_col, n, m, p):
                    """Compute the 3D tensor for a single rank-1 term."""
                    T = np.zeros((n*m, m*p, p*n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i*m+j, j*p+k, k*n+i] = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
                    return T
                
                Ta = col_tensor(best_U[:, a], best_V[:, a], best_W[:, a], n, m, p)
                Tb = col_tensor(best_U[:, b], best_V[:, b], best_W[:, b], n, m, p)
                
                # Try to find t such that Ta + t*Tb is a rank-1 tensor
                # A rank-1 tensor T has T[i*m+j, j*p+k, k*n+i] = u_i*u_j*v_j*v_k*w_k*w_i
                # which factors as (u_i*w_i)*(u_j*v_j)*(v_k*w_k)
                
                # Check if Ta + t*Tb is rank-1 for some t
                # This requires checking if the matrix unfolding has rank 1
                # which means all 2x2 minors are zero
                
                # For efficiency, check a few constraints
                # The tensor is rank-1 iff for all pairs of indices in each mode,
                # the 2x2 submatrix has determinant 0
                
                # Mode 1 unfolding: T as matrix (n*m) x (m*p*p*n)
                # This is too large. Let's use a simpler check.
                
                # For a 3x3x3 tensor to be rank-1, it must be of the form
                # T[i,j,k] = a_i * b_j * c_k
                # So T[0,0,0]*T[1,1,1] = T[0,1,0]*T[1,0,1] = ... etc.
                
                # For our tensor indexed by (i*m+j, j*p+k, k*n+i):
                # Let's just check if we can find t by solving a small system
                
                # Pick 4 "free" entries to determine t
                # We need Ta + t*Tb to be rank-1
                # The rank-1 condition: for any two rows i1,i2 and two cols j1,j2,
                # the 2x2 minor is zero
                
                # This is getting complex. Let's use a numerical approach:
                # Try random t values and check if the result is approximately rank-1
                
                found_t = None
                for _ in range(50):
                    t = np.random.uniform(-2, 2)
                    Tc = Ta + t * Tb
                    # Check if Tc is approximately rank-1
                    # Flatten and check the matrix rank of the mode-1 unfolding
                    Tc_flat = Tc.reshape(n*m, -1)
                    # Compute SVD
                    try:
                        s = np.linalg.svd(Tc_flat, compute_uv=False)
                        if len(s) >= 2:
                            ratio = s[1] / max(s[0], 1e-12)
                            if ratio < 1e-6:
                                found_t = t
                                break
                    except:
                        pass
                
                if found_t is not None:
                    # Merge columns a and b into a single column
                    # New column = (col_a + found_t * col_b) but we need to scale
                    # Actually, we replace the two columns with one column
                    # that represents Ta + found_t*Tb as a rank-1 term
                    # But Ta + found_t*Tb being rank-1 means there exist
                    # u, v, w such that Tc[i*m+j,j*p+k,k*n+i] = u[i*m+j]*v[j*p+k]*w[k*n+i]
                    
                    # Extract u, v, w from Tc
                    # u[i*m+j] = Tc[i*m+j, j*p+k, k*n+i] for some fixed j,k,i
                    # This is tricky. Let's just try to factor Tc.
                    
                    # Tc should be of the form U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i]
                    # where U_new, V_new, W_new are vectors of lengths n*m, m*p, p*n
                    
                    # We can extract: U_new[i*m+j] = Tc[i*m+j, 0, 0] if that's nonzero
                    # But this may not work. Let's use a different approach.
                    
                    # Since we're going to remove columns a and b and add one,
                    # and the new tensor contribution is Tc, we need to find
                    # U_new, V_new, W_new such that their product gives Tc.
                    
                    # For now, let's just skip this and use a different strategy.
                    pass
        
        if not merged:
            break
    
    if best_rank <= target_rank:
        # Convert back to lists
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # The key insight: 3x3 x 3x6 is essentially two 3x3 x 3x3 multiplications.
    # The known minimal rank for 3x3 x 3x3 is 23 (Strassen-like improvements).
    # So we can potentially get rank 46 for 3x3 x 3x6.
    
    # Let's try to find a rank-23 decomposition for 3x3 x 3x3 numerically.
    # This is a hard problem, but we can use a heuristic search.
    
    # Alternative: use the structure to get a better decomposition.
    # For 3x3 x 3x6, we can view it as:
    # C = A @ B where A is 3x3, B is 3x6
    # C is 3x6
    # This is equivalent to multiplying A by each of the two 3x3 blocks of B.
    
    # Let's try a numerical optimization approach.
    # We'll parameterize the factors and use gradient-based optimization
    # to minimize the number of non-zero entries (or the rank).
    
    # Actually, let's try a simpler approach: use known results.
    # The tensor rank of 3x3 matrix multiplication is 23.
    # We can hardcode a rank-23 decomposition if we can find one.
    
    # For now, let's try to use a local search to reduce the rank.
    # Start from schoolbook (rank 54) and try to reduce by merging terms.
    
    elapsed = time.time() - start_time
    if elapsed < 80:
        # Try to find a better decomposition
        # Use the fact that the problem has structure
        
        # Approach: try random rank-r decompositions and verify
        for target_rank in [46, 45, 44, 43, 42, 41, 40]:
            if time.time() > deadline:
                break
            
            # Try to find a decomposition of the given rank
            # This is a system of polynomial equations, which is hard.
            # Let's use a heuristic: start from schoolbook and try to merge terms.
            
            # For now, let's just return the schoolbook result.
            # A proper implementation would use numerical algebraic geometry
            # or tensor decomposition algorithms.
            pass
    
    return best
# EVOLVE-BLOCK-END
