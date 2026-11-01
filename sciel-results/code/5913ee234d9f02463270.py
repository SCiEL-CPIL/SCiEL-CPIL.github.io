"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with the standard multiplication tensor
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1.0 if (j == j and k == k) else 0.0
                # Wait, the multiplication tensor T_{i,j,k} = 1 if the indices match in the right way
                # Actually for matrix mult: (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
                # So the tensor is T[i,j,k] which contributes to (A*B)[i,k] via A[i,j]*B[j,k]
                # The tensor for the bilinear form is: for each (i,j,k), the term is A[i,j]*B[j,k]
                # So T[i,j,k] should be 1 for all valid (i,j,k) since it's just the product term
                # Actually no - the tensor decomposition represents the bilinear map
                # Let me reconsider: the multiplication tensor M has M[i,j,k] = 1 for all i,j,k
                # because (A*B)[i,k] = sum_j A[i,j] * B[j,k], and each term A[i,j]*B[j,k] 
                # corresponds to a rank-1 tensor with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
                # So the full tensor is sum over all (i,j,k) of e_i ⊗ e_j ⊗ e_k
                # Thus M[i,j,k] = 1 for all i,j,k in range
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_rank_r(n, m, p, r, time_budget=30.0):
    """Try to find a rank-r decomposition using random initialization and verification."""
    start = time.time()
    np.random.seed(42)
    # Try random initializations
    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        # Random rank-r decomposition
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Check if this decomposition exactly reproduces the tensor
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err < 1e-9:
            # Convert to list format
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
            return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    safety_margin = 10.0  # seconds before deadline
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 to 53
    for r in range(27, best_rank):
        if time.time() + safety_margin > deadline:
            break
        time_budget = min(30.0, deadline - safety_margin - time.time())
        if time_budget <= 5.0:
            break
        result = _try_rank_r(n, m, p, r, time_budget=time_budget)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = r
            break  # Found the lowest possible rank in this range
    
    return best_result
# EVOLVE-BLOCK-END
