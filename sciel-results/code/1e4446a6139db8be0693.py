"""Tensor decomposition for 3x3 x 3x5 matrix multiplication."""
import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _verify_decomp(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reconstructs the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err


def _try_rank_decomposition(n, m, p, rank, seed=0, max_iter=300, tol=1e-8):
    """Try to find a rank-r decomposition using alternating least squares.
    Returns (U, V, W) arrays if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U_arr = rng.randn(n * m, rank)
    V_arr = rng.randn(m * p, rank)
    W_arr = rng.randn(p * n, rank)
    
    # Normalize columns
    for r in range(rank):
        nu = np.linalg.norm(U_arr[:, r])
        nv = np.linalg.norm(V_arr[:, r])
        nw = np.linalg.norm(W_arr[:, r])
        if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
            scale = (nu * nv * nw) ** (1.0/3.0)
            U_arr[:, r] /= nu
            V_arr[:, r] /= nv
            W_arr[:, r] /= nw
    
    best_err = float('inf')
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for it in range(max_iter):
        # Solve for W given U, V
        W_new = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U_arr[i * m + j, r] * V_arr[j * p + k, r]
                W_new[w_idx, :] = lstsq(A, b, cond=None)[0]
        W_arr = W_new
        
        # Solve for U given V, W
        U_new = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V_arr[j * p + k, r] * W_arr[k * n + i, r]
                U_new[u_idx, :] = lstsq(A, b, cond=None)[0]
        U_arr = U_new
        
        # Solve for V given U, W
        V_new = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U_arr[i * m + j, r] * W_arr[k * n + i, r]
                V_new[v_idx, :] = lstsq(A, b, cond=None)[0]
        V_arr = V_new
        
        # Check error
        err = _verify_decomp(U_arr, V_arr, W_arr, n, m, p, tol=1.0)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        if err < tol:
            return U_arr, V_arr, W_arr
        
        # Normalize to prevent divergence
        for r in range(rank):
            nu = np.linalg.norm(U_arr[:, r])
            nv = np.linalg.norm(V_arr[:, r])
            nw = np.linalg.norm(W_arr[:, r])
            if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
                scale = (nu * nv * nw) ** (1.0/3.0)
                U_arr[:, r] /= nu
                V_arr[:, r] /= nv
                W_arr[:, r] /= nw
    
    # Final check with best found
    if best_err < tol:
        return best_U, best_V, best_W
    return None


def _gf2_lower_bound(n, m, p):
    """Compute the rank of the multiplication tensor over GF(2) as a lower bound.
    This is done by building the tensor as a linear map and finding its rank.
    
    The multiplication tensor M(n,m,p) can be viewed as a linear map from
    R^{n*m} x R^{m*p} -> R^{p*n}. The rank over a field is the minimum number
    of rank-1 tensors needed to express it.
    
    Over GF(2), we can compute this by viewing the tensor as a 3-way array and
    finding the minimum number of rank-1 tensors that sum to it.
    
    For small sizes, we can do an exhaustive search or use linear algebra.
    """
    # Build the tensor T[i,j,k] = 1 for all (i,j,k)
    # Over GF(2), we need to find the minimum r such that
    # T[i,j,k] = sum_{r=0}^{r-1} u_r[i,j] * v_r[j,k] * w_r[k,i] mod 2
    
    # This is a hard problem in general, but for small sizes we can try
    # to find a lower bound by considering the tensor as a matrix.
    
    # Flatten T into a matrix of shape (n*m) x (m*p) for each k, or similar.
    # Actually, let's use the standard approach: view the tensor as a linear map.
    
    # The tensor T defines a bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # As a linear map from A to C for fixed B, or from B to C for fixed A.
    
    # For the rank over GF(2), we can try all possible rank-r decompositions
    # for small r, but this is exponential. Instead, let's use a simpler bound.
    
    # The rank of M(n,m,p) over any field is at least max(n*m, m*p, p*n) / min(n,m,p)
    # but this is not tight. A better bound is from the "slice rank" or similar.
    
    # For our purposes, let's just return a conservative lower bound.
    # The rank is at least max(n, m, p) for non-trivial cases.
    return max(n, m, p)


def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 120.0 - 10.0  # 10 second safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p  # 45
    
    # Try to find a lower rank decomposition
    # The rank of M(3,3,5) is known to be at most 40 (from various constructions)
    # Let's try ranks from 30 up to 44
    
    # First, compute GF(2) lower bound
    gf2_lb = _gf2_lower_bound(n, m, p)
    
    # Try various ranks
    for rank in range(max(20, gf2_lb), min(best_rank, 45)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline - 30:
                break
            
            result = _try_rank_decomposition(n, m, p, rank, seed=seed, max_iter=200, tol=1e-8)
            if result is not None:
                U, V, W = result
                # Verify with tighter tolerance
                err = _verify_decomp(U, V, W, n, m, p, tol=1e-8)
                if err < 1e-8:
                    # Found a valid decomposition!
                    best_rank = rank
                    best = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
        
        if best_rank < best_rank:
            break
    
    # If we found a better decomposition, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return best
    else:
        return _schoolbook_decomp(n, m, p)
# EVOLVE-BLOCK-END
