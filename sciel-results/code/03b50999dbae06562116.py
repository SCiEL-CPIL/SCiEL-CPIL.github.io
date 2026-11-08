import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i*p+k, j*p+k, i*m+j] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait, let me re-read the convention.
    
    W[k*n+i, r] - so W has p*n rows, indexed by (k, i) where k is col of B/C, i is row of A/C
    U[i*m+j, r] - U has n*m rows, indexed by (i, j) where i is row of A, j is col of A
    V[j*p+k, r] - V has m*p rows, indexed by (j, k) where j is row of B, k is col of B
    
    The matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[C_index, A_index, B_index] = 1 when C_index=(i,k), A_index=(i,j), B_index=(j,k)
    i.e., T[k*n+i, i*m+j, j*p+k] = 1
    
    And we need: T[a, b, c] = sum_r U[b, r] * V[c, r] * W[a, r]
    where a = k*n+i (C index), b = i*m+j (A index), c = j*p+k (B index)
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    # Compute the decomposed tensor
    # T_decomp[a, b, c] = sum_r U[b,r] * V[c,r] * W[a,r]
    # This is a tensor of order 3
    # We can compute it as: for each r, outer product of W[:,r], U[:,r], V[:,r]
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
    
    return np.allclose(T_decomp, T, atol=1e-6)

def _try_rank(rank, n=3, m=3, p=6, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank using random search with local refinement."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Build target tensor
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute error: ||T_decomp - T||_F
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_decomp - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            return best_U, best_V, best_W
        
        # Random perturbation
        scale = max(0.1, 0.5 * best_err)
        U += scale * rng.randn(n * m, rank) * 0.1
        V += scale * rng.randn(m * p, rank) * 0.1
        W += scale * rng.randn(p * n, rank) * 0.1
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_result = (U_fb, V_fb, W_fb, schoolbook_rank)
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for matrix multiplication tensor is related to
    # the number of distinct "patterns". For 3x3 x 3x6, let's try ranks from 
    # a reasonable lower bound upward.
    
    # A simple lower bound: the tensor has 54 nonzero entries in schoolbook,
    # but the rank could be lower. Let's try from 30 upward.
    
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor for m x n * n x p has rank at most m*n*p.
    # For 3x3 * 3x6 = 54.
    
    # Let's try a structured approach: since p=6=2*3, maybe we can exploit
    # the structure. But first, let's just try random search at various ranks.
    
    # Try ranks from a lower bound to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But that's a very weak bound. Let's try from 30 to 53.
    
    for rank in range(30, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try a few seeds
        found = None
        for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline:
                break
            result = _try_rank(rank, n, m, p, max_iters=2000, seed=seed)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W, rank)
                    break
        
        if found is not None:
            best_result = found
            # Keep trying lower ranks
            continue
        else:
            # If we can't find rank r, likely can't find lower either
            # But let's try one more lower rank before giving up
            break
    
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
