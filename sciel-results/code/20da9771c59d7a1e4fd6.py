"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Check each (i,j,k) entry: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # We check a set of test matrices
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C = A @ B
    # Compute C' = sum_c U_c * W_c * V_c (using the trace convention)
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each column c: outer product structure
    # C'[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    C_prime = np.zeros((n, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # wc is (p, n), we need wc[k*n+i] = wc[i,k] transposed
        # Actually W[k*n+i] means index k*n+i, so W reshaped to (n, p) gives W[i,k]
        wc_mat = wc.reshape(p, n)
        # C'[i,k] += wc_mat.T[i,k] * (uc @ vc)[i,k]
        C_prime += wc_mat.T * (uc @ vc)
    if not np.allclose(C_prime, C, atol=1e-9):
        return False
    # Also verify with a second random pair
    A2 = rng.randn(n, m)
    B2 = rng.randn(m, p)
    C2 = A2 @ B2
    C_prime2 = np.zeros((n, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        wc_mat = wc.reshape(p, n)
        C_prime2 += wc_mat.T * (uc @ vc)
    if not np.allclose(C_prime2, C2, atol=1e-9):
        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=100):
    """Try to reduce the rank of the decomposition using random linear combinations."""
    start_time = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    rng = np.random.RandomState(123)
    
    # Try to reduce rank by finding linear dependencies
    # The key insight: if we can find a non-trivial linear combination of the rank-1
    # tensors that sums to zero, we can eliminate one term.
    
    # Method: For each pair of columns, try to see if their sum (with some coefficients)
    # can be absorbed into other columns, reducing the effective rank.
    
    # Simpler approach: use the fact that for 3x3x6, the minimal rank is known to be
    # at most 55 (from literature). Let's try random search.
    
    # Actually, let's use a more systematic approach:
    # Start with the schoolbook and try to find linear relations among the rank-1 tensors.
    
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    # A rank-1 tensor is u_i * v_j * w_k.
    # We need sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.
    
    # Let's try to find a lower-rank decomposition using random initialization
    # and gradient-free optimization (random perturbation + acceptance if valid).
    
    for target_rank in range(55, 53, -1):  # Try ranks 55, 54
        if time.time() - start_time > max_time:
            break
        # Try random search for rank target_rank
        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
            # Initialize with random values
            U_try = rng.randn(n * m, target_rank)
            V_try = rng.randn(m * p, target_rank)
            W_try = rng.randn(p * n, target_rank)
            
            # Check if this decomposition works
            # We need: for all i,j,k: sum_c U_try[i*m+j,c]*V_try[j*p+k,c]*W_try[k*n+i,c] = 1
            # This is a system of n*m*p = 54 equations.
            # With target_rank variables per matrix, we have 3*target_rank*3 = 9*target_rank unknowns
            # For target_rank=55: 495 unknowns, 54 equations -> overdetermined but likely solvable
            
            # Actually, let's just verify
            valid = _verify_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p)
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist()

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank
    try:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=100)
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
