"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}}
    ]
    
    for j_block in range(4):
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + lv][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[lu * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + (2 + lv)][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[(2 + lu) * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we need A[i,j]*B[j,k]
    # The decomposition should satisfy: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i,j,k) combinations that shouldn't contribute, the sum should be 0.
    
    # Actually, let's verify by checking that for random matrices A (n x m) and B (m x p),
    # C = A @ B equals the reconstructed C from the decomposition.
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C using the decomposition
    C_reconstructed = np.zeros((n, p))
    for r in range(U.shape[1]):
        # u_r is vector of length n*m
        u_r = U[:, r].reshape(n, m)
        # v_r is vector of length m*p  
        v_r = V[:, r].reshape(m, p)
        # w_r is vector of length p*n
        w_r = W[:, r].reshape(p, n)
        
        # C_reconstructed += w_r.T @ (u_r * v_r) ... no, that's not right
        # The tensor is sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # So C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        for i in range(n):
            for k in range(p):
                wk_i = W[k * n + i, r]
                if wk_i != 0:
                    for j in range(m):
                        u_val = U[i * m + j, r]
                        v_val = V[j * p + k, r]
                        C_reconstructed[i, k] += u_val * v_val * wk_i
    
    return np.allclose(C_expected, C_reconstructed, atol=1e-10)

def _try_rank_search(target_rank, max_iterations=50, seed=0):
    """Try to find a rank-r decomposition using ALS with rounding."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U_size = n * m  # 12
    V_size = m * p  # 32
    W_size = p * n  # 24
    
    # Initialize with random Gaussian
    U = rng.randn(U_size, target_rank) * 0.1
    V = rng.randn(V_size, target_rank) * 0.1
    W = rng.randn(W_size, target_rank) * 0.1
    
    # The tensor we want to match: T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We'll use a weighted least squares approach
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for iteration in range(max_iterations):
        if time.time() > _START_TIME + _DEADLINE - 10:
            break
        
        # Simpler: use gradient descent with projection
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target is 1
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - 1.0) ** 2
        
        # Also penalize non-zero entries where they shouldn't be
        # But for matrix multiplication tensor, only (i,j,k) with matching j matter
        # Actually all (i,j,k) are valid since j is shared
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        # Gradient step
        lr = 0.001
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(target_rank):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    err = val - 1.0
                    for r in range(target_rank):
                        grad_U[idx_u, r] += err * V[idx_v, r] * W[idx_w, r]
                        grad_V[idx_v, r] += err * U[idx_u, r] * W[idx_w, r]
                        grad_W[idx_w, r] += err * U[idx_u, r] * V[idx_v, r]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Check if we've found an exact solution
        if error < 1e-6:
            # Verify and return
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the guaranteed rank-88 solution
    rank88_result = _build_rank88()
    
    # Try to find a better solution with less time
    time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
    if time_remaining > 20:
        # Try rank 80
        result = _try_rank_search(80, max_iterations=30, seed=0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
    if time_remaining > 15:
        # Try rank 78
        result = _try_rank_search(78, max_iterations=25, seed=1)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
    if time_remaining > 10:
        # Try rank 76
        result = _try_rank_search(76, max_iterations=20, seed=2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return the guaranteed rank-88 solution
    return rank88_result
# EVOLVE-BLOCK-END
