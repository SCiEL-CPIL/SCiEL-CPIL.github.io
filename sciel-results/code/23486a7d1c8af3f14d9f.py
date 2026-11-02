"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import svd
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(T, U, V, W, tol=1e-9):
    """Verify that U, V, W with the same number of columns (rank) reconstruct T.
    T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
    """
    rank = U.shape[1]
    # Efficient check: for each rank element, add outer product
    # But for speed, use einsum
    reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol)

def _try_svd_decomposition(n, m, p, max_rank_to_try=60, num_trials=50, time_budget=100):
    """Attempt to find a low-rank decomposition via random projections + SVD.
    
    Strategy:
    1. Randomly project the tensor to a smaller 3-tensor.
    2. Find the "rank" by checking if a truncated SVD of the mode-2 flattening
       reconstructs the projected tensor.
    3. Use that rank to try a random linear combination approach.
    
    Actually, a more direct approach:
    - Try ranks from a lower bound up to n*m*p.
    - For each rank, try to find a decomposition using random initialization
      and iterative refinement (like a simplified tensor train or CP approximation).
    """
    T = _build_tensor(n, m, p)
    shape = T.shape
    total_elements = np.prod(shape)
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Also rank >= n*m*p / max dimension... not straightforward.
    # A known lower bound is rank >= n*m*p / (n+m+p-2) for some cases, but let's use 30.
    min_rank = max(n * m, m * p, p * n)
    max_rank = n * m * p
    
    print(f"Tensor shape: {shape}, min_rank: {min_rank}, max_rank: {max_rank}")
    
    # First, let's try a few SVD-based approaches to estimate rank.
    # Flatten to mode-2: (n*m * m*p, p*n)
    T2 = T.transpose(0, 1, 2).reshape(n * m * m * p, p * n)
    U2, S2, Vt2 = svd(T2, full_matrices=False)
    
    # Check how many singular values are non-negligible
    print(f"Singular values (top 40): {S2[:40]}")
    
    # Find the effective rank
    for r in range(1, len(S2) + 1):
        if S2[r - 1] < 1e-10:
            print(f"Effective SVD rank (mode-2 flattening): {r - 1}")
            break
    
    # Let's try a different approach: use the fact that the tensor has a
    # structured form. We can try to find a decomposition by solving a
    # system of equations for a given rank.
    
    # For a rank-r decomposition, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a nonlinear system. Let's try a random initialization + gradient descent.
    
    def random_decomposition(rank, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        return U, V, W
    
    def compute_loss(U, V, W):
        recon = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((recon - T) ** 2)
    
    def gradient_step(U, V, W, lr=0.001):
        # Gradient of loss w.r.t. U
        recon = np.einsum('ak,bk,ck->abc', U, V, W)
        diff = recon - T
        grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
        grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
        grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
        
        U_new = U - lr * grad_U
        V_new = V - lr * grad_V
        W_new = W - lr * grad_W
        return U_new, V_new, W_new
    
    best_result = None
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            print("Time budget exceeded in SVD approach")
            break
        
        # Try a few different ranks
        for rank in range(max(min_rank, 30), min(max_rank + 1, 60)):
            if time.time() - start_time > time_budget:
                break
                
            U, V, W = random_decomposition(rank, seed=trial * 100 + rank)
            
            # Run a few gradient steps
            for step in range(1000):
                U, V, W = gradient_step(U, V, W, lr=0.01 / (1 + step / 100))
                
                if step % 100 == 0:
                    loss = compute_loss(U, V, W)
                    if loss < 1e-8:
                        if best_result is None or rank < best_result['rank']:
                            best_result = {
                                'rank': rank,
                                'U': U.copy(),
                                'V': V.copy(),
                                'W': W.copy(),
                                'loss': loss
                            }
                        break
            else:
                loss = compute_loss(U, V, W)
                if loss < 1e-6:
                    if best_result is None or rank < best_result['rank']:
                        best_result = {
                            'rank': rank,
                            'U': U.copy(),
                            'V': V.copy(),
                            'W': W.copy(),
                            'loss': loss
                        }
    
    if best_result is not None:
        print(f"Found decomposition with rank {best_result['rank']} and loss {best_result['loss']}")
        return best_result
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Try to find a better decomposition
    try:
        result = _try_svd_decomposition(n, m, p, max_rank_to_try=60, num_trials=20, time_budget=90)
        if result is not None and result['rank'] < n * m * p:
            # Convert to list of lists for the required output format
            U = result['U'].tolist()
            V = result['V'].tolist()
            W = result['W'].tolist()
            return {'U': U, 'V': V, 'W': W}
    except Exception as e:
        print(f"Exception in SVD approach: {e}")
    
    # Fallback to schoolbook
    if time.time() > deadline:
        # Emergency: return immediately with minimal work
        pass
    
    sb = _schoolbook_decomposition(n, m, p)
    U = sb['U'].tolist()
    V = sb['V'].tolist()
    W = sb['W'].tolist()
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
