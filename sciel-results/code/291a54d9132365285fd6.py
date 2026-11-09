import time
import numpy as np
from scipy.linalg import null_space, qr

DIMENSIONS = [4, 4, 9]
n, m, p = DIMENSIONS
# Tensor shape: (n, m, p), T[i,j,k] = 1 if i,j,k are independent indices
# Actually, the matrix multiplication tensor T is indexed by (i,j,k) where:
#   i in [0,n) for output row
#   j in [0,m) for input columns / intermediate
#   k in [0,p) for output columns
# T[i,j,k] = 1 (the tensor is the identity-like structure of matrix mult)
# More precisely, T is the tensor such that (A * B)[i,k] = sum_j A[i,j] * B[j,k]
# So the tensor T[i,j,k] = delta (it's 1 for all i,j,k in the product structure)
# Wait, no. The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
# Actually, the tensor is T[i,j,k] = 1 for all valid i,j,k, representing the
# bilinear form A[i,j] * B[j,k] -> C[i,k].

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.ones((n, m, p), dtype=float)
    return T

def _tucker_decompose(T, max_rank=200, time_limit=120, seed=42):
    """
    Attempt to find a low-rank decomposition of the matrix multiplication tensor.
    We use a structured approach: since the MM tensor has special structure,
    we can use the known rank bounds and attempt specific decompositions.
    
    For 4x4 * 4x9, the rank is known to be at most n*m*p = 144 (schoolbook).
    We try to find a lower rank decomposition.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    deadline = start_time + time_limit - 5  # safety margin
    
    T = _build_tensor(n, m, p)
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition as fallback
    rank = best_rank
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
    
    # Verify schoolbook
    def _verify(U, V, W, T, tol=1e-9):
        n, m, p = T.shape
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        rank = U.shape[1]
        # Compute the decomposed tensor: sum_r U3[:,:,r] * V3[:,:,r] * W3[:,:,r]
        # But W is indexed differently: W[k*n+i, r], so W3 should be (p, n, rank)
        # Actually, let's think about this more carefully.
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So we need to compute for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        result = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # Outer product: u_col[i,j] * v_col[j,k] * w_col[k,i]
            # This is a 3D outer product with specific index structure
            # We can compute this as: for each i,j,k, sum over r
            # Let's do it via einsum
            # u_col is (n,m), v_col is (m,p), w_col is (p,n)
            # We want T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            # = einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
            result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col, optimize=True)
        return np.max(np.abs(result - T)) < tol
    
    if _verify(U, V, W, T):
        best_U, best_V, best_W = U, V, W
    
    # Try Strassen-like block decompositions
    # For 4x4 * 4x9, we can try to exploit block structure
    # 4x4 can be seen as 2x2 blocks of 2x2, and 4x9 as 2x5 blocks of 2x...
    # This gets complex. Let's try a random restart approach with optimization.
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for 4x4*4x9: at least n*p = 36 (or m*p=36, or n*m=16)
    # Actually, the rank is at least max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But more precisely, for MM(n,m,p), rank >= n*m*p / (something)
    
    # Let's try a few specific ranks
    for target_rank in [36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 108, 120, 144]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
        
        # Try random initialization with several restarts
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Check if this random decomposition is close to T
            if _verify(U_rand, V_rand, W_rand, T, tol=1e-6):
                found = True
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
        
        if found:
            break
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback to schoolbook
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    start_time = time.time()
    deadline = start_time + 120 - 10  # 110 seconds to be safe
    
    # Try the tucker decomposition approach
    try:
        result = _tucker_decompose(n, m, p, time_limit=120)
        if time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
