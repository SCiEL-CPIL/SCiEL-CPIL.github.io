"""Matrix multiplication 3x3x5 - attempt to find a lower rank decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
N, M, P = 3, 3, 5
RANK_SCHOOLBOOK = N * M * P  # 45

# Build the matmul tensor T of shape (N*M, M*P, P*N)
# T[i*m+j, j*p+k, k*n+i] = 1

def build_tensor():
    T = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M+j, j*P+k, k*N+i] = 1.0
    return T

T = build_tensor()

def verify_decomposition(U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor T."""
    rank = U.shape[1]
    # Compute sum_c U[:,c] V[:,c]^T W[:,c]^T and compare to T
    residual = np.zeros((N*M, M*P, P*N))
    for c in range(rank):
        residual += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return np.max(np.abs(residual - T)) < tol

def schoolbook():
    rank = RANK_SCHOOLBOOK
    U = np.zeros((N*M, rank))
    V = np.zeros((M*P, rank))
    W = np.zeros((P*N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j, col] = 1
                V[j*P+k, col] = 1
                W[k*N+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def try_rank(r, max_iters=50000, deadline=140.0):
    """Try to find a rank-r decomposition via random search with local refinement."""
    start_time = time.time()
    
    # Random initialization with small integers
    best_residual = np.inf
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        U = np.random.randint(-2, 3, (N*M, r)) * 0.5
        V = np.random.randint(-2, 3, (M*P, r)) * 0.5
        W = np.random.randint(-2, 3, (P*N, r)) * 0.5
        
        residual = np.max(np.abs(np.einsum('ac,bc,dc->abd', U, V, W) - T))
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if residual < 1e-9:
                break
    
    if best_residual < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Try a few lower ranks
    for r in range(40, 45):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iters=20000, deadline=deadline - 5)
        if result is not None:
            U, V, W = result
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
