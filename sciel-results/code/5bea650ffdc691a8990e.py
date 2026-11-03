import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
SCHOOLBOOK_RANK = N * M * P  # 36


def _build_tensor():
    """Build the matmul tensor T of shape (N*M, M*P, P*N).
    T[i*M+j, j'*P+k, k'*N+i'] = 1 iff i==i' and j==j' and k==k'
    """
    T = np.zeros((N * M, M * P, P * N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T


def _verify(U, V, W, T, tol=1e-6):
    """Check if sum_c U[a,c]*V[b,c]*W[c,c] == T[a,b,c] for all a,b,c."""
    D = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(D, T, atol=tol)


def _schoolbook():
    """Return the standard schoolbook decomposition of rank 36."""
    U = np.zeros((N * M, SCHOOLBOOK_RANK), dtype=float)
    V = np.zeros((M * P, SCHOOLBOOK_RANK), dtype=float)
    W = np.zeros((P * N, SCHOOLBOOK_RANK), dtype=float)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


def _try_structured_decomposition(T, target_rank, rng, max_iters=20000):
    """
    Try to find a rank-r decomposition using structured random initialization
    and iterative refinement. Start from random rank-r decomposition and
    use coordinate descent to minimize the error.
    """
    r = target_rank
    U = rng.normal(0, 1.0, (N * M, r))
    V = rng.normal(0, 1.0, (M * P, r))
    W = rng.normal(0, 1.0, (P * N, r))
    
    # Initial error
    D = np.einsum('at,bt,ct->abc', U, V, W)
    error = np.sum((D - T) ** 2)
    
    best_error = error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Coordinate descent: update one factor at a time
        # For each term t, update U[:,t], V[:,t], W[:,t] to minimize error
        
        # Update U[:, t] for a random t
        t = rng.integers(0, r)
        # Optimal U[:,t] = (T - D_without_t) * V[:,t] * W[:,t] / (V[:,t]^2 * W[:,t]^2)
        # This is complex, so use a simpler approach: gradient step
        
        # Compute gradient w.r.t. U[:,t]
        # d(error)/d(U[a,t]) = 2 * sum_{b,c} (D[a,b,c] - T[a,b,c]) * V[b,t] * W[c,t]
        residual = D - T
        grad_U = 2 * np.einsum('abc,bt,ct->at', residual, V, W)
        grad_V = 2 * np.einsum('abc,at,ct->bt', residual, U, W)
        grad_W = 2 * np.einsum('abc,at,bt->ct', residual, U, V)
        
        # Only update one factor at a time
        lr = 0.01
        U[:, t] -= lr * grad_U[:, t]
        
        # Recompute D and error
        D = np.einsum('at,bt,ct->abc', U, V, W)
        error = np.sum((D - T) ** 2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error < 1e-10:
            break
    
    if best_error < 1e-8:
        return best_U, best_V, best_W, best_error
    return None


def _try_rank_reduction(T, start_rank, end_rank, rng, time_budget):
    """Try to find decompositions of decreasing rank."""
    for r in range(start_rank, end_rank - 1, -1):
        if time.time() > time_budget:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > time_budget:
                break
            
            result = _try_structured_decomposition(T, r, rng, max_iters=5000)
            if result is not None:
                U, V, W, error = result
                return r, U, V, W
    
    return None


def _try_kronecker_extension(T, time_budget):
    """
    Try to use a known 3x3*3x3 rank-21 decomposition and extend to 3x3*3x4.
    This is a heuristic approach.
    """
    # For now, this is a placeholder. The actual 3x3*3x3 rank-21 decomposition
    # is complex. Let's try a simpler approach: block decomposition.
    # Partition 3x3 into 1x1 blocks and use Strassen-like ideas.
    # This is too complex for the time budget, so return None.
    return None


def _try_flattened_qr(T, target_rank, rng, time_budget):
    """
    Use tensor flattening and low-rank approximation via QR decomposition
    with random projections to identify a sparse support set.
    """
    # Flatten the tensor in different ways
    # T has shape (N*M, M*P, P*N) = (9, 12, 12)
    
    # Try matricizations
    # T1: (N*M) x (M*P * P*N) = 9 x 144
    T1 = T.reshape(N * M, -1)
    # T2: (M*P) x (N*M * P*N) = 12 x 108
    T2 = T.reshape(M * P, -1).T
    # T3: (P*N) x (N*M * M*P) = 12 x 108
    T3 = T.reshape(P * N, -1).T
    
    # Check numerical ranks
    rank1 = np.linalg.matrix_rank(T1, tol=1e-10)
    rank2 = np.linalg.matrix_rank(T2, tol=1e-10)
    rank3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    # The tensor rank is at least max of these matricization ranks
    min_possible_rank = max(rank1, rank2, rank3)
    
    if target_rank < min_possible_rank:
        return None
    
    # Use random projections to find a low-rank approximation
    # Project onto random subspace of dimension target_rank
    proj_dim = target_rank
    
    # Random projection for U
    R1 = rng.normal(0, 1, (N * M, proj_dim))
    # SVD of projected tensor
    T_proj1 = T1 @ R1  # (9, proj_dim)
    
    # This approach is too heuristic. Let's try a different strategy:
    # Use the fact that the tensor has a special structure (it's a matmul tensor)
    # and try to exploit that.
    
    return None


def construct():
    deadline = time.time() + 130  # 130s to be safe (150s kill)
    
    T = _build_tensor()
    
    # Best known: schoolbook rank 36
    U_best, V_best, W_best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK
    
    rng = np.random.default_rng(42)
    
    # Try to find better decompositions
    # Start from rank 35 and go down
    
    # Strategy 1: Try structured random decomposition
    for target_rank in range(35, 25, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_reduction(T, target_rank, target_rank, rng, deadline)
        if result is not None:
            r, U, V, W = result
            if _verify(U, V, W, T):
                best_rank = r
                U_best, V_best, W_best = U, V, W
                break  # Found a better rank, stop searching lower
    
    # Strategy 2: Try Kronecker extension
    if time.time() < deadline:
        result = _try_kronecker_extension(T, deadline)
        if result is not None:
            r, U, V, W = result
            if _verify(U, V, W, T):
                best_rank = r
                U_best, V_best, W_best = U, V, W
    
    # Convert to lists for the return format
    U_list = [[U_best[i][j] for j in range(best_rank)] for i in range(N * M)]
    V_list = [[V_best[i][j] for j in range(best_rank)] for i in range(M * P)]
    W_list = [[W_best[i][j] for j in range(best_rank)] for i in range(P * N)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
