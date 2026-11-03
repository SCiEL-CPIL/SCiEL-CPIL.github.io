"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE_MARGIN = 10  # seconds before the 150s kill
START_TIME = time.time()


def _time_left():
    return (START_TIME + 150 - DEADLINE_MARGIN) - time.time()


def _verify(U, V, W):
    """Check exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    n, m, p = DIMENSIONS
    # Build tensor T from U,V,W
    # T has shape (n, m, p)
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    rank = len(U[0]) if U else 0
    # Efficient check: for each (i,j,k), compute sum
    # Use numpy for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # This is a tensor contraction. We can compute it as:
    # For each c: contribution to T[i,j,k] is U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # Sum over c.
    
    # Efficient way: einsum
    # U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c] summed over c
    # = einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor
    # where (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the tensor for matrix mult is T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Wait, no. The matrix multiplication tensor T has T[i,j,k] = 1 iff the (i,k) entry of AB
    # involves A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < 1e-9


def _try_decompose(rank, max_iters=2000, seed=0):
    """Try to find a rank-rank decomposition via random initialization + local search."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Target tensor
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    # Initialize U, V, W randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize columns to avoid scale issues
    for c in range(rank):
        norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if norm > 1e-12:
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
    
    best_diff = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        # Compute current tensor
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff = T_target - T_cur
        diff_norm = np.sum(diff**2)
        
        if diff_norm < best_diff:
            best_diff = diff_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if diff_norm < 1e-12:
            break
        
        # Gradient descent on U, V, W
        # dL/dU[i,j,c] = sum_{k} diff[i,j,k] * V_3d[j,k,c] * W_3d[k,i,c]
        # dL/dV[j,k,c] = sum_{i} diff[i,j,k] * U_3d[i,j,c] * W_3d[k,i,c]
        # dL/dW[k,i,c] = sum_{j} diff[i,j,k] * U_3d[i,j,c] * V_3d[j,k,c]
        
        # dU[i,j,c] = sum_k diff[i,j,k] * V_3d[j,k,c] * W_3d[k,i,c]
        dU_3d = np.einsum('ijk,jkc,kic->ijc', diff, V_3d, W_3d)
        dV_3d = np.einsum('ijk,ijc,kic->jkc', diff, U_3d, W_3d)
        dW_3d = np.einsum('ijk,ijc,jkc->kic', diff, U_3d, V_3d)
        
        # Learning rate with decay
        lr = 0.01 / (1 + it * 0.001)
        
        U -= lr * dU_3d.reshape(n * m, rank)
        V -= lr * dV_3d.reshape(m * p, rank)
        W -= lr * dW_3d.reshape(p * n, rank)
        
        # Periodic restart if stuck
        if it % 200 == 199 and diff_norm > 1e-6:
            # Add noise
            U += 0.1 * rng.randn(n * m, rank)
            V += 0.1 * rng.randn(m * p, rank)
            W += 0.1 * rng.randn(p * n, rank)
    
    # Check if we found a solution
    if best_diff < 1e-10:
        # Convert to integer/half-integer if possible
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list, True
    
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actual minimal rank for 3x4x6 is likely higher.
    # Let's try a range of ranks.
    
    best_result = None
    
    # First, always have the schoolbook solution as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    # But we need to be time-efficient. Try a few key ranks.
    ranks_to_try = list(range(lower_bound, min(schoolbook_rank, 40)))
    
    for rank in ranks_to_try:
        if _time_left() < 15:
            break
        
        # Try multiple seeds for this rank
        found = False
        for seed in range(5):
            if _time_left() < 10:
                break
            U, V, W, success = _try_decompose(rank, max_iters=500, seed=seed)
            if success:
                # Verify
                if _verify(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    return best_result
                found = True
                break
        
        if found:
            break
    
    # If no better solution found, return schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
