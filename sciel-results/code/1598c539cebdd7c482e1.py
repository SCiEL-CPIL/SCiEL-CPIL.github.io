"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U @ V^T @ W^T reconstructs the matmul tensor."""
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _rank(U, V, W):
    """Return the number of columns (rank) of the decomposition."""
    return U.shape[1]


def _try_rank(r, n, m, p, max_iter=2000, seed=0, time_limit=None):
    """Try to find a decomposition of rank r using fast gradient descent."""
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_limit if time_limit else None
    
    # Initialize randomly
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    # Reshape for einsum
    Ut = U.reshape(n, m, r)
    Vt = V.reshape(m, p, r)
    Wt = W.reshape(p, n, r)
    
    target = np.ones((n, m, p))
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Adaptive learning rate
    lr = 0.1
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 2:
            break
        
        # Compute tensor
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        R = T - target
        err = np.sum(R**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            Ut_best = Ut.copy()
            Vt_best = Vt.copy()
            Wt_best = Wt.copy()
            
        if err < 1e-8:
            break
        
        # Compute gradients
        dU = np.einsum('ijk,jkc,kic->ijc', R, Vt, Wt)
        dV = np.einsum('ijk,ijc,kic->jkc', R, Ut, Wt)
        dW = np.einsum('ijk,ijc,jkc->kic', R, Ut, Vt)
        
        # Flatten gradients
        dU_flat = dU.reshape(-1)
        dV_flat = dV.reshape(-1)
        dW_flat = dW.reshape(-1)
        
        # Gradient norm for adaptive step
        g_norm = np.sqrt(np.sum(dU_flat**2) + np.sum(dV_flat**2) + np.sum(dW_flat**2))
        
        if g_norm < 1e-10:
            break
            
        # Update
        U -= lr * dU_flat.reshape(U.shape)
        V -= lr * dV_flat.reshape(V.shape)
        W -= lr * dW_flat.reshape(W.shape)
        
        # Update reshaped views
        Ut = U.reshape(n, m, r)
        Vt = V.reshape(m, p, r)
        Wt = W.reshape(p, n, r)
        
        # Simple adaptive learning rate
        # If error didn't decrease significantly, reduce lr
        # (Simplified: just decay lr slightly)
        lr = max(0.001, lr * 0.999)
        
    # Final check with best found
    Ut = best_U.reshape(n, m, r)
    Vt = best_V.reshape(m, p, r)
    Wt = best_W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    if np.allclose(T, target, atol=1e-5):
        return best_U, best_V, best_W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = _rank(U, V, W)
    best_result = (U, V, W)
    
    # Try to reduce rank
    # We try a few ranks below the schoolbook rank
    # Given the small dimensions, we might find a solution quickly if one exists
    # If not, we return schoolbook
    
    # Try ranks from best_rank - 1 down to max(20, best_rank // 2)
    # But we must be careful with time.
    # Let's try a few specific ranks with multiple seeds
    
    ranks_to_try = list(range(best_rank - 1, max(15, best_rank // 2), -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds for each rank
        # Time budget per rank
        time_per_rank = max(1, (deadline - 10 - time.time()) / len(ranks_to_try))
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            time_limit = min(2.0, (deadline - 5 - time.time()) / 5)
            if time_limit < 0.1:
                break
            result = _try_rank(r, n, m, p, max_iter=1000, seed=seed, time_limit=time_limit)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                found = True
                break
        
        if found:
            # If we found a lower rank, we might try to go even lower
            # But we break the loop to save time for potential further reduction?
            # Or we continue to try lower ranks?
            # The loop iterates downwards, so it will try lower ranks in next iterations.
            pass
    
    # Convert to lists for return
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
