"""Matrix multiplication 2x3x10 - rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify tensor decomposition."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ur[i,j], Vr[j,k] * Wr[k,i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_numerical(R, n, m, p, max_iters=500, base_lr=0.02, seed=42, time_limit=6.0):
    """Try to find rank-R decomposition via gradient descent."""
    rng = np.random.RandomState(seed)
    start = time.time()
    
    U = rng.randn(n*m, R) * 0.3
    V = rng.randn(m*p, R) * 0.3
    W = rng.randn(p*n, R) * 0.3
    
    def loss_and_grads():
        loss = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = val - 1.0
                    loss += err * err
                    gU[i*m+j] += 2 * err * (V[j*p+k] * W[k*n+i])
                    gV[j*p+k] += 2 * err * (U[i*m+j] * W[k*n+i])
                    gW[k*n+i] += 2 * err * (U[i*m+j] * V[j*p+k])
        return loss, gU, gV, gW
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = base_lr
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        loss, gU, gV, gW = loss_and_grads()
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_loss < 1e-12:
            break
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        lr *= 0.995
    
    if best_loss < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Schoolbook fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = 130.0
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = 30
    lower_bound = max(n*m, m*p, p*n)
    
    for R in range(lower_bound, best_rank):
        if time.time() - start_time > deadline:
            break
        for seed in range(10):
            if time.time() - start_time > deadline:
                break
            remaining = deadline - (time.time() - start_time)
            if remaining < 2.0:
                break
            result = _try_rank_numerical(R, n, m, p, max_iters=400, base_lr=0.03,
                                         seed=seed, time_limit=min(4.0, remaining - 1.0))
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
