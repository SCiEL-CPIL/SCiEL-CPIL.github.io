"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition matches the tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    S = np.zeros_like(T)
    for c in range(U.shape[1]):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.allclose(S, T, atol=tol)

def _optimize_rank(U, V, W, n, m, p, target_rank, deadline, seed=0):
    """Try to optimize a decomposition to target_rank."""
    np.random.seed(seed)
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Initialize with random or from current
    if U.shape[1] >= target_rank:
        cols = np.random.choice(U.shape[1], target_rank, replace=False)
        U0 = U[:, cols].copy()
        V0 = V[:, cols].copy()
        W0 = W[:, cols].copy()
    else:
        U0 = np.random.randn(nm, target_rank) * 0.5
        V0 = np.random.randn(mp, target_rank) * 0.5
        W0 = np.random.randn(pn, target_rank) * 0.5
    
    def objective(params):
        U_ = params[:nm * target_rank].reshape(nm, target_rank)
        V_ = params[nm * target_rank:nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W_ = params[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
        S = np.zeros((nm, mp, pn))
        for c in range(target_rank):
            S += np.einsum('a,b,c->abc', U_[:, c], V_[:, c], W_[:, c])
        return np.sum((S - T) ** 2)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-6})
        if res.fun < 1e-10:
            U_ = res.x[:nm * target_rank].reshape(nm, target_rank)
            V_ = res.x[nm * target_rank:nm * target_rank + mp * target_rank].reshape(mp, target_rank)
            W_ = res.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
            if _verify(U_, V_, W_, n, m, p):
                return U_, V_, W_
    except:
        pass
    return None

def _greedy_remove(U, V, W, n, m, p, deadline):
    """Try to remove redundant terms."""
    rank = U.shape[1]
    removed = set()
    while rank - len(removed) > 1:
        if time.time() > deadline:
            break
        found = False
        for idx in range(rank):
            if idx in removed:
                continue
            mask = np.ones(rank, dtype=bool)
            mask[idx] = False
            U_rem = U[:, mask]
            V_rem = V[:, mask]
            W_rem = W[:, mask]
            if _verify(U_rem, V_rem, W_rem, n, m, p):
                U, V, W = U_rem, V_rem, W_rem
                removed.add(idx)
                rank = U.shape[1]
                found = True
                break
        if not found:
            break
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_sb
    
    # Try to reduce rank by optimization
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Try ranks from 28 upward
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        for seed in range(3):
            if time.time() > deadline:
                break
            result = _optimize_rank(best_U, best_V, best_W, n, m, p, target_rank, deadline, seed)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break
    
    # Greedy removal
    if time.time() < deadline:
        best_U, best_V, best_W = _greedy_remove(best_U, best_V, best_W, n, m, p, deadline)
        best_rank = best_U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
