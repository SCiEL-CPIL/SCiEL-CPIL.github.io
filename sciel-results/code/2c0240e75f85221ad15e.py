# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # safety margin before 600s kill

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for i.i.d. samples from discrete measure."""
    # Normalize weights
    w = weights / weights.sum()
    s = support
    n = len(s)
    # Pairwise sums s_i + s_j with weights w_i * w_j
    # P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
    # Vectorize: for each (k,l), need sum over (i,j) of w_i w_j * 1[s_i + s_j < 2*s_l - s_k]
    # Build matrix S = s_i + s_j (n x n), W = w_i * w_j (n x n)
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    # For each (k, l), threshold T = 2*s_l - s_k
    # T matrix: T[k, l] = 2*s_l - s_k  (shape n x n)
    T = 2.0 * s[None, :] - s[:, None]  # T[k,l] = 2*s_l - s_k
    # For each (k,l), inner = sum(S < T[k,l]) * W
    # This is n^2 outer * n^2 inner = n^4 total, but we can vectorize
    # For a fixed (k,l), inner = W.sum() where S < T[k,l]
    # We can do: for each k, l: np.sum(W * (S < T[k,l]))
    # To vectorize over all (k,l): create 4D comparison
    # S is (n,n) -> S[None, None, :, :] is (1,1,n,n)
    # T is (n,n) -> T[:, :, None, None] is (n,n,1,1)
    # Comparison: S < T gives (n,n,n,n) boolean
    # Then sum over last two axes weighted by W
    # W is (n,n) -> W[None, None, :, :]
    # result[k,l] = sum_{i,j} W[i,j] * (S[i,j] < T[k,l])
    # = sum over i,j of W[i,j] * (S[i,j] < T[k,l])
    # Let's build: cmp = S[None, None, :, :] < T[:, :, None, None]  shape (n,k,l,i,j)? 
    # Actually T shape is (n,n) where T[k,l]. S shape (n,n) where S[i,j].
    # cmp = (S[None, None, :, :] < T[:, :, None, None])  -> shape (n, n, n, n) = (k, l, i, j)
    # inner_sum[k,l] = sum_{i,j} W[i,j] * cmp[k,l,i,j]
    # = np.einsum or: reshape and multiply
    # W_b = W[None, None, :, :] shape (1,1,n,n)
    # inner = np.sum(W_b * cmp, axis=(2,3))  shape (n,n)
    # Then P = sum_{k,l} w[k]*w[l] * inner[k,l] = w[:,None]*w[None,:] * inner, sum all
    cmp = (S[None, None, :, :] < T[:, :, None, None])  # (k, l, i, j)
    W_b = W[None, None, :, :]
    inner = np.sum(W_b * cmp, axis=(2, 3))  # (k, l)
    P = np.sum(w[:, None] * w[None, :] * inner)
    return P

def _objective(x):
    """Objective to minimize: -P. x = [support..., weights...]"""
    n = _n_current
    s = x[:n]
    w = x[n:]
    # Enforce s >= 0, w >= 0 via penalty if slightly negative
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 1e-12)
    if w.sum() < 1e-9:
        return 0.0
    return -_compute_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    global _n_current
    best_support = np.arange(32, dtype=np.float64)
    best_weights = np.ones(32, dtype=np.float64) / 32
    best_P = _compute_P(best_support, best_weights)
    
    # Try various small support sizes with optimization
    for n in [3, 4, 5, 6, 7, 8]:
        if time.time() > _DEADLINE - 60:
            break
        _n_current = n
        # Initial guess: spread points, optimize weights
        # Try a few initializations
        inits = []
        # Init 1: uniform spacing
        s0 = np.linspace(0, n, n)
        w0 = np.ones(n) / n
        inits.append(np.concatenate([s0, w0]))
        # Init 2: clustered near 0 with one large
        s0b = np.concatenate([np.zeros(n-1), [n * 2.0]])
        w0b = np.ones(n) / n
        inits.append(np.concatenate([s0b, w0b]))
        # Init 3: geometric spacing
        s0c = np.geomspace(0.01, n * 1.5, n)
        w0c = np.ones(n) / n
        inits.append(np.concatenate([s0c, w0c]))
        
        for x0 in inits:
            if time.time() > _DEADLINE - 60:
                break
            try:
                res = minimize(_objective, x0, method='SLSQP',
                              bounds=[(0, 100)] * (2 * n),
                              options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
                if res.success or res.fun < 0:
                    s_opt = np.maximum(res.x[:n], 0.0)
                    w_opt = np.maximum(res.x[n:], 0.0)
                    if w_opt.sum() > 1e-9:
                        P_val = _compute_P(s_opt, w_opt)
                        if P_val > best_P:
                            best_P = P_val
                            best_support = s_opt.copy()
                            best_weights = w_opt.copy()
            except Exception:
                pass
    
    # Also try a hand-crafted bimodal that often works well
    # Put mass at 0 and at a large value M, plus some intermediate
    for n_extra in [2, 3, 4]:
        if time.time() > _DEADLINE - 30:
            break
        # Try: points [0, a, M] with weights [p, q, 1-p-q]
        # Optimize a, M, p, q
        def _obj3(v):
            a, M, p, q = v
            s = np.array([0.0, max(a, 0), max(M, 0)])
            w = np.array([max(p, 1e-6), max(q, 1e-6), max(1 - p - q, 1e-6)])
            return -_compute_P(s, w)
        
        for a0 in [1.0, 2.0, 5.0]:
            for M0 in [5.0, 10.0, 20.0]:
                for p0 in [0.5, 0.6, 0.7]:
                    for q0 in [0.1, 0.2, 0.3]:
                        if time.time() > _DEADLINE - 30:
                            break
                        try:
                            v0 = np.array([a0, M0, p0, q0])
                            res = minimize(_obj3, v0, method='Nelder-Mead',
                                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                            a, M, p, q = res.x
                            s = np.array([0.0, max(a, 0), max(M, 0)])
                            w = np.array([max(p, 1e-6), max(q, 1e-6), max(1 - p - q, 1e-6)])
                            P_val = _compute_P(s, w)
                            if P_val > best_P:
                                best_P = P_val
                                best_support = s.copy()
                                best_weights = w.copy()
                        except Exception:
                            pass
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    # Clamp tiny negatives
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    if best_weights.sum() < 1e-9:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
