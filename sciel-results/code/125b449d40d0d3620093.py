import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
TOTAL_TIME = 150.0
SAFETY_MARGIN = 15.0

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return 0.0
    # Normalize to l1 = 1 for the ratio (scale-invariant under constraint)
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    linf = np.max(conv)
    if linf <= 1e-12:
        return 0.0
    return l2_sq / linf

def _is_sidon_extension(S, k, sums):
    for s in S:
        if s + k in sums:
            return False
    if k + k in sums:
        return False
    return True

def _greedy_sidon(N, order):
    S = []
    sums = set()
    for k in order:
        if _is_sidon_extension(S, k, sums):
            S.append(k)
            for s in S:
                sums.add(s + k)
            sums.add(k + k)
    return S

def find_best_sidon(N, time_budget):
    deadline = time.time() + time_budget
    best_set = _greedy_sidon(N, list(range(N)))
    it = 0
    while time.time() < deadline:
        order = list(range(N))
        random.shuffle(order)
        S = _greedy_sidon(N, order)
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= 12:  # reasonable max
                break
        it += 1
    return best_set

def optimize_weights_on_support(support, time_budget):
    """Optimize weights on the given support to maximize the ratio."""
    K = len(support)
    if K == 0:
        return None
    
    def objective(w):
        # w are weights on the support, sum to 1, w >= 0
        h = np.zeros(N)
        h[support] = w
        return -compute_ratio(h)  # minimize negative
    
    def objective_smooth(w, beta=50.0):
        h = np.zeros(N)
        h[support] = w
        l2_sq = np.sum(w**2)
        conv = np.convolve(h, h, mode='full')
        # soft-max
        conv_pos = np.maximum(conv, 1e-12)
        logsumexp = np.log(np.sum(np.exp(beta * conv_pos))) / beta
        return -(l2_sq / logsumexp)
    
    # Initial: equal weights
    w0 = np.ones(K) / K
    
    best_w = w0.copy()
    best_val = -objective(w0)
    
    deadline = time.time() + time_budget
    
    # Try L-BFGS-B with the smooth objective
    for beta in [10.0, 50.0, 200.0]:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective_smooth, w0, args=(beta,), method='L-BFGS-B',
                          bounds=[(0, None)] * K,
                          constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success or res.fun < -best_val:
                w_opt = res.x
                # Project to simplex
                w_opt = np.maximum(w_opt, 0)
                w_opt /= np.sum(w_opt)
                val = -objective(w_opt)
                if val > best_val:
                    best_val = val
                    best_w = w_opt.copy()
                w0 = w_opt.copy()  # warm start
        except Exception:
            pass
    
    # Final polish with exact objective using a local search
    if time.time() < deadline:
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline:
                break
            w_pert = best_w + 0.01 * np.random.randn(K)
            w_pert = np.maximum(w_pert, 0)
            s = np.sum(w_pert)
            if s > 0:
                w_pert /= s
            val = -objective(w_pert)
            if val > best_val:
                best_val = val
                best_w = w_pert.copy()
    
    h = np.zeros(N)
    h[support] = best_w
    return h

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME - SAFETY_MARGIN
    
    # Phase 1: Find best Sidon set (use ~30% of time)
    sidon_time = min(30.0, (deadline - time.time()) * 0.3)
    support = find_best_sidon(N, sidon_time)
    
    if not support:
        # Fallback: single spike
        h = np.zeros(N)
        h[0] = 1.0
        return h.tolist()
    
    # Phase 2: Optimize weights (use remaining time)
    remaining = deadline - time.time()
    if remaining < 5:
        # Not enough time, use equal weights
        h = np.zeros(N)
        h[support] = 1.0 / len(support)
        return h.tolist()
    
    h_opt = optimize_weights_on_support(support, remaining)
    
    if h_opt is None:
        h = np.zeros(N)
        h[support] = 1.0 / len(support)
        return h.tolist()
    
    # Ensure normalization
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    
    return h_opt.tolist()
