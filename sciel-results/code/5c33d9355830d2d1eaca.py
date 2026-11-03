import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def _find_sidon_set(n):
    """Find a Sidon set of size n (all pairwise sums distinct)."""
    # Greedy construction
    S = [0]
    for i in range(1, n):
        candidate = i
        while True:
            S_new = S + [candidate]
            sums = set()
            ok = True
            for a in range(len(S_new)):
                for b in range(a, len(S_new)):
                    s = S_new[a] + S_new[b]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                S.append(candidate)
                break
            candidate += 1
    return S

def _neg_obj(h, beta):
    """Smoothed negative objective for optimization."""
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-12:
        return 0.0
    hc = np.convolve(h, h, mode='full')
    if np.max(hc) < 1e-12:
        return 0.0
    # Soft max approximation
    scaled = beta * hc
    max_scaled = np.max(scaled)
    soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
    return -(l2_sq / soft_max)

def _neg_obj_exact(h):
    """Exact negative objective."""
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    hc = np.convolve(h, h, mode='full')
    mc = np.max(hc)
    if mc < 1e-12:
        return 0.0
    return -(l2_sq / mc)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Find Sidon set for support
    sidon_set = _find_sidon_set(N)
    K = len(sidon_set)
    sorted_sidon = sorted(sidon_set)

    # Initialize with exponential weights on Sidon set
    # This creates a skew that helps the ratio
    best_h = np.zeros(N)
    # Use exponential decay: weights proportional to r^i for some r < 1
    # This makes one weight dominant, reducing max autoconv relative to sum of squares
    r = 0.5
    weights = r ** np.arange(K)
    weights /= np.sum(weights)
    best_h[sorted_sidon] = weights
    best_score = _score_raw(best_h)

    # Phase 1: Global optimization with differential evolution on the support
    # We optimize the weights on the Sidon set positions
    # Use a parametrization: h_i = exp(x_i) / sum(exp(x_j)) for j in support
    # This automatically satisfies non-negativity and normalization

    def _to_h(x, support):
        """Convert unconstrained x to normalized nonnegative h on support."""
        x_clipped = np.clip(x, -50, 50)  # Avoid overflow
        w = np.exp(x_clipped)
        w /= np.sum(w)
        h = np.zeros(N)
        h[support] = w
        return h

    def _neg_obj_param(x):
        h = _to_h(x, sorted_sidon)
        return _neg_obj_exact(h)

    # Differential evolution on the support weights
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10)] * K
            de_result = differential_evolution(
                _neg_obj_param, bounds,
                seed=42, maxiter=100, tol=1e-10,
                popsize=20, mutation=(0.5, 1.0), recombination=0.7,
                polish=True, init='latinhypercube'
            )
            if de_result.success:
                h_candidate = _to_h(de_result.x, sorted_sidon)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Phase 2: Local optimization with SLSQP using smoothed objective
    # Try multiple beta values for the soft max
    betas = [100.0, 50.0, 20.0, 10.0]
    for beta in betas:
        if time.time() >= deadline - 30:
            break
        try:
            res = minimize(
                lambda h: _neg_obj(h, beta), best_h,
                method='SLSQP',
                bounds=[(0.0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Phase 3: Exact local optimization
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_obj_exact, best_h,
                method='SLSQP',
                bounds=[(0.0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}],
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Phase 4: Try different exponential skew parameters
    if time.time() < deadline - 5:
        for r_try in [0.3, 0.4, 0.6, 0.7, 0.8]:
            if time.time() >= deadline - 5:
                break
            h_try = np.zeros(N)
            w = r_try ** np.arange(K)
            w /= np.sum(w)
            h_try[sorted_sidon] = w
            s_try = _score_raw(h_try)
            if s_try > best_score:
                best_score = s_try
                best_h = h_try

    # Final polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _neg_obj_exact, best_h,
                method='SLSQP',
                bounds=[(0.0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}],
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
