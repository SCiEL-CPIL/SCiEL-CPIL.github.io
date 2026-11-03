import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def _objective(x):
    h = np.asarray(x, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-12:
        return 0.0
    return -l2_sq / (DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin for 150s limit

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    x_idx = np.arange(N)
    inits = []

    # 1. Uniform
    inits.append(np.ones(N) / N)

    # 2. Gaussian peaks
    for c in np.linspace(2, N-2, 10):
        for w in np.linspace(1, N/3, 10):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = np.sum(h_g)
            if s > 0:
                inits.append(h_g / s)

    # 3. Triangular peaks
    for c in np.linspace(2, N-2, 10):
        for w in np.linspace(2, N, 10):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            s = np.sum(h_t)
            if s > 0:
                inits.append(h_t / s)

    # 4. Sparse "Sidon-like" sets
    def greedy_sidon(n, k):
        S = []
        used_sums = set()
        for i in range(n):
            if len(S) >= k:
                break
            potential_sums = set()
            for s in S:
                potential_sums.add(i + s)
            potential_sums.add(i + i)
            if not any(ps in used_sums for ps in potential_sums):
                S.append(i)
                used_sums.update(potential_sums)
        return S

    for k in range(3, 15):
        if time.time() > deadline: break
        try:
            S = greedy_sidon(N, k)
            if len(S) < 2: continue
            h_s = np.zeros(N)
            h_s[S] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)
        except Exception:
            pass

    # 5. Random Sparse
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline: break
        k = np.random.randint(4, 12)
        idx = np.random.choice(N, k, replace=False)
        h_r = np.zeros(N)
        h_r[idx] = np.random.rand(k)
        s = np.sum(h_r)
        if s > 0:
            inits.append(h_r / s)

    # 6. Alternating patterns
    for period in [2, 3, 4, 5]:
        h_a = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                h_a[i] = 1.0
        s = np.sum(h_a)
        if s > 0:
            inits.append(h_a / s)

    # Evaluate all initializations
    for h_init in inits:
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()

    # Optimization using SLSQP
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    bounds = [(0.0, 1.0)] * N

    def neg_score_direct(h):
        l1 = np.sum(h)
        if l1 < 1e-10: return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-10: return 0.0
        return -l2_sq / (DELTA * max_conv)

    init_scores = [(_score_raw(h), h) for h in inits]
    init_scores.sort(key=lambda x: -x[0])

    top_inits = [h for s, h in init_scores[:10]]
    if len(top_inits) == 0:
        top_inits = [best_h / np.sum(best_h)]
    else:
        if not any(np.allclose(h, best_h) for h in top_inits):
            top_inits.append(best_h / np.sum(best_h))

    # Optimization
    for h_init in top_inits:
        if time.time() > deadline: break
        h_init = h_init / np.sum(h_init)
        try:
            res = minimize(neg_score_direct, h_init, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_score_direct, h_start, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
