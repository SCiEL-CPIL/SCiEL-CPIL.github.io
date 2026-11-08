import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Extra safety margin

    best_score = 1.0
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0

    def neg_score_raw(params):
        h = np.maximum(params, 0)
        s = np.sum(h)
        if s < 1e-12:
            return 0.0
        h = h / s
        return -_score(h)

    np.random.seed(42)

    # Build a diverse set of starting points
    candidates = []

    # Delta
    h = np.zeros(N); h[N // 2] = 1.0
    candidates.append(h)

    # Two-point asymmetric clusters
    for a in [0.85, 0.9, 0.95, 0.7, 0.6, 0.5]:
        h = np.zeros(N)
        h[N // 2 - 1] = a
        h[N // 2] = 1 - a
        candidates.append(h)

    # Three-point clusters
    for (a, b, c) in [(0.7, 0.2, 0.1), (0.8, 0.1, 0.1), (0.6, 0.3, 0.1),
                       (0.5, 0.3, 0.2), (0.9, 0.05, 0.05)]:
        h = np.zeros(N)
        h[N // 2 - 1] = a
        h[N // 2] = b
        h[N // 2 + 1] = c
        candidates.append(h)

    # Four-point
    for (a, b, c, d) in [(0.6, 0.2, 0.1, 0.1), (0.5, 0.3, 0.1, 0.1)]:
        h = np.zeros(N)
        h[N // 2 - 1] = a
        h[N // 2] = b
        h[N // 2 + 1] = c
        h[N // 2 + 2] = d
        candidates.append(h)

    # Random inits
    num_random = 50
    for i in range(num_random):
        if time.time() > deadline - 60:
            break
        r = i % 3
        if r == 0:
            k = np.random.randint(1, 12)
            h = np.zeros(N)
            idx = np.random.choice(N, size=k, replace=False)
            h[idx] = np.random.dirichlet(np.ones(k))
        elif r == 1:
            h = np.random.dirichlet(np.ones(N) * np.random.uniform(0.1, 10))
        else:
            k = np.random.randint(2, 18)
            center = np.random.randint(0, N - k)
            h = np.zeros(N)
            h[center:center + k] = np.random.dirichlet(np.ones(k) * np.random.uniform(0.1, 5))
        candidates.append(h)

    # Evaluate all candidates
    scored = []
    for h in candidates:
        if time.time() > deadline - 30:
            break
        sc = _score(h)
        scored.append((sc, h))
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    scored.sort(key=lambda x: -x[0])

    # Refine top candidates
    max_refine = min(12, len(scored))
    for sc, h in scored[:max_refine]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(neg_score_raw, h, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxls': 60})
            h_opt = np.maximum(res.x, 0)
            s = np.sum(h_opt)
            if s > 1e-12:
                h_opt = h_opt / s
                sc_opt = _score(h_opt)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    return [float(x) for x in best_h]
