import numpy as np
import time

N = 50
EPS = 1e-12

def _autoconv(h):
    # Full autoconvolution
    return np.correlate(h, h, mode='full')

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    l2_sq = float(np.dot(h, h))
    c = _autoconv(h)
    sup_c = float(np.max(c))
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def _score_full(h):
    return compute_score(h)

def _smooth_score(h, kappa=200.0):
    # Soft surrogate: use log-sum-exp to approximate sup of autoconvolution
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    l2_sq = float(np.dot(h, h))
    c = _autoconv(h)
    # log-sum-exp for max
    m = np.max(c)
    if m < 1e-15:
        return 0.0
    # Use a soft max: m + log(sum(exp(c_i / tau))/n) where tau controls smoothness
    # For large kappa, this approximates m; for smaller, it's smoother
    tau = 1.0 / kappa
    # Scale to avoid overflow
    c_scaled = c / tau
    c_max = np.max(c_scaled)
    # logsumexp
    logsum = c_max + np.log(np.sum(np.exp(c_scaled - c_max)))
    soft_max = tau * logsum
    if soft_max < 1e-15:
        return 0.0
    return l2_sq / (l1 * soft_max)

def _neg_smooth_score(h, kappa=200.0):
    return -_smooth_score(h, kappa=kappa)

def _neg_score(h):
    return -compute_score(h)

def _polish(h0, max_iter=5000, tol=1e-14):
    """High-precision local polish using coordinate descent with line search."""
    h = np.asarray(h0, dtype=float).copy()
    h = np.maximum(h, 0.0)
    best_score = _score_full(h)
    
    # Coordinate descent
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            old = h[i]
            # Try a few perturbations
            for delta in [0.01, 0.001, 0.0001, -0.01, -0.001, -0.0001, 0.1, -0.1]:
                new_val = old + delta
                if new_val < 0:
                    new_val = 0.0
                h[i] = new_val
                s = _score_full(h)
                if s > best_score + 1e-15:
                    best_score = s
                    improved = True
                else:
                    h[i] = old
            # Fine tune around current best
            for scale in [0.01, 0.1, 1.0]:
                step = scale * 0.001
                h[i] = old + step
                s = _score_full(h)
                if s > best_score + 1e-15:
                    best_score = s
                    improved = True
                h[i] = old - step
                if h[i] < 0:
                    h[i] = 0.0
                s = _score_full(h)
                if s > best_score + 1e-15:
                    best_score = s
                    improved = True
                h[i] = old
        
        if not improved:
            break
    
    return h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial candidate: known good 2-element solution
    h_init = np.zeros(N)
    h_init[0] = 2.0 / 3.0
    h_init[1] = 1.0 / 3.0
    best_h = h_init.copy()
    best_score = _score_full(best_h)
    
    # Generate multiple diverse initial points
    candidates = []
    
    # Candidate 1: 2-element
    candidates.append(h_init.copy())
    
    # Candidate 2: 3-element geometric
    h3 = np.zeros(N)
    h3[0] = 0.5
    h3[1] = 0.3
    h3[2] = 0.2
    candidates.append(h3)
    
    # Candidate 3: 4-element
    h4 = np.zeros(N)
    h4[0] = 0.4
    h4[1] = 0.3
    h4[2] = 0.2
    h4[3] = 0.1
    candidates.append(h4)
    
    # Candidate 4: Uniform 5
    h5 = np.zeros(N)
    h5[:5] = 0.2
    candidates.append(h5)
    
    # Candidate 5: Uniform 10
    h10 = np.zeros(N)
    h10[:10] = 0.1
    candidates.append(h10)
    
    # Candidate 6: Triangular shape
    htri = np.zeros(N)
    for i in range(7):
        htri[i] = (i + 1) * 0.1
    htri = htri / htri.sum()
    candidates.append(htri)
    
    # Candidate 7: Inverse triangular
    hinvt = np.zeros(N)
    for i in range(7):
        hinvt[i] = (8 - i) * 0.1
    hinvt = hinvt / hinvt.sum()
    candidates.append(hinvt)
    
    # Candidate 8: Two separated peaks
    h2p = np.zeros(N)
    h2p[0] = 0.4
    h2p[1] = 0.2
    h2p[10] = 0.3
    h2p[11] = 0.1
    h2p = h2p / h2p.sum()
    candidates.append(h2p)
    
    # Candidate 9: Random sparse
    rng = np.random.default_rng(42)
    hrand = np.zeros(N)
    indices = rng.choice(N, size=5, replace=False)
    hrand[indices] = rng.dirichlet(np.ones(5))
    candidates.append(hrand)
    
    # Candidate 10: Another random
    hrand2 = np.zeros(N)
    indices2 = rng.choice(N, size=8, replace=False)
    hrand2[indices2] = rng.dirichlet(np.ones(8))
    candidates.append(hrand2)
    
    # Phase 1: Quick coordinate descent polish on each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        h_polished, s = _polish(cand, max_iter=100, tol=1e-12)
        if s > best_score:
            best_score = s
            best_h = h_polished.copy()
    
    # Phase 2: Smooth surrogate optimization from best candidate
    if time.time() < deadline - 20:
        h_smooth = best_h.copy()
        # Use gradient-free optimization with smooth score
        from scipy.optimize import minimize
        
        # L-BFGS-B on smooth surrogate
        try:
            bounds = [(0.0, 1.0)] * N
            res = minimize(
                lambda x: _neg_smooth_score(x, kappa=50.0),
                h_smooth,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_candidate = np.maximum(res.x, 0.0)
            s = _score_full(h_candidate)
            if s > best_score:
                best_score = s
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start with smooth surrogate
    if time.time() < deadline - 15:
        from scipy.optimize import minimize
        bounds = [(0.0, 1.0)] * N
        
        for i, cand in enumerate(candidates):
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: _neg_smooth_score(x, kappa=30.0),
                    cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_candidate = np.maximum(res.x, 0.0)
                s = _score_full(h_candidate)
                if s > best_score:
                    best_score = s
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Phase 4: High-precision polish on best
    if time.time() < deadline - 5:
        h_final, s_final = _polish(best_h, max_iter=2000, tol=1e-14)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    return best_h.tolist()
