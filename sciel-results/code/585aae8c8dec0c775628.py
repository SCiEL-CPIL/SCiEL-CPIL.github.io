import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = N / 2.0
DEADLINE = 140.0  # Safety margin before 150s kill

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _corr_vec(h):
    """Compute correlation vector C_k = sum_i h_i * (1 - h_{i+k}) for all lags."""
    h = np.asarray(h, dtype=np.float64)
    return np.correlate(h, 1.0 - h, mode="full")

def _max_corr(h):
    return float(np.max(_corr_vec(h)))

def _smooth_max(corr, lam):
    """Log-sum-exp approximation to max(corr)."""
    m = np.max(corr)
    if m > 1e10:
        return m
    scaled = corr / lam
    scaled = np.clip(scaled, -700, 700)
    return m + lam * np.log(np.sum(np.exp(scaled - np.max(scaled))))

def _fix_sum(h):
    """Adjust h to satisfy sum == N/2, keeping in [0,1]."""
    h = h.copy()
    for _ in range(50):
        diff = HALF - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _local_binary_search(h_bin, max_iters=150, time_budget=20.0):
    """Try swapping 1->0 and 0->1 pairs to reduce max correlation.
    Uses incremental correlation update for speed."""
    h = h_bin.astype(np.float64)
    corr = _corr_vec(h)
    current = float(np.max(corr))
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    if len(ones) == 0 or len(zeros) == 0:
        return h
    
    t_start = time.time()
    improved = True
    it = 0
    while improved and it < max_iters:
        if time.time() - t_start > time_budget:
            break
        improved = False
        it += 1
        best_gain = 1e-12
        best_i = -1
        best_j = -1
        for i in ones:
            for j in zeros:
                # Delta when h[i] goes 1->0 and h[j] goes 0->1
                # C_k = sum_m h[m]*(1-h[m+k])
                # Change: -1*(1-h[i+k]) + 1*(1-h[j+k]) for each k
                # Also: -h[i]*(1-h[i+k]) was there, now 0; +h[j]*(1-h[j+k]) was 0, now 1
                # Net: for lag k: delta_k = -(1-h[i+k]) + (1-h[j+k]) ... but need to be careful
                # Actually C_k = sum_m h[m] * (1-h[m+k])
                # When h[i]: 1->0, term at m=i: was 1*(1-h[i+k]), now 0 => delta = -(1-h[i+k])
                # When h[j]: 0->1, term at m=j: was 0, now 1*(1-h[j+k]) => delta = +(1-h[j+k])
                # But also h[i] appears as (1-h[i+k']) in terms where m+k'=i, i.e., m=i-k'
                # So we need full recomputation or careful delta
                # For safety and correctness, just recompute
                h[i] = 0.0
                h[j] = 1.0
                new_corr = _corr_vec(h)
                new_val = float(np.max(new_corr))
                gain = current - new_val
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_corr = new_corr.copy()
                h[i] = 1.0
                h[j] = 0.0
        if best_i >= 0:
            h[best_i] = 0.0
            h[best_j] = 1.0
            corr = best_corr
            current = current - best_gain
            ones = np.where(h > 0.5)[0]
            zeros = np.where(h < 0.5)[0]
            improved = True
    return h

def _local_continuous_refine(h, max_iters=50, time_budget=10.0):
    """Gradient-based local refinement for continuous h."""
    h = np.clip(h.copy(), 0.0, 1.0)
    h = _fix_sum(h)
    t_start = time.time()
    lr = 0.01
    best_val = _value(h)
    best_h = h.copy()
    
    for it in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros(N)
        for i in range(N):
            hp = h.copy()
            hp[i] = min(1.0, hp[i] + eps)
            hp = _fix_sum(hp)
            hm = h.copy()
            hm[i] = max(0.0, hm[i] - eps)
            hm = _fix_sum(hm)
            grad[i] = (_value(hp) - _value(hm)) / (2 * eps)
        
        # Projected gradient step
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        
        new_val = _value(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
            h = h_new
            lr = min(lr * 1.5, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_h

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}]
    
    def make_objective(lam):
        def objective(x):
            h = np.clip(x, 0.0, 1.0)
            corr = np.correlate(h, 1.0 - h, mode="full")
            return _smooth_max(corr, lam)
        return objective
    
    def try_optimize(start_h, lam, maxiter=100):
        nonlocal best_h, best_val
        try:
            res = minimize(
                make_objective(lam), start_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'iprint': 0}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            val = _value(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    def try_binary_refine(h_cont):
        nonlocal best_h, best_val
        try:
            h_bin = (h_cont > 0.5).astype(np.int8)
            if np.sum(h_bin) != 50:
                sorted_idx = np.argsort(h_cont)
                if np.sum(h_bin) > 50:
                    for idx in sorted_idx:
                        if np.sum(h_bin) == 50:
                            break
                        h_bin[idx] = 0
                else:
                    for idx in sorted_idx[::-1]:
                        if np.sum(h_bin) == 50:
                            break
                        h_bin[idx] = 1
            h_refined = _local_binary_search(h_bin, max_iters=150, time_budget=15.0)
            val = _value(h_refined)
            if val < best_val:
                best_h = h_refined.copy()
                best_val = val
        except Exception:
            pass
    
    def try_continuous_refine(h_cont):
        nonlocal best_h, best_val
        try:
            h_refined = _local_continuous_refine(h_cont, max_iters=50, time_budget=8.0)
            val = _value(h_refined)
            if val < best_val:
                best_h = h_refined.copy()
                best_val = val
        except Exception:
            pass
    
    # Generate focused set of starting points
    rng = np.random.default_rng(42)
    starts = []
    
    # Seed
    starts.append(np.full(N, 0.5))
    
    # Sine waves - key patterns for correlation minimization
    for freq in range(1, 15):
        for phase in np.linspace(0, 2*np.pi, 3, endpoint=False):
            h_s = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_s = np.clip(h_s, 0, 1)
            h_s = _fix_sum(h_s)
            starts.append(h_s)
    
    # Random continuous
    for _ in range(20):
        h_r = rng.uniform(0, 1, N)
        h_r = _fix_sum(h_r)
        starts.append(h_r)
    
    # Multi-frequency
    for _ in range(15):
        h_m = np.full(N, 0.5)
        for _ in range(3):
            f = rng.integers(1, 12)
            p = rng.uniform(0, 2*np.pi)
            a = rng.uniform(0.1, 0.3)
            h_m += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        h_m = np.clip(h_m, 0, 1)
        h_m = _fix_sum(h_m)
        starts.append(h_m)
    
    # Binary-like starts
    for _ in range(15):
        h_b = (rng.uniform(0, 1, N) > 0.5).astype(np.float64)
        if np.sum(h_b) != 50:
            idx = np.argsort(rng.uniform(0, 1, N))
            if np.sum(h_b) > 50:
                for i in idx:
                    if np.sum(h_b) == 50:
                        break
                    h_b[i] = 0
            else:
                for i in idx[::-1]:
                    if np.sum(h_b) == 50:
                        break
                    h_b[i] = 1
        starts.append(h_b)
    
    # Periodic patterns
    for period in [3, 4, 5, 6, 7, 8, 10, 12]:
        for offset in range(period):
            h_p = np.zeros(N)
            for i in range(N):
                if (i + offset) % period < period // 2:
                    h_p[i] = 1.0
            h_p = _fix_sum(h_p)
            starts.append(h_p)
    
    # Phase: optimize with annealing from best starts
    # Sort starts by initial value to prioritize good ones
    starts.sort(key=lambda x: _value(x))
    
    # First pass: quick optimization with moderate lambda
    for si, start_h in enumerate(starts[:40]):
        if time.time() > deadline - 30:
            break
        try_optimize(start_h, 1.0, maxiter=80)
    
    # Second pass: refine best with smaller lambda
    if time.time() < deadline - 25:
        try_optimize(best_h, 0.1, maxiter=100)
        try_optimize(best_h, 0.01, maxiter=100)
    
    # Continuous local refinement
    if time.time() < deadline - 20:
        try_continuous_refine(best_h)
    
    # Binary refinement
    if time.time() < deadline - 15:
        try_binary_refine(best_h)
    
    # Final polish: alternate continuous and binary
    for _ in range(2):
        if time.time() > deadline - 5:
            break
        try_continuous_refine(best_h)
        try_binary_refine(best_h)
        try_optimize(best_h, 0.005, maxiter=50)
    
    return best_h.tolist()
