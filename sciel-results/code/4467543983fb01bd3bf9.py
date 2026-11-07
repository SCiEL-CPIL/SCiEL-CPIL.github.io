import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = N / 2.0
DEADLINE = 135.0  # Safety margin before 150s kill

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

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

def _local_binary_search(h_bin, max_iters=300):
    """Try swapping 1->0 and 0->1 pairs to reduce max correlation."""
    h = h_bin.astype(np.float64)
    current = _max_corr(h)
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    if len(ones) == 0 or len(zeros) == 0:
        return h
    
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        best_gain = 1e-12
        best_i = -1
        best_j = -1
        for i in ones:
            for j in zeros:
                h[i] = 0.0
                h[j] = 1.0
                new_val = _max_corr(h)
                gain = current - new_val
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                h[i] = 1.0
                h[j] = 0.0
        if best_i >= 0:
            h[best_i] = 0.0
            h[best_j] = 1.0
            current = current - best_gain
            # Update index sets
            ones = np.where(h > 0.5)[0]
            zeros = np.where(h < 0.5)[0]
            improved = True
    return h

def _local_continuous_refine(h, time_limit=10.0):
    """Random perturbation local search in continuous space."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0).copy()
    h = _fix_sum(h)
    best = h.copy()
    bestv = _value(best)
    t0 = time.time()
    step = 0.15
    rng = np.random.default_rng(12345)
    
    while time.time() - t0 < time_limit and step > 1e-5:
        improved = False
        for _ in range(5000):
            if time.time() - t0 > time_limit:
                break
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j:
                continue
            delta = rng.uniform(-step, step)
            ni = best[i] - delta
            nj = best[j] + delta
            if ni < 0 or ni > 1 or nj < 0 or nj > 1:
                continue
            trial = best.copy()
            trial[i] = ni
            trial[j] = nj
            trial = _fix_sum(trial)
            v = _value(trial)
            if v < bestv - 1e-12:
                best = trial
                bestv = v
                improved = True
        if not improved:
            step *= 0.7
    return best, bestv

def _epigraph_solve(h0, time_limit=20.0):
    """
    Solve the epigraph formulation:
    min t
    s.t. C_k <= t for all k
         sum(h) = 50
         0 <= h_i <= 1
    Using scipy.optimize.minimize with SLSQP.
    """
    # Initial t: use current max correlation
    c0 = np.correlate(h0, 1.0 - h0, mode="full")
    t0 = float(np.max(c0)) + 0.1
    
    z0 = np.concatenate([h0, [t0]])
    
    bounds = [(0.0, 1.0)] * N + [(0.0, float('inf'))]
    
    def sum_constraint(x):
        return x[:N].sum() - HALF
    
    constraints = [{'type': 'eq', 'fun': sum_constraint}]
    
    # Inequality constraints: t - C_k >= 0 for each lag
    for k in range(2 * N - 1):
        def make_ineq(k):
            def ineq(x):
                h = x[:N]
                t = x[N]
                c = np.correlate(h, 1.0 - h, mode="full")
                return t - c[k]
            return ineq
        constraints.append({'type': 'ineq', 'fun': make_ineq(k)})
    
    def objective(z):
        return z[N]
    
    t_start = time.time()
    
    try:
        res = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-10, 'iprint': 0},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout")),
        )
        
        if res.success or (hasattr(res, 'x') and res.x is not None):
            h_sol = np.clip(res.x[:N], 0.0, 1.0)
            h_sol = _fix_sum(h_sol)
            return h_sol
    except Exception:
        pass
    
    return None

def _generate_starts():
    """Generate diverse starting points."""
    rng = np.random.default_rng(42)
    starts = []
    
    # Seed
    starts.append(np.full(N, 0.5))
    
    # Sine waves with various frequencies and phases
    for freq in range(1, 25):
        for phase in np.linspace(0, 2*np.pi, 4, endpoint=False):
            h_s = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_s = np.clip(h_s, 0, 1)
            h_s = _fix_sum(h_s)
            starts.append(h_s)
    
    # Random
    for _ in range(40):
        h_r = rng.uniform(0, 1, N)
        h_r = _fix_sum(h_r)
        starts.append(h_r)
    
    # Multi-frequency
    for _ in range(30):
        h_m = np.full(N, 0.5)
        for _ in range(4):
            f = rng.integers(1, 18)
            p = rng.uniform(0, 2*np.pi)
            a = rng.uniform(0.08, 0.35)
            h_m += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        h_m = np.clip(h_m, 0, 1)
        h_m = _fix_sum(h_m)
        starts.append(h_m)
    
    # Binary-like starts
    for _ in range(20):
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
    
    # Phase-shifted patterns (Erdos-style)
    for period in [3, 4, 5, 6, 7, 8, 10, 12]:
        for offset in range(period):
            h_p = np.zeros(N)
            for i in range(N):
                if (i + offset) % period < period // 2:
                    h_p[i] = 1.0
                else:
                    h_p[i] = 0.0
            h_p = _fix_sum(h_p)
            starts.append(h_p)
    
    return starts

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
    
    def try_optimize(start_h, lam, maxiter=200):
        nonlocal best_h, best_val
        try:
            res = minimize(
                make_objective(lam), start_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'iprint': 0}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            val = _value(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    def try_epigraph(h0):
        nonlocal best_h, best_val
        remaining = deadline - 5.0 - time.time()
        if remaining < 2.0:
            return
        h_sol = _epigraph_solve(h0, time_limit=min(15.0, remaining))
        if h_sol is not None:
            val = _value(h_sol)
            if val < best_val:
                best_h = h_sol.copy()
                best_val = val
    
    def try_binary_refine(h_cont):
        nonlocal best_h, best_val
        try:
            h_bin = (h_cont > 0.5).astype(np.int8)
            # Ensure exactly 50 ones
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
            h_refined = _local_binary_search(h_bin, max_iters=150)
            val = _value(h_refined)
            if val < best_val:
                best_h = h_refined.copy()
                best_val = val
        except Exception:
            pass
    
    # Generate diverse starting points
    starts = _generate_starts()
    
    # Process starts with multiple lambda values (annealing)
    # Use a subset of lambda values for speed
    lambda_schedule = [5.0, 1.0, 0.1, 0.01]
    
    for si, start_h in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        # Try a few lambda values for each start
        for lam in lambda_schedule:
            if time.time() > deadline - 30.0:
                break
            try_optimize(start_h, lam, maxiter=100)
    
    # Refine best solution with binary local search
    if time.time() < deadline - 20.0:
        try_binary_refine(best_h)
    
    # Epigraph solve on the best solution found so far
    if time.time() < deadline - 15.0:
        try_epigraph(best_h)
    
    # Additional refinement rounds on best solution
    for _ in range(3):
        if time.time() > deadline - 10.0:
            break
        try_optimize(best_h, 0.01, maxiter=150)
        try_binary_refine(best_h)
        if time.time() < deadline - 10.0:
            try_epigraph(best_h)
    
    # Final continuous local refinement
    if time.time() < deadline - 3.0:
        h_final, v_final = _local_continuous_refine(best_h, time_limit=max(1.0, deadline - 3.0 - time.time()))
        if v_final < best_val:
            best_h = h_final.copy()
            best_val = v_final
    
    # Ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)
    
    return best_h.tolist()
