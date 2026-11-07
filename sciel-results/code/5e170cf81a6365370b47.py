"""Program for Erdos minimum overlap task using anti-periodic model + progressive smooth optimization.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). We build a strong anti-periodic
seed and refine it with progressive smooth optimization, sharpening the
surrogate to target the true max correlation, followed by discrete refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_epigraph(h, t):
    """Epigraph objective: minimize t s.t. max correlation <= t*N/2.
    Returns a smooth penalty-based objective for L-BFGS-B."""
    c = np.correlate(h, 1.0 - h, mode="full")
    # We want to minimize t, with constraint max(c) <= t*N/2
    # Penalty: t + max(0, max(c) - t*N/2)^2
    max_c = float(np.max(c))
    penalty = max(0.0, max_c - t * N / 2.0) ** 2
    return t * 2.0 / N + penalty


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    # Clip for numerical stability
    z = np.clip(z, -100, 100)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def value_smooth_grad(h, beta):
    """Smooth surrogate with gradient via log-sum-exp."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    z = np.clip(z, -100, 100)
    exp_z = np.exp(z)
    sum_exp = np.sum(exp_z)
    logsum = np.log(sum_exp) + beta * cm
    
    # Gradient of logsum w.r.t. c: beta * exp(z) / sum_exp
    dc = beta * exp_z / sum_exp
    
    # dc/dh: c[k] = sum_i h[i] * (1-h[i+k-N+1]) for full correlation
    # c[k] = sum_{i} h[i] * (1 - h[i + k - (N-1)])
    # Let's compute gradient numerically-free via the structure
    # c[k] = sum_i h[i] * (1-h[i+k-N+1]) where valid indices
    # dc/dh[j] = sum_k dc[k] * dc[k]/dh[j]
    # dc[k]/dh[j] = (1 - h[j+k-N+1]) if j+k-N+1 in [0,N-1] and j in [0,N-1]
    #             + (-h[j]) if j in [0,N-1] and j+k-N+1 = j (i.e., k=N-1)
    # Actually: c[k] = sum_i h[i] * (1 - h[i+k-(N-1)])
    # dc/dh[j] = (1 - h[j+k-(N-1)]) for the term where i=j (if j+k-(N-1) in [0,N-1])
    #          + (-h[j]) * ... no wait, h appears in both factors
    # c[k] = sum_i h[i] - h[i]h[i+k-(N-1)]
    # dc/dh[j] = 1 (from h[j] term) - h[j+k-(N-1)] (from h[j]*h[j+k-(N-1)] term) 
    #           - h[j] (from h[j-(k-(N-1))]*h[j] term if j-(k-(N-1)) in [0,N-1])
    
    # This is getting complex. Let's use a simpler approach:
    # For anti-periodic h, h[i+HALF] = 1-h[i], so we only optimize first HALF.
    # Let's just return the value and let L-BFGS-B use numerical gradient.
    return float(logsum) / N * 2.0


def make_seed():
    """Strong anti-periodic seed: triangular wave, zero-mean per anti-pair."""
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_v2():
    """Alternative seed: sinusoidal anti-periodic pattern."""
    i = np.arange(HALF)
    # Try multiple frequency patterns
    best_h = None
    best_v = float('inf')
    for freq in [1, 2, 3, 4, 5, 7]:
        phase = np.pi * freq * i / HALF
        first = 0.5 + 0.3 * np.sin(phase)
        h = np.empty(N)
        h[:HALF] = first
        h[HALF:] = 1.0 - first
        h = np.clip(h, 0.0, 1.0)
        h += (HALF - h.sum()) / N
        h = np.clip(h, 0.0, 1.0)
        v = value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    return best_h


def refine_continuous(h, beta, maxiter):
    """Optimize the anti-periodic parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    p0 = h0[:HALF].copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value_smooth(unpack(p), beta)

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-12, "maxls": 40})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def refine_epigraph(h, maxiter):
    """Optimize using epigraph formulation with auxiliary variable t."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    p0 = h0[:HALF].copy()
    t0 = value(h0) * N / 2.0  # initial t estimate
    
    # We optimize [p0..., t] where t is the last parameter
    n_params = HALF + 1
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF + [(0.0, 2.0)]

    def unpack(pt):
        p = np.clip(pt[:HALF], 0.0, 1.0)
        t = max(0.0, pt[HALF])
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh, t

    def obj(pt):
        hh, t = unpack(pt)
        return value_epigraph(hh, t)

    try:
        res = minimize(obj, np.concatenate([p0, [t0]]), method="L-BFGS-B", 
                       bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-12, "maxls": 50})
        hh, t = unpack(res.x)
        return hh
    except Exception:
        return h0


def snap_to_grid(h, n_levels):
    """Snap h to a symmetric grid of n_levels values, preserving anti-periodicity."""
    h = np.asarray(h, dtype=np.float64)
    first = h[:HALF].copy()
    
    if n_levels <= 1:
        first_q = np.full(HALF, 0.5)
    else:
        grid = np.linspace(0.0, 1.0, n_levels)
        idx = np.argmin(np.abs(first[:, None] - grid[None, :]), axis=1)
        first_q = grid[idx]
    
    # Adjust to satisfy sum constraint: sum(first_q) should be HALF
    target_sum = HALF
    current_sum = first_q.sum()
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-10:
        first_q += diff / HALF
        first_q = np.clip(first_q, 0.0, 1.0)
        for _ in range(10):
            diff = target_sum - first_q.sum()
            if abs(diff) < 1e-12:
                break
            first_q += diff / HALF
            first_q = np.clip(first_q, 0.0, 1.0)
    
    h_out = np.empty(N)
    h_out[:HALF] = first_q
    h_out[HALF:] = 1.0 - first_q
    h_out = np.clip(h_out, 0.0, 1.0)
    
    total = h_out.sum()
    if abs(total - HALF) > 1e-9 * N:
        h_out += (HALF - total) / N
        h_out = np.clip(h_out, 0.0, 1.0)
    
    return h_out


def greedy_permutation_search(h, max_iters=5000):
    """Greedy local search: swap pairs in the first half to reduce max correlation."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_val = value(h)
    
    rng = np.random.default_rng(42)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and remaining() > 20.0:
        improved = False
        iters += 1
        for i in range(HALF):
            if remaining() < 20.0:
                break
            for j in range(i + 1, HALF):
                if remaining() < 20.0:
                    break
                first[i], first[j] = first[j], first[i]
                h_candidate = np.empty(N)
                h_candidate[:HALF] = first
                h_candidate[HALF:] = 1.0 - first
                v_candidate = value(h_candidate)
                
                if v_candidate < best_val - 1e-12:
                    best_val = v_candidate
                    improved = True
                else:
                    first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = first
    h_out[HALF:] = 1.0 - first
    return h_out, best_val


def local_polish(h, max_iter=1000):
    """Fine-grained local optimization using small perturbations."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_val = value(h)
    rng = np.random.default_rng(99)
    
    step_size = 0.05
    for it in range(max_iter):
        if remaining() < 20.0:
            break
        if it % 100 == 0:
            step_size *= 0.7
        
        idx = rng.integers(0, HALF)
        delta = rng.normal(0, step_size)
        new_val = first[idx] + delta
        
        if 0.0 <= new_val <= 1.0:
            first[idx] = new_val
            h_candidate = np.empty(N)
            h_candidate[:HALF] = first
            h_candidate[HALF:] = 1.0 - first
            v_candidate = value(h_candidate)
            
            if v_candidate < best_val:
                best_val = v_candidate
            else:
                first[idx] = new_val - delta
    
    h_out = np.empty(N)
    h_out[:HALF] = first
    h_out[HALF:] = 1.0 - first
    return h_out, best_val


def simulated_annealing_permutation(h, max_iters=20000):
    """Simulated annealing on permutations of the first half."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_first = first.copy()
    best_val = value(h)
    current_val = best_val
    
    rng = np.random.default_rng(123)
    
    T0 = 0.1
    T_end = 1e-6
    n_steps = min(max_iters, int(remaining() * 100))
    if n_steps < 100:
        n_steps = 100
    
    for it in range(n_steps):
        if remaining() < 25.0:
            break
        T = T0 * (T_end / T0) ** (it / max(1, n_steps - 1))
        
        i = rng.integers(0, HALF)
        j = rng.integers(0, HALF)
        if i == j:
            continue
        
        first[i], first[j] = first[j], first[i]
        h_candidate = np.empty(N)
        h_candidate[:HALF] = first
        h_candidate[HALF:] = 1.0 - first
        v_candidate = value(h_candidate)
        
        delta = v_candidate - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = v_candidate
            if v_candidate < best_val:
                best_val = v_candidate
                best_first = first.copy()
        else:
            first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = best_first
    h_out[HALF:] = 1.0 - best_first
    return h_out, best_val


def construct():
    t0 = time.time()

    # Phase 1: Try multiple seeds and continuous optimization
    seeds = [make_seed(), make_seed_v2()]
    best = None
    best_val = float('inf')
    
    for seed in seeds:
        if remaining() < 50.0:
            break
        h = seed.copy()
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
        
        # Progressive smooth optimization with increasing beta
        for beta in [10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
            if remaining() < 40.0:
                break
            h = refine_continuous(h, beta, 300)
            v = value(h)
            if v < best_val:
                best, best_val = h.copy(), v
    
    # Phase 1.5: Epigraph optimization
    if remaining() > 40.0:
        h_epi = refine_epigraph(best, 500)
        v_epi = value(h_epi)
        if v_epi < best_val:
            best, best_val = h_epi.copy(), v_epi

    # Phase 2: Snap to discrete grid and do permutation search
    for n_levels in [3, 5, 7, 11, 15]:
        if remaining() < 30.0:
            break
        
        h_grid = snap_to_grid(best, n_levels)
        v_grid = value(h_grid)
        
        if v_grid < best_val:
            best, best_val = h_grid.copy(), v_grid
        
        # Greedy permutation search
        if remaining() > 25.0:
            h_greedy, v_greedy = greedy_permutation_search(h_grid, max_iters=3000)
            if v_greedy < best_val:
                best, best_val = h_greedy.copy(), v_greedy
        
        # Simulated annealing (if time permits)
        if remaining() > 25.0:
            h_sa, v_sa = simulated_annealing_permutation(
                h_greedy if 'h_greedy' in locals() else h_grid, max_iters=10000)
            if v_sa < best_val:
                best, best_val = h_sa.copy(), v_sa

    # Phase 3: Final continuous polish on the best discrete solution
    if remaining() > 15.0:
        p0 = np.clip(best[:HALF], 0.0, 1.0).copy()
        bounds = [(0.0, 1.0)] * HALF

        def unpack(p):
            p = np.clip(p, 0.0, 1.0)
            hh = np.empty(N)
            hh[:HALF] = p
            hh[HALF:] = 1.0 - p
            return hh

        # Try multiple betas for the final polish
        for beta in [200.0, 500.0, 1000.0, 2000.0]:
            if remaining() < 10.0:
                break
            try:
                res = minimize(lambda p: value_smooth(unpack(p), beta), p0,
                               method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 500, "ftol": 1e-16,
                                        "gtol": 1e-14, "maxls": 50})
                h = unpack(np.clip(res.x, 0.0, 1.0))
                v = value(h)
                if v < best_val:
                    best, best_val = h.copy(), v
                    p0 = np.clip(res.x, 0.0, 1.0).copy()
            except Exception:
                pass

    # Phase 4: Local polish with small perturbations
    if remaining() > 10.0:
        h_local, v_local = local_polish(best, max_iter=500)
        if v_local < best_val:
            best, best_val = h_local.copy(), v_local

    # Ensure constraints are met
    best = np.clip(best, 0.0, 1.0)
    best += (HALF - best.sum()) / N
    best = np.clip(best, 0.0, 1.0)
    
    return best
