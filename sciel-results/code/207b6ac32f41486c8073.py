import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET = N / 2.0

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all non-zero shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # Exclude the zero shift (index N-1)
    c1 = c[:N]      # negative shifts
    c2 = c[N+1:]    # positive shifts
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2 using bisection on Lagrange multiplier."""
    h = np.clip(h, 0.0, 1.0).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    lo = -2.0
    hi = 2.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s_mid = np.sum(np.clip(h - mid, 0.0, 1.0))
        if abs(s_mid - target) < 1e-12:
            break
        if s_mid > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    h = np.clip(h - lam, 0.0, 1.0)
    
    # Final adjustment for floating point residual
    residual = target - np.sum(h)
    if abs(residual) > 1e-14:
        interior = (h > 1e-10) & (h < 1.0 - 1e-10)
        n_int = np.sum(interior)
        if n_int > 0:
            h[interior] += residual / n_int
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_lse(x, beta, lam_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500.0
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, lam_viol, lam_sum):
    """Epigraph: minimize t + smooth penalty for violations + sum penalty."""
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    overlaps = _compute_overlaps(h)
    
    violations = np.maximum(overlaps - t, 0)
    constraint_penalty = lam_viol * np.sum(violations ** 2) + 0.05 * lam_viol * np.sum(violations)
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    return t + constraint_penalty + sum_penalty

def _local_search(h, max_iter=300, step_size=0.05):
    """Local search: random coordinate adjustments to reduce max overlap."""
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    rng = np.random.default_rng(123)
    
    for _ in range(max_iter):
        idx = rng.integers(0, N)
        delta = rng.choice([-step_size, step_size])
        h_new = best_h.copy()
        h_new[idx] += delta
        h_new = _project_to_feasible(h_new)
        score = _max_overlap(h_new)
        if score < best_score - 1e-12:
            best_h = h_new
            best_score = score
            step_size = max(0.01, step_size * 0.9)
        else:
            step_size = min(0.1, step_size * 1.1)
    
    return best_h, best_score

def _swap_local_search(h, max_rounds=30, swaps_per_round=500):
    """Binary swap local search: swap pairs of values to reduce max overlap."""
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    rng = np.random.default_rng(456)
    
    for _ in range(max_rounds):
        improved = False
        for _ in range(swaps_per_round):
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            h_try = best_h.copy()
            h_try[i], h_try[j] = h_try[j], h_try[i]
            h_try = _project_to_feasible(h_try)
            new_score = _max_overlap(h_try)
            if new_score < best_score - 1e-10:
                best_score = new_score
                best_h = h_try
                improved = True
        if not improved:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds = bounds_h + [(-1.0, 2.0)]
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt.copy())
    
    # Sinusoidal modulations
    t_idx = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/2, np.pi]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t_idx/N + phase)
                candidates.append(trial.copy())
    
    # Cosine variants
    for freq in range(1, 15):
        for amp in [0.2, 0.3, 0.4, 0.45]:
            trial = 0.5 + amp * np.cos(2*np.pi*freq*t_idx/N)
            candidates.append(trial.copy())
    
    # Sawtooth
    h_saw = np.linspace(0.1, 0.9, N)
    candidates.append(h_saw.copy())
    
    # Triangular wave
    h_tri = 0.5 + 0.4 * np.abs(2 * (t_idx / N - 0.5) - 1)
    candidates.append(h_tri.copy())
    
    # Step functions
    for split in range(10, 90, 10):
        h_step = np.zeros(N)
        h_step[:split] = 0.6
        h_step[split:] = 0.4
        candidates.append(h_step.copy())
    
    # Sturmian word candidates (quasi-periodic, low-discrepancy)
    for alpha in [0.6180339887, 0.7071067812, 0.75, 0.65, 0.7]:
        n_vals = np.arange(N)
        h_sturm = np.floor((n_vals + 1) * alpha) - np.floor(n_vals * alpha)
        h_sturm = h_sturm.astype(float)
        # Scale to have mean 0.5 while preserving structure
        h_sturm = 0.5 + 0.4 * (h_sturm - 0.5)
        candidates.append(h_sturm.copy())
    
    # Random restarts
    for i in range(40):
        h_rand = rng.uniform(0.05, 0.95, size=N)
        candidates.append(h_rand.copy())
    
    # Bimodal initializations
    for i in range(10):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_bimodal = np.where(h_rand > 0.5, 0.75, 0.25)
        candidates.append(h_bimodal.copy())
    
    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # LSE optimization with beta schedule
    beta_schedule = [3, 5, 10, 15, 25, 40, 60, 100]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        x0 = best_h.copy()
        
        def obj_func(x, b=beta):
            return _objective_lse(x, b, 500.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Epigraph optimization
    if time.time() < deadline - 25.0:
        t_init = best_score
        x0_epigraph = np.concatenate([best_h, [t_init]])
        
        def obj_epigraph(x):
            return _objective_epigraph(x, 100.0, 1000.0)
        
        try:
            result = minimize(
                obj_epigraph,
                x0_epigraph,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Random restarts with LSE
    for i in range(15):
        if time.time() > deadline - 30.0:
            break
        
        x0 = rng.uniform(0.1, 0.9, size=N)
        x0 = _project_to_feasible(x0)
        
        beta = 10.0 + i * 5.0
        
        def obj_rand(x, b=beta):
            return _objective_lse(x, b, 500.0)
        
        try:
            result = minimize(
                obj_rand,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # High-precision polish with very high beta
    if time.time() < deadline - 15.0:
        def obj_polish(x):
            return _objective_lse(x, 200.0, 5000.0)
        
        try:
            result = minimize(
                obj_polish,
                best_h,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Epigraph polish
    if time.time() < deadline - 10.0:
        t_init = best_score
        x0_epigraph = np.concatenate([best_h, [t_init]])
        
        def obj_epigraph_polish(x):
            return _objective_epigraph(x, 200.0, 5000.0)
        
        try:
            result = minimize(
                obj_epigraph_polish,
                x0_epigraph,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Coordinate local search
    if time.time() < deadline - 8.0:
        best_h, best_score = _local_search(best_h, max_iter=200, step_size=0.05)
    
    # Binary swap local search (preserves sum exactly, directly targets max overlap)
    if time.time() < deadline - 4.0:
        best_h, best_score = _swap_local_search(best_h, max_rounds=20, swaps_per_round=400)
    
    # Final tiny perturbation polish
    if time.time() < deadline - 1.0:
        for _ in range(30):
            if time.time() > deadline - 0.5:
                break
            perturbation = rng.normal(0, 0.005, size=N)
            h_perturbed = best_h + perturbation
            h_perturbed = _project_to_feasible(h_perturbed)
            score_perturbed = _max_overlap(h_perturbed)
            if score_perturbed < best_score:
                best_score = score_perturbed
                best_h = h_perturbed
    
    return best_h
