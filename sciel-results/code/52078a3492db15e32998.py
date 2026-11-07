"""Second autocorrelation inequality: optimize 50 nonnegative step heights.

Strategy: Genetic algorithm over discrete support patterns and relative magnitudes.
The key insight is that the objective l2/(norm_1 * norm_inf) is scale-invariant,
so we only need to optimize the shape. A genetic algorithm can explore
discrete sparsity patterns (which indices are non-zero) and continuous
magnitudes simultaneously, avoiding the vanishing gradient problem that
traps continuous optimizers at boundary points.

We combine:
1. A genetic algorithm that evolves (support_mask, magnitudes) pairs
2. Local L-BFGS-B polish on the best candidates
3. Parametric family search for good initial populations
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100 spacing
TIME_LIMIT = 150.0
DEADLINE = time.time() + (TIME_LIMIT - 12.0)  # 12s safety margin
_T = 1.0 / 40.0  # Temperature for soft-max approximation


def compute_value(h):
    """Compute the true objective value (non-smooth)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    y = np.concatenate(([0.0], g, [0.0]))
    l2 = (W / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0.0 or norm_inf <= 0.0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def compute_value_smooth(h):
    """Compute a smooth approximation using log-sum-exp for max."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    y = np.concatenate(([0.0], g, [0.0]))
    l2 = (W / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    
    abs_g = np.abs(g)
    max_val = np.max(abs_g)
    if max_val <= 0:
        return 0.0
    shifted = (abs_g - max_val) / _T
    shifted = np.clip(shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    norm_inf_smooth = max_val + _T * log_sum_exp
    
    if norm_1 <= 0.0 or norm_inf_smooth <= 0.0:
        return 0.0
    return l2 / (norm_1 * norm_inf_smooth)


def neg_value_smooth(x):
    s = np.sum(x)
    if s <= 0.0:
        return 1e10
    h = x / s
    return -compute_value_smooth(h)


def neg_value_exact(x):
    s = np.sum(x)
    if s <= 0.0:
        return 1e10
    h = x / s
    return -compute_value(h)


def polish(h, use_smooth=True, maxiter=500):
    """Polish a candidate using L-BFGS-B."""
    try:
        if use_smooth:
            obj = neg_value_smooth
        else:
            obj = neg_value_exact
        res = minimize(obj, h.copy(), method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10,
                                "maxls": 50})
        h2 = res.x / np.sum(res.x)
        return compute_value(h2), h2
    except Exception:
        return compute_value(h), h


def parametric_search():
    """Search over parametric families: plateaus, Gaussians, combinations."""
    best_v, best_h = 0.0, np.ones(N) / N
    t = np.linspace(0, 1, N)
    
    # Plateau family: a at ends, b in middle
    for a in range(0, N // 2 + 1, 1):
        for b in np.linspace(0.0, 2.0, 21):
            h = np.ones(N)
            if a > 0:
                h[:a] = 1.0
                h[N - a:] = 1.0
            if N - 2 * a > 0:
                h[a:N - a] = b
            v = compute_value(h)
            if v > best_v:
                best_v, best_h = v, h.copy()
    
    # Gaussian mixture family
    rng = np.random.default_rng(42)
    for _ in range(200):
        n_bumps = rng.integers(1, 4)
        h = np.zeros(N)
        for _ in range(n_bumps):
            c = rng.uniform(0.15, 0.85)
            sig = rng.uniform(0.03, 0.25)
            amp = rng.uniform(0.5, 2.0)
            h += amp * np.exp(-((t - c) ** 2) / (2 * sig ** 2))
        h = np.clip(h, 0, None)
        if np.sum(h) > 0:
            v = compute_value(h)
            if v > best_v:
                best_v, best_h = v, h.copy()
    
    return best_v, best_h


def genetic_algorithm(pop_size=60, n_gen=30, elite=5):
    """Genetic algorithm over (support_mask, magnitudes) representation.
    
    Each individual is represented as:
    - mask: binary array of length N indicating non-zero positions
    - mags: array of length N with magnitudes (zero where mask is 0)
    
    Crossover: uniform crossover on the full vector
    Mutation: 
      - Flip bits in mask (add/remove support)
      - Perturb magnitudes multiplicatively
    """
    rng = np.random.default_rng(123)
    t = np.linspace(0, 1, N)
    
    # Initialize population
    pop = []
    scores = []
    
    # Seed with parametric best
    v0, h0 = parametric_search()
    pop.append(h0.copy())
    scores.append(v0)
    
    # Add diverse random individuals
    for _ in range(pop_size - 1):
        h = np.zeros(N)
        n_bumps = rng.integers(1, 5)
        for _ in range(n_bumps):
            c = rng.uniform(0.1, 0.9)
            sig = rng.uniform(0.03, 0.3)
            amp = rng.uniform(0.3, 2.0)
            h += amp * np.exp(-((t - c) ** 2) / (2 * sig ** 2))
        h = np.clip(h, 0, None)
        if np.sum(h) > 0:
            h = h / np.sum(h)
        else:
            h = np.ones(N) / N
        pop.append(h)
        scores.append(compute_value(h))
    
    best_v = max(scores)
    best_h = pop[scores.index(best_v)].copy()
    
    for gen in range(n_gen):
        if time.time() > DEADLINE:
            break
        
        # Sort by score
        idx = np.argsort(scores)[::-1]
        pop = [pop[i] for i in idx]
        scores = [scores[i] for i in idx]
        
        # Elitism: keep top elite
        new_pop = [pop[i].copy() for i in range(elite)]
        new_scores = list(scores[:elite])
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() > DEADLINE:
                break
            
            # Select parents via tournament
            i1 = rng.choice(pop_size, size=3, replace=False)
            i2 = rng.choice(pop_size, size=3, replace=False)
            p1 = pop[np.argmax([scores[i] for i in i1])]
            p2 = pop[np.argmax([scores[i] for i in i2])]
            
            # Crossover: uniform
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation: multiplicative perturbation on magnitudes
            # and support flipping
            flip_prob = 0.05
            flip_mask = rng.random(N) < flip_prob
            child[flip_mask] = 0.0  # zero out some positions
            
            # Add new support
            if rng.random() < 0.3:
                n_add = rng.integers(1, 4)
                add_idx = rng.choice(N, size=n_add, replace=False)
                child[add_idx] = rng.uniform(0.1, 1.0, size=n_add)
            
            # Multiplicative perturbation
            child *= rng.lognormal(0, 0.3, size=N)
            child = np.clip(child, 0, None)
            
            if np.sum(child) > 0:
                child = child / np.sum(child)
            else:
                child = np.ones(N) / N
            
            new_pop.append(child)
            new_scores.append(compute_value(child))
        
        pop = new_pop
        scores = new_scores
        
        # Track best
        gen_best = max(scores)
        if gen_best > best_v:
            best_v = gen_best
            best_h = pop[scores.index(gen_best)].copy()
    
    return best_v, best_h


def multi_start_lbfgs(n_restarts=30):
    """Multi-start L-BFGS-B with diverse initializations."""
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, N)
    best_v, best_h = 0.0, np.ones(N) / N
    
    for r in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        # Diverse initialization
        style = rng.integers(0, 4)
        if style == 0:
            # Gaussian mixture
            n_bumps = rng.integers(1, 4)
            h = np.zeros(N)
            for _ in range(n_bumps):
                c = rng.uniform(0.1, 0.9)
                sig = rng.uniform(0.03, 0.25)
                amp = rng.uniform(0.5, 2.0)
                h += amp * np.exp(-((t - c) ** 2) / (2 * sig ** 2))
        elif style == 1:
            # Plateau-like
            a = rng.integers(0, N // 2 + 1)
            b = rng.uniform(0, 2.0)
            h = np.ones(N)
            if a > 0:
                h[:a] = 1.0
                h[N - a:] = 1.0
            if N - 2 * a > 0:
                h[a:N - a] = b
        elif style == 2:
            # Random with smoothing
            h = rng.random(N)
            # Simple box smoothing
            kernel = np.ones(5) / 5
            h = np.convolve(h, kernel, mode='same')
        else:
            # Sparse random
            h = np.zeros(N)
            n_nonzero = rng.integers(5, 25)
            idx = rng.choice(N, size=n_nonzero, replace=False)
            h[idx] = rng.uniform(0.1, 2.0, size=n_nonzero)
        
        h = np.clip(h, 0, None)
        if np.sum(h) > 0:
            h = h / np.sum(h)
        
        v, h2 = polish(h, use_smooth=True, maxiter=300)
        if v > best_v:
            best_v, best_h = v, h2.copy()
    
    return best_v, best_h


def construct():
    global DEADLINE
    DEADLINE = time.time() + (TIME_LIMIT - 12.0)
    
    best_v, best_h = 0.0, np.ones(N) / N
    
    # Phase 1: Parametric search for a good starting point
    if time.time() < DEADLINE:
        v1, h1 = parametric_search()
        best_v, best_h = v1, h1.copy()
    
    # Phase 2: Polish the parametric best
    if time.time() < DEADLINE:
        v2, h2 = polish(best_h, use_smooth=True, maxiter=500)
        if v2 > best_v:
            best_v, best_h = v2, h2.copy()
    
    # Phase 3: Genetic algorithm to explore diverse landscape
    if time.time() < DEADLINE:
        v3, h3 = genetic_algorithm(pop_size=50, n_gen=25, elite=5)
        if v3 > best_v:
            best_v, best_h = v3, h3.copy()
    
    # Phase 4: Multi-start L-BFGS-B from diverse points
    if time.time() < DEADLINE:
        v4, h4 = multi_start_lbfgs(n_restarts=25)
        if v4 > best_v:
            best_v, best_h = v4, h4.copy()
    
    # Phase 5: Polish the overall best with exact value
    if time.time() < DEADLINE:
        v5, h5 = polish(best_h, use_smooth=False, maxiter=500)
        if v5 > best_v:
            best_v, best_h = v5, h5.copy()
    
    # Phase 6: One more round of GA seeded with current best
    if time.time() < DEADLINE:
        # Quick GA with smaller population
        rng = np.random.default_rng(99)
        t = np.linspace(0, 1, N)
        ga_best_v, ga_best_h = best_v, best_h.copy()
        
        pop = [best_h.copy()]
        for _ in range(20):
            h = best_h * rng.lognormal(0, 0.5, size=N)
            # Flip some support
            flip = rng.random(N) < 0.1
            h[flip] = 0
            # Add some new support
            if rng.random() < 0.5:
                n_add = rng.integers(1, 5)
                idx = rng.choice(N, size=n_add, replace=False)
                h[idx] = rng.uniform(0.1, 1.0, size=n_add)
            h = np.clip(h, 0, None)
            if np.sum(h) > 0:
                h = h / np.sum(h)
            else:
                h = best_h.copy()
            pop.append(h)
        
        for _ in range(10):
            if time.time() > DEADLINE:
                break
            for i in range(len(pop)):
                v, h2 = polish(pop[i], use_smooth=True, maxiter=200)
                if v > ga_best_v:
                    ga_best_v, ga_best_h = v, h2.copy()
                    # Add polished version to population
                    pop.append(h2.copy())
            if len(pop) > 40:
                # Keep best 40
                scored = [(compute_value(p), p) for p in pop]
                scored.sort(key=lambda x: -x[0])
                pop = [p for _, p in scored[:40]]
        
        if ga_best_v > best_v:
            best_v, best_h = ga_best_v, ga_best_h.copy()
    
    # Final clip and normalize
    h = np.clip(best_h, 0.0, None)
    if np.sum(h) <= 0:
        h = np.ones(N)
    
    return [float(x) for x in h]
