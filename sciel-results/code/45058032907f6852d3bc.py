import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def quick_ratio(h):
    """Fast ratio computation for coordinate descent."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Global random search with diverse initializations ===
    # Generate diverse random step functions
    n_random = 50
    population = []
    
    # Start with uniform
    population.append(np.ones(N))
    
    # Random smooth functions via low-frequency Fourier
    for i in range(n_random // 2):
        h = np.ones(N) * rng.uniform(0.5, 2.0)
        for k in range(1, 6):
            amp = rng.uniform(-0.3, 0.3)
            phase = rng.uniform(0, 2 * np.pi)
            h += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
        population.append(np.clip(h, 0, None))
    
    # Random piecewise constant with few segments
    for i in range(n_random // 4):
        n_segments = rng.integers(2, 8)
        boundaries = np.sort(rng.uniform(0, N, n_segments - 1))
        boundaries = np.concatenate([[0], boundaries, [N]])
        h = np.empty(N)
        for j in range(n_segments):
            h[int(boundaries[j]):int(boundaries[j+1])] = rng.uniform(0.1, 2.0)
        population.append(h)
    
    # Random Gaussian bumps
    for i in range(n_random // 4):
        h = np.zeros(N)
        n_bumps = rng.integers(1, 4)
        for _ in range(n_bumps):
            amp = rng.uniform(0.5, 3.0)
            mu = rng.uniform(-0.2, 0.2)
            sig = rng.uniform(0.03, 0.12)
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        population.append(h)
    
    # Evaluate all
    for h in population:
        v = ratio_from_h(h)
        update_best(h, v)
    
    # === Phase 2: Genetic Algorithm ===
    pop_size = 40
    n_gen = 30
    
    # Initialize population from best random + more random
    ga_pop = [best_h.copy()]
    for i in range(pop_size - 1):
        h = best_h.copy() * rng.uniform(0.5, 1.5)
        h += rng.normal(0, 0.2, N)
        ga_pop.append(np.clip(h, 0, None))
    
    # Evaluate initial GA population
    ga_scores = [ratio_from_h(h) for h in ga_pop]
    
    # Find best in GA pop
    ga_best_idx = np.argmin(ga_scores)
    update_best(ga_pop[ga_best_idx], ga_scores[ga_best_idx])
    
    # Genetic algorithm loop
    for gen in range(n_gen):
        if time.time() > deadline - 30.0:
            break
        
        # Tournament selection
        def tournament_select(pop, scores, k=3):
            candidates = rng.integers(0, len(pop), size=k)
            best_cand = candidates[np.argmin([scores[c] for c in candidates])]
            return pop[best_cand].copy()
        
        # Create new generation
        new_pop = []
        new_scores = []
        
        # Elitism: keep best
        new_pop.append(ga_pop[ga_best_idx].copy())
        new_scores.append(ga_scores[ga_best_idx])
        
        # Generate offspring
        while len(new_pop) < pop_size:
            parent1 = tournament_select(ga_pop, ga_scores)
            parent2 = tournament_select(ga_pop, ga_scores)
            
            # Crossover: uniform
            mask = rng.random(N) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation: Gaussian noise on random subset
            n_mutate = rng.integers(10, 50)
            mutate_idx = rng.choice(N, size=n_mutate, replace=False)
            child[mutate_idx] += rng.normal(0, 0.1, n_mutate)
            child = np.clip(child, 0, None)
            
            new_pop.append(child)
        
        # Evaluate new population
        for i in range(len(new_pop)):
            if time.time() > deadline - 25.0:
                break
            new_scores.append(ratio_from_h(new_pop[i]))
            update_best(new_pop[i], new_scores[-1])
        
        # Update best in GA
        ga_best_idx = np.argmin(new_scores)
        
        # Replace population
        ga_pop = new_pop
        ga_scores = new_scores
    
    # === Phase 3: Coordinate descent refinement ===
    if time.time() < deadline - 15.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        # Multi-scale coordinate descent
        step_scales = [1.0, 0.5, 0.25, 0.1]
        n_iter = 8000
        
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Adaptive step scale based on iteration
            scale_idx = min(it // 2000, len(step_scales) - 1)
            scale = step_scales[scale_idx]
            
            # Pick random coordinates
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                # Multiplicative change with adaptive range
                factor_range = (1.0 - 0.15 * scale, 1.0 + 0.15 * scale)
                factor = float(rng2.uniform(*factor_range))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 4: Hand-crafted shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-2.0 * x_norm**2),
            np.exp(-0.5 * x_norm**2),
            0.5 + 0.5 * np.exp(-8.0 * x_norm**2),
            np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2),
            0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2),
            1.0 - 0.3 * np.exp(-5.0 * x_norm**2),
            0.5 + 0.5 * x_norm**2,
            0.5 + 0.5 * np.cos(np.pi * x_norm),
        ]
        
        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
