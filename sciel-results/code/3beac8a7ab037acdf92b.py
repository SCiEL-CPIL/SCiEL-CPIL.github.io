import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 115.0
K = 20  # Number of frequency components

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _phases_to_h(phases):
    """Convert phase vector to h in [0,1] with sum HALF."""
    # Construct complex spectrum with flat magnitude and given phases
    # Use first K positive frequencies and their conjugates for real signal
    spec = np.zeros(N, dtype=complex)
    # DC component: set to 0 to avoid bias
    for k in range(1, K + 1):
        mag = 1.0  # Flat magnitude
        spec[k] = mag * np.exp(1j * phases[k - 1])
        spec[N - k] = np.conj(spec[k])  # Conjugate symmetry for real signal
    
    h = np.fft.ifft(spec).real
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Normalize to sum HALF
    s = np.sum(h)
    if s > 0:
        h = h * (HALF / s)
    # Re-clip after scaling
    h = np.clip(h, 0.0, 1.0)
    # If still not exactly HALF, adjust
    diff = HALF - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute difference evenly
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Final adjustment on largest elements
        s = np.sum(h)
        if abs(s - HALF) > 1e-10:
            idx = np.argmax(h) if s < HALF else np.argmin(h)
            h[idx] += (HALF - s)
            h = np.clip(h, 0.0, 1.0)
    return h

def _ga_phases(rng, pop_size=40, generations=150, deadline=None):
    """Genetic algorithm to optimize phase vectors."""
    # Initialize population
    pop = [rng.random(K) * 2 * np.pi for _ in range(pop_size)]
    fitness = [_max_overlap(_phases_to_h(p)) for p in pop]
    
    best_idx = np.argmin(fitness)
    best_phases = pop[best_idx].copy()
    best_cost = fitness[best_idx]
    
    for gen in range(generations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort by fitness
        order = np.argsort(fitness)
        pop = [pop[i] for i in order]
        fitness = [fitness[i] for i in order]
        
        # Elitism: keep top 2
        new_pop = [pop[0].copy(), pop[1].copy()]
        
        # Fill rest with crossover and mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            i1 = rng.integers(0, min(10, len(pop)))
            i2 = rng.integers(0, min(10, len(pop)))
            if fitness[i1] < fitness[i2]:
                p1 = pop[i1]
            else:
                p1 = pop[i2]
            
            i3 = rng.integers(0, min(10, len(pop)))
            i4 = rng.integers(0, min(10, len(pop)))
            if fitness[i3] < fitness[i4]:
                p2 = pop[i3]
            else:
                p2 = pop[i4]
            
            # Crossover: uniform
            mask = rng.random(K) > 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if rng.random() < 0.3:
                child += rng.normal(0, 0.5, K)
                child %= (2 * np.pi)
            
            new_pop.append(child)
        
        pop = new_pop
        fitness = [_max_overlap(_phases_to_h(p)) for p in pop]
        
        # Update best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_cost:
            best_cost = fitness[best_idx]
            best_phases = pop[best_idx].copy()
    
    return best_phases, best_cost

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    deadline = t0 + DEADLINE
    
    best_h = None
    bestc = np.inf
    
    # Run GA with multiple random restarts
    n_restarts = 5
    time_per_restart = (DEADLINE - 5) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break
        restart_deadline = t0 + (r + 1) * time_per_restart + 2
        
        phases, cost = _ga_phases(rng, pop_size=30, generations=200, deadline=restart_deadline)
        h = _phases_to_h(phases)
        c = _max_overlap(h)
        
        if c < bestc:
            bestc = c
            best_h = h
    
    # If GA didn't find anything good, fall back to binary SA
    if best_h is None or bestc > 1.0:
        # Simple binary SA fallback
        h = np.zeros(N)
        h[:HALF] = 1.0
        cur = _max_overlap(h)
        best = h.copy()
        bestc = cur
        
        for it in range(50000):
            if time.time() > deadline - 1:
                break
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            h[i], h[j] = h[j], h[i]
            newc = _max_overlap(h)
            T = 2.0 * (0.01 / 2.0) ** (it / 50000)
            if newc < cur or rng.random() < np.exp(-(newc - cur) / max(T, 1e-9)):
                cur = newc
                if newc < bestc:
                    bestc = newc
                    best = h.copy()
            else:
                h[i], h[j] = h[j], h[i]
        
        best_h = best
        bestc = bestc if bestc < cur else cur
    
    # Final local search on best_h if it's continuous
    if best_h is not None and bestc < np.inf:
        # Try small perturbations
        h = best_h.copy()
        cur = _max_overlap(h)
        for _ in range(5000):
            if time.time() > deadline - 0.5:
                break
            idx = rng.integers(N)
            delta = rng.normal(0, 0.05)
            old_val = h[idx]
            h[idx] = np.clip(old_val + delta, 0.0, 1.0)
            # Renormalize
            s = np.sum(h)
            if s > 0:
                h = h * (HALF / s)
            h = np.clip(h, 0.0, 1.0)
            newc = _max_overlap(h)
            if newc < cur:
                cur = newc
                if cur < bestc:
                    bestc = cur
                    best_h = h.copy()
            else:
                h[idx] = old_val
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure sum is exactly HALF
    s = np.sum(best_h)
    if abs(s - HALF) > 1e-10:
        best_h = best_h + (HALF - s) / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
