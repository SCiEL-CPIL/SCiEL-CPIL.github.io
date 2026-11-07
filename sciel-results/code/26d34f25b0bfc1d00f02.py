"""
Discrete combinatorial search over support structure of h using a genetic
algorithm. The key insight: the ratio l2/(norm_1 * norm_inf) is maximized
when g = conv(h,h) has a dominant peak (small norm_inf relative to norm_1)
while maintaining a broad l2 contribution. This suggests h should be structured
so that its autocorrelation concentrates mass at lag 0 while spreading energy.

Strategy:
1. Evolve binary support masks + continuous amplitudes via genetic algorithm.
2. For each candidate h, optimize amplitudes via local coordinate descent.
3. Use log-space conditioning for the ratio objective.
4. Deadline guard with safety margin.
"""
import numpy as np
import time

N = 50
M = 2 * N - 1  # len(g) = 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 130.0  # safety margin before 150s kill

RNG = np.random.RandomState(42)


def compute_value(h):
    """Exact objective value."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def compute_value_log(h):
    """Log-conditioned version for numerical stability."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return -np.inf
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return -np.inf
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return -np.inf
    return np.log(l2) - np.log(norm_1) - np.log(norm_inf)


def local_optimize(h, max_iter=300, step_init=0.1):
    """Local coordinate descent on exact value with adaptive step."""
    h = np.maximum(0.0, np.asarray(h, dtype=float).copy())
    best_val = compute_value(h)
    step = step_init
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            orig = h[i]
            # Try several perturbations
            for delta in [step, -step, step * 0.5, -step * 0.5, step * 2.0]:
                trial = h.copy()
                trial[i] = max(0.0, orig + delta)
                v = compute_value(trial)
                if v > best_val + 1e-12:
                    best_val = v
                    h = trial
                    improved = True
                    break
        if not improved:
            step *= 0.8
            if step < 1e-7:
                break
    return h, best_val


def crossover(parent1, parent2):
    """Uniform crossover on the full h vector."""
    mask = RNG.rand(N) < 0.5
    child = np.where(mask, parent1, parent2)
    return child


def mutate(h, rate=0.15, sigma=0.3):
    """Gaussian mutation with nonnegativity."""
    child = h.copy()
    mask = RNG.rand(N) < rate
    child[mask] += RNG.normal(0, sigma, mask.sum())
    child = np.maximum(0.0, child)
    return child


def support_crossover(h1, h2):
    """Crossover that respects support structure: keep non-zero indices from one parent."""
    support1 = h1 > 1e-8
    support2 = h2 > 1e-8
    # Randomly choose which support to use
    if RNG.rand() < 0.5:
        child = h1[support1].copy()
        child = np.zeros(N)
        child[support1] = h1[support1]
        # Add some from parent2's support
        extra_support = support2 & ~support1
        if extra_support.sum() > 0:
            n_extra = min(RNG.randint(0, extra_support.sum() + 1), extra_support.sum())
            if n_extra > 0:
                extra_indices = np.where(extra_support)[0]
                chosen = RNG.choice(extra_indices, size=n_extra, replace=False)
                child[chosen] = h2[chosen] * RNG.uniform(0.5, 1.5, n_extra)
    else:
        child = h2.copy()
        extra_support = support1 & ~support2
        if extra_support.sum() > 0:
            n_extra = min(RNG.randint(0, extra_support.sum() + 1), extra_support.sum())
            if n_extra > 0:
                extra_indices = np.where(extra_support)[0]
                chosen = RNG.choice(extra_indices, size=n_extra, replace=False)
                child[chosen] = h1[chosen] * RNG.uniform(0.5, 1.5, n_extra)
    return np.maximum(0.0, child)


def generate_initial_population(pop_size=40):
    """Generate diverse initial population."""
    pop = []
    x = np.linspace(0, N - 1, N)
    
    # Structured starts
    pop.append(np.ones(N))
    pop.append(np.exp(-0.5 * ((x - N / 2) / (N / 5)) ** 2))
    pop.append(np.exp(-0.5 * ((x - N * 0.3) / (N / 6)) ** 2) + np.exp(-0.5 * ((x - N * 0.7) / (N / 6)) ** 2))
    pop.append(np.exp(-0.5 * ((x - N * 0.2) / (N / 8)) ** 2) + np.exp(-0.5 * ((x - N * 0.8) / (N / 8)) ** 2))
    pop.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    pop.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    
    # Sparse spikes
    for _ in range(5):
        h = np.zeros(N)
        k = RNG.randint(1, 6)
        idx = RNG.choice(N, size=k, replace=False)
        h[idx] = RNG.uniform(0.5, 2.0, k)
        pop.append(h)
    
    # Random smooth profiles
    for _ in range(pop_size - len(pop)):
        freqs = RNG.randint(1, 10, size=5)
        phases = RNG.uniform(0, 2 * np.pi, size=5)
        amps = RNG.uniform(0.3, 1.0, size=5)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        pop.append(np.maximum(0.05, prof))
    
    return pop


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, np.asarray(h, dtype=float).ravel())
        if h.size != N:
            h = h[:N]
            if h.size < N:
                h = np.pad(h, (0, N - h.size), 'constant')
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 1: Evaluate initial population
    pop = generate_initial_population(40)
    for s in pop:
        track(s)
        if time.time() > deadline:
            break
    
    # Phase 2: Genetic algorithm
    pop_size = 30
    elite_size = 5
    generations = 80
    
    # Sort initial population by value
    pop_vals = [compute_value(p) for p in pop[:pop_size]]
    order = np.argsort(pop_vals)[::-1]
    pop = [pop[i] for i in order[:pop_size]]
    
    for gen in range(generations):
        if time.time() > deadline - 30:  # Leave time for final polish
            break
        
        gen_start = time.time()
        new_pop = []
        
        # Elitism: keep best individuals
        for i in range(min(elite_size, len(pop))):
            new_pop.append(pop[i].copy())
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 30:
                break
            
            # Select parents (tournament)
            idx1 = RNG.randint(0, len(pop))
            idx2 = RNG.randint(0, len(pop))
            
            # Choose crossover method
            r = RNG.rand()
            if r < 0.4:
                child = crossover(pop[idx1], pop[idx2])
            elif r < 0.7:
                child = support_crossover(pop[idx1], pop[idx2])
            else:
                child = pop[idx1].copy()
            
            # Mutate
            child = mutate(child, rate=0.2, sigma=0.25)
            new_pop.append(child)
        
        # Local optimization on a subset of new population
        n_local = min(8, len(new_pop))
        local_indices = RNG.choice(len(new_pop), size=n_local, replace=False)
        for idx in local_indices:
            if time.time() > deadline - 25:
                break
            h_opt, v_opt = local_optimize(new_pop[idx], max_iter=150, step_init=0.08)
            new_pop[idx] = h_opt
            track(h_opt)
        
        # Track all
        for h in new_pop:
            track(h)
        
        # Select next generation by value
        pop_vals = [compute_value(p) for p in new_pop]
        order = np.argsort(pop_vals)[::-1]
        pop = [new_pop[i] for i in order[:pop_size]]
        
        # Print progress every 10 generations
        if (gen + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Gen {gen+1}/{generations}, best={best_val:.6f}, elapsed={elapsed:.1f}s")
    
    # Phase 3: Final intensive local optimization from best and nearby points
    n_final = 20
    for i in range(n_final):
        if time.time() > deadline - 5:
            break
        if i == 0:
            h0 = best_h.copy()
        else:
            h0 = best_h + RNG.normal(0.0, 0.1, N)
            h0 = np.maximum(0.0, h0)
        h_opt, v_opt = local_optimize(h0, max_iter=400, step_init=0.05)
        track(h_opt)
    
    # Phase 4: Very fine final polish
    if time.time() < deadline - 2:
        h_opt, v_opt = local_optimize(best_h, max_iter=500, step_init=0.01)
        track(h_opt)
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
