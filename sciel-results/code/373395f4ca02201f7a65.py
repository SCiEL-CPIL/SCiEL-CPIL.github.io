"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def _compute_min_autocorr(h):
    """
    Compute the minimum circular autocorrelation of h.
    C_h(t) = sum_i h[i] * h[(i+t) % N]
    We want to maximize min_t C_h(t).
    """
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.fft(h)
    # Autocorrelation is ifft( H * conj(H) )
    # Wait, circular autocorrelation r[t] = sum_i x[i] x[i+t]
    # This is ifft( X * conj(X) )? 
    # Let's check: ifft( X * conj(X) )[t] = sum_k X[k] conj(X[k]) e^{2pi i k t / N}
    # Parseval: sum_k X[k] conj(X[k]) e^{...} = sum_i x[i] x[i-t]? 
    # Standard result: ifft( fft(x) * conj(fft(x)) ) is the circular autocorrelation.
    # Specifically, r[t] = sum_i x[i] x[(i+t) % N].
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    return np.min(autocorr)

def _project_to_constraints(h):
    """
    Project h to satisfy sum(h) = N/2 and 0 <= h[i] <= 1.
    """
    h = np.clip(h, 0.0, 1.0)
    # Iterative projection to fix sum
    for _ in range(20):
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    np.random.seed(42)
    
    # Initialize population with random binary-like sequences
    pop_size = 60
    generations = 150
    
    # Start with random 0/1 vectors with sum N/2
    def random_individual():
        h = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1.0
        # Add some noise to explore continuous space
        h += np.random.normal(0, 0.1, N)
        return _project_to_constraints(h)

    population = [random_individual() for _ in range(pop_size)]
    
    # Evaluate initial fitness
    def fitness(h):
        # We want to maximize min autocorrelation
        # Lower bound on overlap is N/2 - max_autocorr? No.
        # Overlap(t) = N/2 - C_h(t).
        # Max Overlap = N/2 - Min C_h(t).
        # So minimizing Max Overlap is equivalent to Maximizing Min C_h(t).
        return _compute_min_autocorr(h)
        
    fitnesses = np.array([fitness(h) for h in population])
    
    best_h = population[np.argmax(fitnesses)]
    best_fit = np.max(fitnesses)
    
    # Genetic Algorithm Loop
    for gen in range(generations):
        # Selection: Tournament selection
        # Crossover: Uniform
        # Mutation: Gaussian noise
        
        new_population = []
        
        # Elitism: keep best 2
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(2):
            new_population.append(population[sorted_indices[i]].copy())
        
        while len(new_population) < pop_size:
            # Select parents
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            # Simple selection: pick random, but bias towards better?
            # Let's use tournament of size 3
            def tournament():
                idxs = np.random.choice(pop_size, 3, replace=False)
                fits = fitnesses[idxs]
                return idxs[np.argmax(fits)]
            
            p1_idx = tournament()
            p2_idx = tournament()
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            mask = np.random.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            # Randomly perturb
            noise = np.random.normal(0, 0.2, N)
            child += noise
            
            # Constraint projection
            child = _project_to_constraints(child)
            new_population.append(child)
            
        population = new_population
        fitnesses = np.array([fitness(h) for h in population])
        
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_h = population[gen_best_idx].copy()
            
    return best_h
# EVOLVE-BLOCK-END
