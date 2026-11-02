"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Genetic Algorithm approach to minimize the Peak Sidelobe Level (PSL)
    of the aperiodic autocorrelation.
    This is a distinct approach from Simulated Annealing, using population-based
    search with crossover and mutation.
    """
    np.random.seed(42)
    
    # Initialize population with random sequences
    pop_size = 200
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Pre-compute FFT size for efficiency
    fft_size = 2 * N
    
    def calculate_psl(seq):
        # Compute aperiodic autocorrelation using FFT
        x = np.zeros(fft_size)
        x[:N] = seq
        X = np.fft.fft(x)
        power = X * np.conj(X)
        r = np.fft.ifft(power).real
        
        # r[0] is R(0) = N
        # r[1:N] are R(1) to R(N-1)
        # r[-(N-1):-1] are R(-(N-1)) to R(-1)
        
        # Extract sidelobes
        sidelobes = np.concatenate([r[1:N], r[-(N-1):-1]])
        return np.max(np.abs(sidelobes))

    # Calculate initial fitness
    fitnesses = np.array([calculate_psl(ind) for ind in population])
    
    start_time = time.time()
    max_time = 115.0 # Leave buffer for return
    
    best_individual = population[np.argmin(fitnesses)].copy()
    best_fitness = np.min(fitnesses)
    
    while (time.time() - start_time) < max_time:
        # Selection: Tournament selection
        # We select parents for the next generation
        new_population = []
        
        # Elitism: Keep the best 10%
        elite_size = max(1, pop_size // 10)
        elite_indices = np.argsort(fitnesses)[:elite_size]
        for i in elite_indices:
            new_population.append(population[i].copy())
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idxs = np.random.randint(0, pop_size, size=5)
            min_fit_idx = idxs[np.argmin(fitnesses[idxs])]
            parent1 = population[min_fit_idx]
            
            idxs2 = np.random.randint(0, pop_size, size=5)
            min_fit_idx2 = idxs2[np.argmin(fitnesses[idxs2])]
            parent2 = population[min_fit_idx2]
            
            # Crossover: Uniform crossover
            mask = np.random.rand(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation: Bit flip with low probability
            mut_mask = np.random.rand(N) < 0.005
            child[mut_mask] = -child[mut_mask]
            
            new_population.append(child)
        
        population = np.array(new_population)
        
        # Evaluate new population
        fitnesses = np.array([calculate_psl(ind) for ind in population])
        
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx].copy()

    return best_individual
# EVOLVE-BLOCK-END
