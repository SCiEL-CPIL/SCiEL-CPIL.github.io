"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Genetic Algorithm approach to minimize the energy of the aperiodic autocorrelation sidelobes.
    Metric: E = sum_{k != 0} R(k)^2
    """
    np.random.seed(42)
    
    # Initialize population
    pop_size = 60
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    def calculate_energy(seq):
        """
        Calculate the sum of squares of the aperiodic autocorrelation sidelobes.
        """
        # Use FFT for faster convolution?
        # np.correlate is O(N^2). For N=400, N^2=160,000 operations.
        # FFT is O(N log N).
        # Let's stick to np.correlate first as N is small, but if it's slow, switch to FFT.
        # Actually, for N=400, np.correlate is very fast.
        ac = np.correlate(seq, seq, mode='full')
        # ac[N-1] is R(0) = N
        return np.sum(ac**2) - ac[N-1]**2

    # Evaluate initial population
    energies = np.array([calculate_energy(ind) for ind in population])
    best_idx = np.argmin(energies)
    best_individual = population[best_idx].copy()
    best_energy = energies[best_idx]
    
    start_time = time.time()
    max_time = 110.0  # Leave 10s buffer for overhead/return
    
    # GA Parameters
    mutation_rate = 0.01
    tournament_size = 3
    crossover_prob = 0.8
    
    generation = 0
    
    while (time.time() - start_time) < max_time:
        next_population = []
        
        # Elitism: Keep the best individual
        next_population.append(best_individual.copy())
        
        # Generate rest of population
        while len(next_population) < pop_size:
            # Selection
            # Tournament selection
            idxs = np.random.randint(0, pop_size, tournament_size)
            winner_idx = idxs[np.argmin(energies[idxs])]
            parent1 = population[winner_idx]
            
            idxs = np.random.randint(0, pop_size, tournament_size)
            winner_idx = idxs[np.argmin(energies[idxs])]
            parent2 = population[winner_idx]
            
            # Crossover
            if np.random.rand() < crossover_prob:
                mask = np.random.rand(N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            # Flip bits with probability mutation_rate
            flip_mask = np.random.rand(N) < mutation_rate
            child[flip_mask] *= -1
            
            next_population.append(child)
        
        population = np.array(next_population)
        
        # Evaluate new population
        energies = np.array([calculate_energy(ind) for ind in population])
        
        # Update best
        current_best_idx = np.argmin(energies)
        current_best_energy = energies[current_best_idx]
        
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_individual = population[current_best_idx].copy()
        
        generation += 1
        
        # Optional: Adaptive mutation rate?
        # If stuck, increase mutation.
        # For simplicity, keep constant.
        
    return best_individual
# EVOLVE-BLOCK-END
