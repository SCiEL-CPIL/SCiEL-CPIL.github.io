"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Uses a Genetic Algorithm to minimize the sum of squares of the aperiodic
    autocorrelation sidelobes (Energy).
    """
    np.random.seed(42)
    
    # Fitness function: Sum of squares of aperiodic autocorrelation sidelobes
    # R(k) for k != 0. 
    # ac = np.correlate(seq, seq, mode='full')
    # R(0) is at index N-1.
    # Energy = sum(ac**2) - ac[N-1]**2
    
    def calculate_energy(seq):
        # Using FFT for faster correlation if N is large, but N=400 is small.
        # np.correlate is O(N^2) = 160,000 ops. Very fast.
        ac = np.correlate(seq, seq, mode='full')
        return np.sum(ac**2) - ac[N-1]**2

    # Initialize population
    pop_size = 100
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Calculate initial fitness
    fitnesses = np.array([calculate_energy(ind) for ind in population])
    
    best_seq = population[np.argmin(fitnesses)].copy()
    best_fitness = np.min(fitnesses)
    
    start_time = time.time()
    max_time = 100.0
    
    # GA Parameters
    mutation_rate = 0.01
    crossover_rate = 0.9
    tournament_size = 5
    elitism = 2
    
    generation = 0
    
    while (time.time() - start_time) < max_time:
        generation += 1
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)
        population = population[sorted_indices]
        fitnesses = fitnesses[sorted_indices]
        
        # Update best
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_seq = population[0].copy()
        
        # Elitism: Keep the best 'elitism' individuals
        new_population = [population[i].copy() for i in range(elitism)]
        
        # Generate rest of the population
        while len(new_population) < pop_size:
            # Tournament Selection
            def tournament():
                contenders = np.random.randint(0, pop_size, size=tournament_size)
                best_idx = contenders[np.argmin(fitnesses[contenders])]
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover
            if np.random.rand() < crossover_rate:
                # Uniform crossover or Single point? Let's use Uniform for binary sequences
                mask = np.random.rand(N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.rand() < mutation_rate:
                # Flip random bits
                num_mutations = np.random.randint(1, 5)
                mut_indices = np.random.choice(N, size=num_mutations, replace=False)
                child[mut_indices] *= -1
            
            new_population.append(child)
        
        population = np.array(new_population)
        
        # Evaluate new fitness
        # We can reuse fitness for elites, but for simplicity and speed, 
        # recalculating is fine for N=400 and pop=100 (100 * 160k ops = 16M ops per gen).
        # 100s / 16M ops * 1000 (ns/op) ~ 6s. So we can do ~10-20 generations.
        # Let's optimize by only calculating for new individuals.
        # Actually, let's just recalculate all to keep code simple, it's fast enough.
        fitnesses = np.array([calculate_energy(ind) for ind in population])

    return best_seq
# EVOLVE-BLOCK-END
