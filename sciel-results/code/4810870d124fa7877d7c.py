"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to improve an autoconvolution upper bound.
    
    Returns:
        sequence of signed step heights (numpy array of shape (N,) with values in {-1, 1})
    
    The score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin within 120s limit
    
    # The autoconvolution upper bound C3 relates to the L1 norm of the autocorrelation
    # For a step function f with values in {-1, 1} on N equal intervals:
    # The autocorrelation R(τ) = ∫ f(t) f(t+τ) dt
    # The "upper bound on C3" likely refers to some norm of the autocorrelation
    # or a related quantity.
    
    # For binary sequences, the autoconvolution (periodic or aperiodic) can be computed.
    # Let's assume the metric is the L1 norm of the aperiodic autocorrelation,
    # or perhaps the maximum absolute value of the autocorrelation,
    # or the sum of squares of the autocorrelation values.
    
    # Given the task mentions "autoconvolution upper bound" and "C3",
    # and the paper places absolute value differently, let's consider:
    # C3 might be related to the third moment or a specific integral.
    
    # A common metric for such problems is the L1 norm of the autocorrelation:
    # sum_{k=0}^{N-1} |R(k)| where R(k) = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    
    # Or it could be the peak sidelobe level.
    
    # Let's define the score as the L1 norm of the aperiodic autocorrelation.
    # Lower is better.
    
    def compute_score(f):
        """Compute the L1 norm of the aperiodic autocorrelation."""
        N_local = len(f)
        total = 0.0
        for lag in range(N_local):
            # Autocorrelation at this lag
            corr = np.dot(f[:N_local-lag], f[lag:])
            total += abs(corr)
        return total
    
    # Initial population
    pop_size = 50
    population = [np.ones(N, dtype=np.int8) for _ in range(pop_size)]
    
    # Try a few initial candidates with random signs
    for i in range(1, pop_size):
        population[i] = np.random.choice([-1, 1], size=N).astype(np.int8)
    
    # Evaluate initial population
    scores = [compute_score(f) for f in population]
    best_idx = np.argmin(scores)
    best_f = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Genetic algorithm
    tournament_size = 5
    mutation_rate = 0.05
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Create new population
        new_population = []
        new_scores = []
        
        # Elitism: keep the best individual
        new_population.append(best_f.copy())
        new_scores.append(best_score)
        
        # Generate offspring
        for _ in range(pop_size - 1):
            # Tournament selection
            candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
            candidate_scores = [scores[c] for c in candidates]
            parent1_idx = candidates[np.argmin(candidate_scores)]
            
            candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
            candidate_scores = [scores[c] for c in candidates]
            parent2_idx = candidates[np.argmin(candidate_scores)]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover: uniform crossover
            mask = np.random.random(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            if np.random.random() < mutation_rate:
                mutation_positions = np.random.choice(N, size=int(N * mutation_rate), replace=False)
                child[mutation_positions] = -child[mutation_positions]
            
            new_population.append(child)
            new_scores.append(compute_score(child))
        
        # Update population
        population = new_population
        scores = new_scores
        
        # Update best
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_f = population[best_idx].copy()
        
        # Adaptive mutation rate: increase if no improvement
        if generation % 20 == 0:
            mutation_rate = min(0.2, mutation_rate * 1.1)
    
    return best_f
# EVOLVE-BLOCK-END
