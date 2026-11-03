"""
Strategy: Global Discrete Search via Genetic Algorithm
The problem is to minimize max|autoconv| / (sum h)^2 for a signed step function.
The current program uses local sign-flipping which gets stuck in local minima.
We reformulate this as a discrete optimization over the sign vector S in {-1, 1}^N.
Note: The "heights" are just the signs if we assume unit width steps, 
or we can scale them. The ratio is scale-invariant if we scale h by c:
max|ac(c h)| / (sum c h)^2 = c^2 max|ac(h)| / c^2 (sum h)^2 = max|ac(h)| / (sum h)^2.
So we only need to find the best sign pattern.

We use a Genetic Algorithm:
1. Population of random sign vectors.
2. Fitness = -1 / raw_score (we want to minimize raw_score, so maximize inverse).
3. Crossover: Uniform crossover.
4. Mutation: Random bit flips.
5. Selection: Tournament selection.

This explores the global landscape of the discrete hypercube, avoiding the 
local traps of greedy local search.
"""

import numpy as np
import time

N = 400

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill

    # We work with sign vectors s in {-1, 1}^N.
    # The actual h is just s (or s * k, but ratio is invariant).
    # sum(h) = sum(s). If sum(s) is 0, score is inf. We must avoid sum(s)=0.
    # For N=400, random signs have sum ~ N(0, N). P(sum=0) is small but non-zero.
    # If sum is close to 0, score is huge.
    
    # To ensure we have a valid candidate, we can bias the signs slightly or 
    # just rely on the fact that most random signs have non-zero sum.
    # Actually, if sum(s) is small, the denominator is small, making the score large.
    # We want to MINIMIZE the score.
    # Note: For constant sequence s = [1, 1, ..., 1], sum = N, max_ac = N, score = N/N^2 = 1/N.
    # For random signs, sum ~ sqrt(N), max_ac ~ N (peak at 0 is N, others ~ sqrt(N)).
    # Score ~ N / N = 1.
    # So constant sequence (1/N) is much better than random (1).
    # Wait, if we use signs, we can't get "constant" unless all signs are same.
    # The problem says "signed step functions". It doesn't say they must be balanced.
    # So all +1 is a valid candidate.
    
    # Let's initialize population with:
    # 1. All +1
    # 2. Random signs
    # 3. Structured signs (e.g., chirp-like mapped to signs)
    
    pop_size = 200
    rng = np.random.default_rng(42)
    
    # Initialize population
    population = []
    
    # Candidate 1: All ones
    population.append(np.ones(N, dtype=int))
    
    # Candidate 2: All -1 (equivalent to all ones in score)
    population.append(-np.ones(N, dtype=int))
    
    # Candidate 3: Random signs
    for _ in range(pop_size - 2):
        population.append(rng.choice([-1, 1], size=N))
    
    # Evaluate initial population
    scores = []
    for ind in population:
        # Convert to float for FFT
        h = ind.astype(float)
        scores.append(_score(h))
    
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_ind = population[best_idx].copy()
    
    # Genetic Algorithm Loop
    # Parameters
    generations = 0
    max_generations = 10000  # Will be limited by time
    mutation_rate = 0.05
    tournament_k = 3
    
    while time.time() < deadline:
        generations += 1
        
        # Selection: Tournament
        # We create a new population
        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(best_ind.copy())
        
        while len(new_population) < pop_size:
            # Select parents
            p1_idx = _tournament_select(scores, population, tournament_k, rng)
            p2_idx = _tournament_select(scores, population, tournament_k, rng)
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover: Uniform
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if rng.random() < mutation_rate:
                # Flip some bits
                num_flips = rng.integers(1, N // 10)
                flip_idx = rng.choice(N, size=num_flips, replace=False)
                child[flip_idx] *= -1
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = []
        for ind in population:
            h = ind.astype(float)
            s_val = _score(h)
            scores.append(s_val)
            
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_ind = population[current_best_idx].copy()
            
        # If we've reached a very good score, we might stop early?
        # The theoretical minimum for any sequence with sum S is 1/|S| * (max_ac/|S|).
        # For constant, it is 1/N.
        # Can we do better?
        # max_ac >= sum(h^2) = N (for signs).
        # sum(h) <= N.
        # Score = max_ac / sum(h)^2 >= N / N^2 = 1/N.
        # So 1/N is the global minimum for sign vectors.
        # If we find 1/N, we can stop.
        if best_score <= 1.0 / N * 1.001:
            break

    # Return the best sign vector as the sequence of signed step heights
    # The interface expects "sequence of signed step heights".
    # The score is invariant to scaling, so returning the sign vector is sufficient.
    return best_ind.astype(float)

def _tournament_select(scores, population, k, rng):
    """Tournament selection."""
    idxs = rng.choice(len(population), size=k, replace=False)
    best_idx = idxs[np.argmin([scores[i] for i in idxs])]
    return best_idx
