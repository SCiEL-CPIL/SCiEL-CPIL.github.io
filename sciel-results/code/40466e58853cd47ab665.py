"""
Strategy: Genetic Algorithm (GA) with Bitwise Crossover.

Rationale:
1. The Thue-Morse sequence is a known strong baseline for minimizing monochromatic APs.
2. Instead of point-mutations (local search), we use bitwise XOR crossover. 
   This allows combining "good" structures from different parts of the search space.
3. We use a population-based approach to escape local minima that single-trajectory 
   methods (like the current local search) might get stuck in.
4. NumPy vectorization allows evaluating the entire population's fitness in parallel 
   for each difference d, making the GA very fast (hundreds of generations in < 1s).
"""

import numpy as np
import time
import random

N = 256

def count_mono_4ap_population(pop):
    """
    Vectorized count for a population of colorings.
    pop: shape (P, N) array of 0s and 1s.
    Returns: array of shape (P,) with counts.
    """
    counts = np.zeros(pop.shape[0], dtype=np.int32)
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a < 1:
            break
        # Extract sequences for all individuals in the population
        # pop[:, :max_a] corresponds to a
        # pop[:, d:d+max_a] corresponds to a+d
        # pop[:, 2*d:2*d+max_a] corresponds to a+2d
        # pop[:, 3*d:3*d+max_a] corresponds to a+3d
        
        seq1 = pop[:, :max_a]
        seq2 = pop[:, d:d+max_a]
        seq3 = pop[:, 2*d:2*d+max_a]
        seq4 = pop[:, 3*d:3*d+max_a]
        
        # Check if all 4 are equal
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        counts += np.sum(mono, axis=1)
        
    return counts

def construct():
    deadline = time.time() + 145
    
    # 1. Generate Thue-Morse sequence (O(1) per element using bit_count)
    # t(n) = popcount(n) % 2
    indices = np.arange(N)
    tm = (np.bit_count(indices) % 2).astype(np.uint8)
    
    # 2. Initialize Population
    # We start with the Thue-Morse sequence and random perturbations of it.
    # This seeds the GA with a high-quality starting point.
    POP_SIZE = 256
    population = np.zeros((POP_SIZE, N), dtype=np.uint8)
    
    # Seed 0: Pure Thue-Morse
    population[0] = tm
    
    # Seed 1..POP_SIZE//2: Thue-Morse with random single-bit flips (low noise)
    for i in range(1, POP_SIZE // 2):
        population[i] = tm.copy()
        flip_idx = random.randint(0, N - 1)
        population[i, flip_idx] ^= 1
        
    # Seed POP_SIZE//2..POP_SIZE//4: Thue-Morse with random block flips (medium noise)
    for i in range(POP_SIZE // 2, 3 * POP_SIZE // 4):
        population[i] = tm.copy()
        # Flip a random contiguous block
        start = random.randint(0, N - 1)
        length = random.randint(1, 8)
        end = min(start + length, N)
        population[i, start:end] ^= 1
        
    # Rest: Random
    for i in range(3 * POP_SIZE // 4, POP_SIZE):
        population[i] = np.random.randint(0, 2, size=N, dtype=np.uint8)
    
    # Evaluate initial population
    scores = count_mono_4ap_population(population)
    best_score = np.min(scores)
    best_idx = np.argmin(scores)
    best_c = population[best_idx].copy()
    
    # 3. Genetic Algorithm Loop
    # Parameters
    ELITE_SIZE = 4
    TOURNAMENT_SIZE = 5
    MUTATION_RATE = 0.02  # Probability of flipping a bit
    
    # Track progress
    stagnation_counter = 0
    
    while time.time() < deadline - 2:
        # --- Selection and Crossover ---
        new_population = np.zeros_like(population)
        
        # 1. Elitism: Keep the best individuals unchanged
        elite_indices = np.argsort(scores)[:ELITE_SIZE]
        for i, idx in enumerate(elite_indices):
            new_population[i] = population[idx]
        
        # 2. Generate offspring via Tournament Selection and XOR Crossover
        for i in range(ELITE_SIZE, POP_SIZE):
            # Tournament selection for parent 1
            candidates = np.random.randint(0, POP_SIZE, size=TOURNAMENT_SIZE)
            parent1_idx = candidates[np.argmin(scores[candidates])]
            
            # Tournament selection for parent 2
            candidates = np.random.randint(0, POP_SIZE, size=TOURNAMENT_SIZE)
            parent2_idx = candidates[np.argmin(scores[candidates])]
            
            # XOR Crossover
            # This effectively combines the "differences" between two good solutions.
            child = population[parent1_idx] ^ population[parent2_idx]
            
            # Mutation
            # Flip each bit with probability MUTATION_RATE
            # Using a boolean mask for vectorized mutation
            mask = np.random.rand(N) < MUTATION_RATE
            child[mask] ^= 1
            
            new_population[i] = child
        
        population = new_population
        
        # --- Evaluation ---
        scores = count_mono_4ap_population(population)
        
        # --- Update Best ---
        current_best_score = np.min(scores)
        current_best_idx = np.argmin(scores)
        
        if current_best_score < best_score:
            best_score = current_best_score
            best_c = population[current_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Optional: If stagnating, increase mutation rate or restart part of population
        # For simplicity, we rely on the diversity from XOR crossover.
        # If we stagnate for too long, we might inject fresh randoms or Thue-Morse variants.
        if stagnation_counter > 50:
             # Inject some fresh Thue-Morse variants to escape local minima
             for i in range(POP_SIZE // 4):
                 idx = random.randint(0, POP_SIZE - 1)
                 population[idx] = tm.copy()
                 # Random perturbation
                 num_flips = random.randint(1, 5)
                 flip_indices = np.random.choice(N, num_flips, replace=False)
                 population[idx, flip_indices] ^= 1
             stagnation_counter = 0

    return list(best_c)

if __name__ == "__main__":
    c = construct()
    # Simple sanity check print
    # print(f"Best score: {count_mono_4ap_population(np.array([c]))[0]}")
    pass
