"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 17
SEED_COMPARATORS = 136  # odd-even transposition for N wires

# EVOLVE-BLOCK-START
def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _apply_comparator(state, i, j):
    """Apply a single compare-exchange to a boolean state (numpy array of 0/1)."""
    a = state[i]
    b = state[j]
    if a > b:
        state[i] = b
        state[j] = a
    return state

def _simulate_network(comps, inputs):
    """
    Simulate the network on a batch of boolean inputs.
    inputs: numpy array of shape (num_inputs, N) with 0/1 values.
    Returns: True if all inputs are sorted (non-decreasing), False otherwise.
    """
    state = inputs.copy()
    for i, j in comps:
        # Compare and swap for all inputs at once
        # a = state[:, i], b = state[:, j]
        a = state[:, i]
        b = state[:, j]
        # If a > b, swap
        mask = a > b
        if np.any(mask):
            state[mask, i] = b[mask]
            state[mask, j] = a[mask]
    
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k + 1]):
            return False
    return True

def _is_valid_network(comps, num_tests=1024):
    """
    Test if the network sorts all boolean inputs (using a subset for speed).
    Returns True if valid, False otherwise.
    """
    # Generate random boolean inputs
    rng = np.random.default_rng(42)
    inputs = rng.integers(0, 2, size=(num_tests, N), dtype=np.uint8)
    return _simulate_network(comps, inputs)

def _is_fully_valid_network(comps):
    """
    Test if the network sorts ALL 131072 boolean inputs.
    This is the definitive test.
    """
    # Generate all 2^17 boolean inputs
    all_inputs = np.array([list(map(int, bin(i)[2:].zfill(N))) for i in range(1 << N)], dtype=np.uint8)
    return _simulate_network(comps, all_inputs)

def _count_comparators(comps):
    return len(comps)

def _remove_redundant_comparator(comps, idx, num_tests=512):
    """
    Try to remove the comparator at index idx.
    Returns True if removal is valid, False otherwise.
    """
    if idx < 0 or idx >= len(comps):
        return False
    new_comps = comps[:idx] + comps[idx+1:]
    return _is_valid_network(new_comps, num_tests)

def _local_search(comps, max_steps=1000, time_limit=30):
    """
    Local search to remove redundant comparators.
    """
    start_time = time.time()
    current_comps = list(comps)
    current_count = len(current_comps)
    
    # First, do a pass to remove obviously redundant comparators
    removed = True
    while removed:
        removed = False
        for i in range(len(current_comps) - 1, -1, -1):
            if _remove_redundant_comparator(current_comps, i, num_tests=256):
                current_comps.pop(i)
                removed = True
                current_count -= 1
        if time.time() - start_time > time_limit:
            break
    
    return current_comps

def _genetic_algorithm(comps, population_size=50, generations=100, time_limit=45):
    """
    Genetic algorithm to evolve a better sorting network.
    """
    start_time = time.time()
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Initialize population
    population = [list(comps)]
    for _ in range(population_size - 1):
        # Create variants by removing random comparators
        variant = list(comps)
        for _ in range(rng.randint(0, 10)):
            if len(variant) > 63:  # Don't go below lower bound
                idx = rng.randint(0, len(variant) - 1)
                variant.pop(idx)
        population.append(variant)
    
    best_comps = list(comps)
    best_count = len(comps)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate fitness
        fitnesses = []
        for ind in population:
            # Fitness: number of comparators (lower is better) + penalty for invalid
            if _is_valid_network(ind, num_tests=128):
                fitness = len(ind)
            else:
                fitness = len(ind) + 1000  # Penalty for invalid
            fitnesses.append(fitness)
        
        # Find best
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_count:
            best_count = fitnesses[best_idx]
            best_comps = list(population[best_idx])
        
        # Selection: keep top half
        indices = np.argsort(fitnesses)
        survivors = [population[i] for i in indices[:population_size // 2]]
        
        # Generate new population
        new_population = list(survivors)
        while len(new_population) < population_size:
            # Crossover
            if len(survivors) >= 2:
                parent1 = survivors[rng.randint(0, len(survivors) - 1)]
                parent2 = survivors[rng.randint(0, len(survivors) - 1)]
                child = list(parent1)
                # Swap some comparators from parent2
                for i in range(len(child)):
                    if i < len(parent2) and rng.random() < 0.3:
                        child[i] = parent2[i]
                new_population.append(child)
            else:
                new_population.append(list(survivors[0]))
        
        # Mutation
        for i in range(len(new_population)):
            if rng.random() < 0.5:
                # Remove a random comparator
                if len(new_population[i]) > 63:
                    idx = rng.randint(0, len(new_population[i]) - 1)
                    new_population[i].pop(idx)
        
        population = new_population
    
    return best_comps

def construct():
    """
    Construct a sorting network with as few comparators as possible.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline (10 second safety margin)
    
    # Start with odd-even transposition
    comps = _odd_even_transposition()
    best_comps = list(comps)
    best_count = len(comps)
    
    # Phase 1: Local search to remove redundant comparators
    local_result = _local_search(comps, time_limit=20)
    if len(local_result) < best_count:
        best_count = len(local_result)
        best_comps = local_result
    
    if time.time() > deadline:
        return best_comps
    
    # Phase 2: Genetic algorithm
    ga_result = _genetic_algorithm(best_comps, time_limit=40)
    if len(ga_result) < best_count:
        best_count = len(ga_result)
        best_comps = ga_result
    
    return best_comps
# EVOLVE-BLOCK-END
