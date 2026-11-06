"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, leave 5s safety margin for 90s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses an improved local search with:
    1. Bit-parallel verification for speed
    2. Greedy removal with restart
    3. Genetic algorithm for escaping local optima
    4. Depth-aware comparator placement
    """
    start_time = time.time()
    
    # Baseline: odd-even transposition
    def odd_even_network(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Start with the odd-even network
    current = odd_even_network(N)
    
    # Verify it works (should always work)
    if not verify_network_fast(current):
        return current
    
    best_count = len(current)
    best_network = current[:]
    
    # Phase 1: Aggressive greedy removal
    current = greedy_remove(current, start_time)
    if len(current) < best_count:
        best_count = len(current)
        best_network = current[:]
    
    # Phase 2: Local search with random restarts and genetic improvement
    current = local_search_with_genetics(current, best_network, best_count, start_time)
    
    # Final polish: try to remove any remaining redundant comparators
    current = greedy_remove(current, start_time)
    if len(current) < best_count:
        best_count = len(current)
        best_network = current[:]
    
    return best_network


def greedy_remove(network, start_time):
    """
    Greedy removal: repeatedly try to remove comparators from the network.
    Uses bit-parallel verification for speed.
    """
    current = network[:]
    
    # Multiple passes to ensure we find all removable comparators
    for pass_num in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Try removing from the end first (often easier to remove)
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                return current
            
            candidate = current[:idx] + current[idx+1:]
            
            if verify_network_fast(candidate):
                current = candidate
                improved = True
                # Restart from the end after removal
                break
        
        if not improved:
            # Try from the beginning
            for idx in range(len(current)):
                if time.time() - start_time > DEADLINE:
                    return current
                
                candidate = current[:idx] + current[idx+1:]
                
                if verify_network_fast(candidate):
                    current = candidate
                    improved = True
                    break
            
            if not improved:
                break
    
    return current


def local_search_with_genetics(initial, best_network, best_count, start_time):
    """
    Local search with genetic algorithm to escape local optima.
    Uses depth-aware mutation and crossover.
    """
    current = initial[:]
    current_count = len(current)
    
    # Population for genetic algorithm
    population = [current[:]]
    pop_size = 4
    
    # Fill population with variations
    while len(population) < pop_size:
        mutant = mutate_network(current)
        if verify_network_fast(mutant):
            population.append(mutant)
        else:
            # Try a different mutation
            mutant = random_network(len(current) + random.randint(-2, 2))
            if verify_network_fast(mutant):
                population.append(mutant)
            else:
                population.append(current[:])
    
    best_in_pop = min(population, key=len)
    if len(best_in_pop) < best_count:
        best_count = len(best_in_pop)
        best_network = best_in_pop[:]
    
    max_generations = 50
    
    for gen in range(max_generations):
        if time.time() - start_time > DEADLINE:
            return best_network
        
        # Evaluate and select
        population.sort(key=lambda x: len(x))
        
        # Keep the best 2
        survivors = population[:2]
        
        # Generate offspring
        while len(population) < pop_size:
            if time.time() - start_time > DEADLINE:
                return best_network
            
            # Crossover or mutation
            if random.random() < 0.5 and len(population) >= 2:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = crossover(parent1, parent2)
            else:
                parent = random.choice(survivors)
                child = mutate_network(parent)
            
            if verify_network_fast(child):
                population.append(child)
            else:
                # Try to repair
                repaired = repair_network(child)
                if verify_network_fast(repaired):
                    population.append(repaired)
                else:
                    population.append(parent[:])
        
        # Track best
        best_in_pop = min(population, key=len)
        if len(best_in_pop) < best_count:
            best_count = len(best_in_pop)
            best_network = best_in_pop[:]
            current = best_in_pop[:]
    
    return best_network


def mutate_network(network):
    """
    Apply a random mutation to the network.
    """
    if len(network) == 0:
        return network[:]
    
    candidate = network[:]
    mutation_type = random.random()
    
    if mutation_type < 0.3:
        # Remove a random comparator
        idx = random.randint(0, len(candidate) - 1)
        candidate = candidate[:idx] + candidate[idx+1:]
    elif mutation_type < 0.6:
        # Add a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(candidate))
        candidate.insert(pos, (i, j))
    elif mutation_type < 0.8:
        # Replace a random comparator
        idx = random.randint(0, len(candidate) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate[idx] = (i, j)
    else:
        # Swap two comparators
        if len(candidate) >= 2:
            idx1 = random.randint(0, len(candidate) - 1)
            idx2 = random.randint(0, len(candidate) - 1)
            if idx1 != idx2:
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
    
    return candidate


def crossover(parent1, parent2):
    """
    Simple crossover: take a prefix from one parent and suffix from another.
    """
    if len(parent1) == 0:
        return parent2[:]
    if len(parent2) == 0:
        return parent1[:]
    
    # Find a good crossover point
    min_len = min(len(parent1), len(parent2))
    if min_len < 2:
        return random.choice([parent1[:], parent2[:]])
    
    point = random.randint(1, min_len - 1)
    
    child = parent1[:point] + parent2[point:]
    
    return child


def repair_network(network):
    """
    Try to repair an invalid network by adding necessary comparators.
    """
    current = network[:]
    
    # Find which inputs are not sorted
    missing_comparators = find_missing_comparators(current)
    
    for comp in missing_comparators[:5]:  # Add at most 5
        pos = random.randint(0, len(current))
        current.insert(pos, comp)
    
    return current


def find_missing_comparators(network):
    """
    Find comparators that might be missing by analyzing unsorted outputs.
    """
    # This is a heuristic: try adding comparators between wires that often end up unsorted
    candidates = []
    
    # Check common problematic pairs
    for i in range(N):
        for j in range(i + 1, N):
            # If wires i and j are far apart, they might need comparison
            if j - i > 2:
                candidates.append((i, j))
    
    return candidates[:10]


def random_network(num_comparators):
    """
    Generate a random network with the given number of comparators.
    """
    comps = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps


def verify_network_fast(network):
    """
    Fast verification using bit-parallelism with larger batches.
    """
    n = len(network)
    if n == 0:
        return N <= 1
    
    # Use larger batches for better numpy performance
    batch_size = 4096
    total_inputs = 1 << N
    
    for batch_start in range(0, total_inputs, batch_size):
        batch_end = min(batch_start + batch_size, total_inputs)
        batch = np.arange(batch_start, batch_end, dtype=np.uint32)
        batch_size_actual = batch_end - batch_start
        
        # Initialize wires: shape (batch_size, N)
        wires = np.zeros((batch_size_actual, N), dtype=np.uint8)
        for i in range(N):
            wires[:, i] = (batch >> i) & 1
        
        # Apply each comparator
        for i, j in network:
            # Compare and swap where needed
            mask = wires[:, i] > wires[:, j]
            if np.any(mask):
                # Use temporary array for swap
                temp = wires[mask, i].copy()
                wires[mask, i] = wires[mask, j]
                wires[mask, j] = temp
        
        # Check if sorted: for each row, check if non-decreasing
        # Use cumulative max trick: if max of each row equals last element, it's sorted
        # Actually, just check adjacent pairs
        for k in range(N - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
    
    return True


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
