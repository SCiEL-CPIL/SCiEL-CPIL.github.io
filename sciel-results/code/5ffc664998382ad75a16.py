"""Size-efficient sorting networks on 16 wires.
Uses a genetic algorithm with constraint-based pruning to find minimal
comparator networks. Starts from diverse initial populations and evolves
them, using fast verification and necessary-comparator checks.
"""
import numpy as np
import time
import random
from collections import deque

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

# ---------- Batcher Bitonic Sort Network ----------
def _bitonic_network(n):
    """Generate Batcher bitonic sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    # Sort first half ascending, second half descending, then merge
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    # Shift right comparators
    right = [(i + half, j + half) for (i, j) in right]
    # Merge: odd-even merge for bitonic
    merge = _bitonic_merge(n)
    return left + right + merge

def _bitonic_merge(n):
    """Bitonic merge for n wires (n power of 2). Assumes input is bitonic."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # First do cross comparisons
    cross = [(i, i + half) for i in range(half)]
    # Then recursively merge both halves
    left_merge = _bitonic_merge(half)
    right_merge = _bitonic_merge(half)
    right_merge = [(i + half, j + half) for (i, j) in right_merge]
    return cross + left_merge + right_merge

def _make_bitonic_network():
    """Build full bitonic sort: split, sort halves opposite, merge."""
    half = N // 2
    left = _bitonic_network(half)
    # Generate ascending network for the right half
    right_asc = _bitonic_network(half)
    # Reverse the comparators in the right half to sort it in descending order.
    # Swapping (i, j) to (j, i) makes min go to the larger index and max to the smaller,
    # which results in a descending sequence on the wire indices.
    right_desc = [(j + half, i + half) for (i, j) in right_asc]
    merge = _bitonic_merge(N)
    return left + right_desc + merge

def _make_random_network(n_comparators):
    """Generate a random network with n_comparators comparators."""
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    return [random.choice(all_pairs) for _ in range(n_comparators)]

def _make_odd_even_transposition():
    """Odd-even transposition network (120 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# ---------- Verification ----------
def _verify(comparators):
    """Check if network sorts all 2^16 zero-one inputs. Returns True if valid."""
    # Precomputed popcount table for 16-bit values
    if not hasattr(_verify, '_popcount'):
        _verify._popcount = np.array([bin(v).count('1') for v in range(65536)], dtype=np.uint8)
    
    all_inputs = np.arange(65536, dtype=np.uint32)
    
    for (i, j) in comparators:
        bit_i = (all_inputs >> i) & 1
        bit_j = (all_inputs >> j) & 1
        # min goes to i, max goes to j
        new_i = np.minimum(bit_i, bit_j)
        new_j = np.maximum(bit_i, bit_j)
        
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        # Clear bits i and j
        all_inputs = all_inputs & (~mask_i & ~mask_j)
        # Set new bits
        all_inputs = all_inputs | (new_i.astype(np.uint32) * mask_i) | (new_j.astype(np.uint32) * mask_j)
        
    ones = _verify._popcount[all_inputs]
    
    targets = np.zeros(17, dtype=np.uint32)
    for k in range(17):
        if k == 0:
            targets[k] = 0
        elif k == 16:
            targets[k] = 65535
        else:
            targets[k] = ((1 << k) - 1) << (16 - k)
            
    expected = targets[ones]
    
    return np.array_equal(all_inputs, expected)

def _verify_partial(comparators, subset_size=1024):
    """Quick verification on a random subset of inputs for fast rejection."""
    if not hasattr(_verify_partial, '_popcount'):
        _verify_partial._popcount = np.array([bin(v).count('1') for v in range(65536)], dtype=np.uint8)
    
    # Use a fixed random seed for reproducibility in subset selection
    rng = np.random.RandomState(42)
    indices = rng.choice(65536, size=subset_size, replace=False)
    all_inputs = np.arange(65536, dtype=np.uint32)[indices]
    
    for (i, j) in comparators:
        bit_i = (all_inputs >> i) & 1
        bit_j = (all_inputs >> j) & 1
        new_i = np.minimum(bit_i, bit_j)
        new_j = np.maximum(bit_i, bit_j)
        
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        all_inputs = all_inputs & (~mask_i & ~mask_j)
        all_inputs = all_inputs | (new_i.astype(np.uint32) * mask_i) | (new_j.astype(np.uint32) * mask_j)
        
    ones = _verify_partial._popcount[all_inputs]
    
    targets = np.zeros(17, dtype=np.uint32)
    for k in range(17):
        if k == 0:
            targets[k] = 0
        elif k == 16:
            targets[k] = 65535
        else:
            targets[k] = ((1 << k) - 1) << (16 - k)
            
    expected = targets[ones]
    
    return np.array_equal(all_inputs, expected)

def _count_unsorted_states(comparators):
    """Count how many of the 65536 inputs are NOT sorted after applying network.
    Returns 0 if fully sorted, >0 otherwise."""
    if not hasattr(_count_unsorted_states, '_popcount'):
        _count_unsorted_states._popcount = np.array([bin(v).count('1') for v in range(65536)], dtype=np.uint8)
    
    all_inputs = np.arange(65536, dtype=np.uint32)
    
    for (i, j) in comparators:
        bit_i = (all_inputs >> i) & 1
        bit_j = (all_inputs >> j) & 1
        new_i = np.minimum(bit_i, bit_j)
        new_j = np.maximum(bit_i, bit_j)
        
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        all_inputs = all_inputs & (~mask_i & ~mask_j)
        all_inputs = all_inputs | (new_i.astype(np.uint32) * mask_i) | (new_j.astype(np.uint32) * mask_j)
        
    ones = _count_unsorted_states._popcount[all_inputs]
    
    targets = np.zeros(17, dtype=np.uint32)
    for k in range(17):
        if k == 0:
            targets[k] = 0
        elif k == 16:
            targets[k] = 65535
        else:
            targets[k] = ((1 << k) - 1) << (16 - k)
            
    expected = targets[ones]
    
    return np.sum(all_inputs != expected)

# ---------- Local Search ----------
def _try_remove(comparators, idx):
    """Try removing comparator at index idx. Return new list if still valid, else None."""
    new_comp = comparators[:idx] + comparators[idx+1:]
    if _verify(new_comp):
        return new_comp
    return None

def _try_swap(comparators, idx1, idx2):
    """Try swapping comparators at idx1 and idx2."""
    new_comp = list(comparators)
    new_comp[idx1], new_comp[idx2] = new_comp[idx2], new_comp[idx1]
    if _verify(new_comp):
        return new_comp
    return None

def _try_replace(comparators, idx, new_comp):
    """Try replacing comparator at idx with new_comp."""
    new_comp_list = list(comparators)
    new_comp_list[idx] = new_comp
    if _verify(new_comp_list):
        return new_comp_list
    return None

def _local_search(comparators, time_limit):
    """Aggressive local search to reduce comparator count."""
    best = list(comparators)
    start_time = time.time()
    
    # All possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # First pass: try to remove each comparator
    i = 0
    while i < len(best):
        if time.time() - start_time > time_limit:
            break
        new_best = _try_remove(best, i)
        if new_best is not None:
            best = new_best
            # Don't increment i, re-check same position
        else:
            i += 1
    
    # Second pass: try swapping adjacent comparators to enable further removals
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(len(best) - 1):
            if time.time() - start_time > time_limit:
                break
            new_best = _try_swap(best, i, i+1)
            if new_best is not None and new_best != best:
                best = new_best
                improved = True
                break  # Restart scan
    
    # Third pass: try replacing comparators with different ones to enable removals
    while time.time() - start_time < time_limit:
        idx = random.randint(0, len(best) - 1)
        new_pair = random.choice(all_pairs)
        if new_pair == best[idx]:
            continue
        new_best = _try_replace(best, idx, new_pair)
        if new_best is not None:
            best = new_best
            j = 0
            while j < len(best) and time.time() - start_time < time_limit:
                nb = _try_remove(best, j)
                if nb is not None:
                    best = nb
                else:
                    j += 1
    return best

def _genetic_algorithm(initial_populations, time_limit, pop_size=20):
    """Genetic algorithm to evolve sorting networks.
    
    Uses a population of candidate networks, with fitness = -len(network)
    (lower is better). Selection, crossover, and mutation are used to
    explore the search space more broadly than local search alone.
    """
    start_time = time.time()
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Initialize population
    population = []
    for net in initial_populations:
        if _verify(net):
            population.append(list(net))
    
    # If we don't have enough valid individuals, add random ones
    while len(population) < pop_size:
        # Try to create a valid random network by starting from a known good one
        # and making random modifications
        base = random.choice(initial_populations) if initial_populations else _make_bitonic_network()
        candidate = list(base)
        # Make a few random mutations
        for _ in range(random.randint(1, 5)):
            move_type = random.choice(['remove', 'replace', 'add'])
            if move_type == 'remove' and len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif move_type == 'replace' and len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] = random.choice(all_pairs)
            elif move_type == 'add' and len(candidate) < 80:
                candidate.append(random.choice(all_pairs))
        if _verify(candidate):
            population.append(candidate)
        else:
            # Fallback: use a copy of a valid network
            population.append(list(base))
    
    best = min(population, key=len)
    best_len = len(best)
    
    # Evolution loop
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Evaluate fitness (lower length is better)
        fitnesses = [len(ind) for ind in population]
        
        # Update best
        min_idx = fitnesses.index(min(fitnesses))
        if fitnesses[min_idx] < best_len:
            best = list(population[min_idx])
            best_len = fitnesses[min_idx]
        
        # Selection: tournament selection
        new_population = []
        while len(new_population) < pop_size:
            # Tournament of 3
            candidates = random.sample(range(len(population)), 3)
            winner = min(candidates, key=lambda i: fitnesses[i])
            new_population.append(list(population[winner]))
        
        # Crossover and mutation
        for i in range(len(new_population)):
            # Crossover: uniform crossover with another individual
            if random.random() < 0.7 and len(new_population) > 1:
                j = random.randint(0, len(new_population) - 1)
                if j != i:
                    other = population[j]
                    # Uniform crossover
                    child = []
                    max_len = max(len(new_population[i]), len(other))
                    for k in range(max_len):
                        if k < len(new_population[i]) and k < len(other):
                            if random.random() < 0.5:
                                child.append(new_population[i][k])
                            else:
                                child.append(other[k])
                        elif k < len(new_population[i]):
                            child.append(new_population[i][k])
                        elif k < len(other):
                            child.append(other[k])
                    # Truncate to reasonable length
                    if len(child) > 70:
                        child = child[:70]
                    new_population[i] = child
            
            # Mutation
            n_mutations = random.randint(1, 3)
            for _ in range(n_mutations):
                if len(new_population[i]) == 0:
                    break
                move_type = random.choice(['remove', 'replace', 'add', 'swap'])
                
                if move_type == 'remove' and len(new_population[i]) > 0:
                    idx = random.randint(0, len(new_population[i]) - 1)
                    new_population[i].pop(idx)
                
                elif move_type == 'replace' and len(new_population[i]) > 0:
                    idx = random.randint(0, len(new_population[i]) - 1)
                    new_population[i][idx] = random.choice(all_pairs)
                
                elif move_type == 'add' and len(new_population[i]) < 75:
                    new_population[i].append(random.choice(all_pairs))
                
                elif move_type == 'swap' and len(new_population[i]) >= 2:
                    i1, i2 = random.sample(range(len(new_population[i])), 2)
                    new_population[i][i1], new_population[i][i2] = new_population[i][i2], new_population[i][i1]
            
            # Verify and repair if needed
            if not _verify(new_population[i]):
                # Try to repair by removing problematic comparators
                repaired = list(new_population[i])
                idx = 0
                while idx < len(repaired) and not _verify(repaired):
                    # Try removing this comparator
                    test = repaired[:idx] + repaired[idx+1:]
                    if _verify(test):
                        repaired = test
                        # Don't increment idx, recheck same position
                    else:
                        idx += 1
                
                if _verify(repaired):
                    new_population[i] = repaired
                else:
                    # Fallback to a valid individual
                    new_population[i] = list(population[random.randint(0, len(population)-1)])
        
        # Elitism: keep the best individual
        new_population.sort(key=len)
        new_population[0] = best
        
        population = new_population
        
        # Occasionally run local search on the best individual
        if generation % 5 == 0 and time.time() - start_time < time_limit - 5:
            local_result = _local_search(best, min(5, time_limit - (time.time() - start_time)))
            if len(local_result) < best_len:
                best = local_result
                best_len = len(best)
                population[0] = list(best)
    
    return best

# ---------- Main construct function ----------
def construct():
    start_time = time.time()
    # Safety margin: finish by 55s to avoid 60s kill
    deadline = start_time + 55.0
    
    # Generate diverse initial populations
    initial_populations = []
    
    # Batcher bitonic sort (63 comparators)
    bitonic = _make_bitonic_network()
    if _verify(bitonic):
        initial_populations.append(bitonic)
    
    # Odd-even transposition (120 comparators) - as a diverse starting point
    oet = _make_odd_even_transposition()
    if _verify(oet):
        initial_populations.append(oet)
    
    # Create several perturbed versions of the bitonic network
    for _ in range(5):
        perturbed = list(bitonic)
        # Remove 1-3 random comparators
        for _ in range(random.randint(1, 3)):
            if len(perturbed) > 0:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        if _verify(perturbed):
            initial_populations.append(perturbed)
    
    # If we still don't have enough, add more random perturbations
    while len(initial_populations) < 10:
        base = random.choice(initial_populations)
        candidate = list(base)
        for _ in range(random.randint(1, 5)):
            move_type = random.choice(['remove', 'replace', 'add'])
            if move_type == 'remove' and len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif move_type == 'replace' and len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
                candidate[idx] = random.choice(all_pairs)
            elif move_type == 'add' and len(candidate) < 80:
                all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
                candidate.append(random.choice(all_pairs))
        if _verify(candidate):
            initial_populations.append(candidate)
        else:
            initial_populations.append(list(base))
    
    # Phase 1: Run genetic algorithm with most of the time budget
    ga_time = deadline - time.time() - 10
    if ga_time > 5:
        best_network = _genetic_algorithm(initial_populations, ga_time, pop_size=15)
    else:
        best_network = min(initial_populations, key=len)
    
    # Phase 2: Final local search on the best
    if time.time() < deadline - 2:
        best_network = _local_search(best_network, deadline - time.time() - 2)
    
    # Verify final result
    if not _verify(best_network):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return best_network
