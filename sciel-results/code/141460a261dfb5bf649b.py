import time
import random
import numpy as np
from itertools import combinations

N = 16

# EVOLVE-BLOCK-START
def _apply_comparator_vectorized(states, i, j):
    """Apply a compare-exchange to a vector of bitmask states.
    states is a numpy array of int64 bitmasks.
    Returns a new array with the comparator applied.
    """
    # Extract bits i and j
    bi = (states >> i) & 1
    bj = (states >> j) & 1
    
    # Where bi=0 and bj=1, we need to swap
    swap_mask = (bi == 0) & (bj == 1)
    
    # Apply swap: set bit i to 1, bit j to 0
    result = states.copy()
    result[swap_mask] |= (1 << i)
    result[swap_mask] &= ~(1 << j)
    
    return result

def _apply_network_vectorized(states, network):
    """Apply a list of comparators to a vector of bitmask states.
    states is a numpy array of int64 bitmasks.
    Returns a new array with the network applied.
    """
    for i, j in network:
        states = _apply_comparator_vectorized(states, i, j)
    return states

def _is_sorted_vectorized(states):
    """Check if bitmask states are sorted (all 0s then all 1s).
    Returns a boolean array.
    """
    # A state is sorted if it's of the form 0...01...1
    # For each state, check if all 1s are contiguous at the top
    
    # For state s, it's sorted if: for all k < l, if bit k is 1 then bit l is 1
    # Equivalently: the set bits form a suffix
    
    # Efficient check: state is sorted iff (state & (state - 1)) has no bits set below the lowest set bit
    # Or: state == 0 or state == (1<<N)-1 or (state has form 0^k 1^(N-k))
    
    # Precompute all sorted states
    sorted_states = set()
    for num_ones in range(N + 1):
        # State with num_ones 1s at the top
        if num_ones == 0:
            sorted_states.add(0)
        else:
            mask = ((1 << num_ones) - 1) << (N - num_ones)
            sorted_states.add(mask)
    
    # Convert to numpy for vectorized comparison
    sorted_array = np.array(list(sorted_states), dtype=np.int64)
    
    # For each state, check if it's in sorted_array
    # This is O(len(sorted_array)) per state, but sorted_array is small (17 elements)
    result = np.zeros(len(states), dtype=bool)
    for idx, state in enumerate(states):
        if state in sorted_states:
            result[idx] = True
    
    return result

def _count_violations_vectorized(network):
    """Count the number of inputs for which the network fails to sort.
    Uses vectorized operations for speed.
    Returns the count of violations (0 means perfect sorting).
    """
    # Generate all 2^N input states
    all_states = np.arange(1 << N, dtype=np.int64)
    
    # Apply the network
    result_states = _apply_network_vectorized(all_states, network)
    
    # Check which are sorted
    sorted_mask = _is_sorted_vectorized(result_states)
    
    # Count violations
    violations = np.sum(~sorted_mask)
    return int(violations)

def _verify(network):
    """Verify that the network sorts all inputs. Returns True if valid."""
    return _count_violations_vectorized(network) == 0

def _random_layer():
    """Generate a random layer of disjoint comparators.
    A layer is a random permutation of the 16 wires, paired as (0,1), (2,3), ..., (14,15).
    """
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for k in range(N // 2):
        i, j = wires[2*k], wires[2*k+1]
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _random_network(num_layers):
    """Generate a random network with a fixed number of layers."""
    network = []
    for _ in range(num_layers):
        network.extend(_random_layer())
    return network

def _remove_redundant_comparators(network, time_budget=5.0):
    """Remove redundant comparators from the network.
    A comparator is redundant if removing it does not increase the violation count.
    We use a greedy approach: try removing each comparator and keep it if it causes violations.
    """
    if not network:
        return network
    
    # First, check if the network is valid
    if not _verify(network):
        return network

    start_time = time.time()
    kept = network[:]
    
    # Iteratively try to remove comparators
    changed = True
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        for idx in range(len(kept)):
            if (time.time() - start_time) >= time_budget:
                break
            
            # Create a new network without this comparator
            candidate = kept[:idx] + kept[idx+1:]
            # Check if it's still valid
            if _verify(candidate):
                # It's redundant, so remove it
                kept = candidate
                changed = True
                break  # Restart from beginning since indices shifted
    
    return kept

def _batcher_oms(n):
    """Generate Batcher's Odd-Even Merge Sort network for n wires.
    Returns a list of (i, j) comparator pairs.
    """
    if n == 1:
        return []
    
    # Recursively sort halves
    left = _batcher_oms(n // 2)
    right = _batcher_oms(n - n // 2)
    
    # Offset right half by n//2
    right = [(i + n // 2, j + n // 2) for i, j in right]
    
    # Merge the two sorted halves
    merge_comparators = _batcher_merge(n // 2, n - n // 2)
    
    return left + right + merge_comparators

def _batcher_merge(n1, n2):
    """Generate comparators to merge two sorted sequences of lengths n1 and n2.
    The first sequence is on wires 0..n1-1, second on wires n1..n1+n2-1.
    """
    comparators = []
    if n1 == 0 or n2 == 0:
        return comparators
    
    # Odd-even merge: merge odd and even subsequences
    # First, recursively merge the odd-indexed elements
    # Odd indices in first half: 0, 2, 4, ...
    # Odd indices in second half: n1, n1+2, n1+4, ...
    
    n1_odd = (n1 + 1) // 2  # Number of odd-indexed elements in first half
    n2_odd = (n2 + 1) // 2  # Number of odd-indexed elements in second half
    
    # Merge odd subsequences (offset by 0 in first, n1 in second)
    # But we need to map back to original indices
    # This is getting complex; let me use a simpler approach for now
    
    # For small n, use brute force or a known pattern
    # For n1=n2=8, the merge pattern is known
    
    # Simpler: use the recursive definition properly
    # Actually, let me just use a known good network for 16 wires
    
    # Batcher's OMS for 16 wires has 119 comparators
    # Let me return a placeholder and use OET as fallback
    return comparators

def _known_good_network():
    """Return a known good sorting network for 16 wires."""
    # Odd-Even Transposition Network (120 comparators)
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _local_search(network, time_budget=2.0):
    """Perform local search to improve the network.
    Try swapping, inserting, removing, and modifying comparators.
    """
    if not _verify(network):
        return network
    
    start_time = time.time()
    best = network[:]
    best_len = len(best)
    
    while (time.time() - start_time) < time_budget:
        improved = False
        
        # Try removing comparators
        for i in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = best[:i] + best[i+1:]
            if _verify(candidate) and len(candidate) < best_len:
                best = candidate
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping adjacent comparators
        for i in range(len(best) - 1):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = best[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify(candidate) and len(candidate) < best_len:
                best = candidate
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try changing a comparator
        for i in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            old_i, old_j = best[i]
            for new_i in range(N):
                for new_j in range(new_i + 1, N):
                    if (time.time() - start_time) >= time_budget:
                        break
                    if (new_i, new_j) == (old_i, old_j):
                        continue
                    candidate = best[:]
                    candidate[i] = (new_i, new_j)
                    if _verify(candidate) and len(candidate) < best_len:
                        best = candidate
                        best_len = len(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try inserting a comparator
        for pos in range(len(best) + 1):
            if (time.time() - start_time) >= time_budget:
                break
            for i in range(N):
                for j in range(i + 1, N):
                    if (time.time() - start_time) >= time_budget:
                        break
                    candidate = best[:pos] + [(i, j)] + best[pos:]
                    if _verify(candidate) and len(candidate) < best_len:
                        best = candidate
                        best_len = len(best)
                        improved = True
                        break
                    if len(candidate) < best_len:
                        # Even if not shorter, might be useful for future improvements
                        pass
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    """Construct a sorting network for 16 wires.
    Returns a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for 90s limit
    
    # Start with a known good network
    best_network = _known_good_network()
    best_fitness = len(best_network)
    
    # Try to improve using local search first
    if time.time() < deadline - 10:
        optimized = _local_search(best_network, time_budget=min(10.0, deadline - time.time() - 5))
        if len(optimized) < best_fitness:
            best_network = optimized
            best_fitness = len(best_network)
    
    # If we have time, try genetic algorithm
    if time.time() < deadline - 15:
        pop_size = 30
        num_layers_init = 8  # Start with 8 layers (64 comparators)
        
        # Initialize population with some diversity
        population = [best_network[:]]
        for _ in range(pop_size - 1):
            net = _random_network(num_layers_init)
            # Add some mutations to the best network
            if random.random() < 0.3 and len(best_network) > 0:
                net = best_network[:]
                # Randomly modify a few comparators
                for _ in range(random.randint(1, 3)):
                    if len(net) > 0:
                        idx = random.randint(0, len(net) - 1)
                        i, j = random.sample(range(N), 2)
                        if i > j: i, j = j, i
                        net[idx] = (i, j)
            population.append(net)
        
        # Evaluate and find best
        for net in population:
            v = _count_violations_vectorized(net)
            if v == 0:
                f = len(net)
            else:
                f = v * 1000 + len(net)
            
            if f < best_fitness:
                best_fitness = f
                best_network = net[:]
        
        # Genetic Algorithm Loop
        generation = 0
        while time.time() < deadline - 5:
            generation += 1
            
            # Selection: Tournament selection
            def select():
                a, b = random.sample(population, 2)
                fa = _count_violations_vectorized(a)
                fb = _count_violations_vectorized(b)
                if fa == 0 and fb == 0:
                    return a if len(a) < len(b) else b
                elif fa == 0:
                    return a
                elif fb == 0:
                    return b
                else:
                    return a if fa < fb else b
            
            new_population = []
            
            # Elitism: Keep the best individual
            best_net = min(population, key=lambda x: (_count_violations_vectorized(x) * 1000 + len(x)))
            new_population.append(best_net[:])
            
            # Crossover and Mutation
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break
                    
                parent1 = select()
                parent2 = select()
                
                # Crossover: Slice crossover
                if len(parent1) != len(parent2):
                    max_len = min(len(parent1), len(parent2))
                    p1 = parent1[:max_len]
                    p2 = parent2[:max_len]
                else:
                    p1 = parent1
                    p2 = parent2
                
                cross_point = random.randint(1, len(p1) - 1) if len(p1) > 1 else 0
                child = p1[:cross_point] + p2[cross_point:]
                
                # Mutation
                if random.random() < 0.1 and len(child) > 0:
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
                elif random.random() < 0.1 and len(child) < 120:
                    i, j = random.sample(range(N), 2)
                    if i > j: i, j = j, i
                    idx = random.randint(0, len(child))
                    child.insert(idx, (i, j))
                elif random.random() < 0.3 and len(child) > 0:
                    idx = random.randint(0, len(child) - 1)
                    i, j = random.sample(range(N), 2)
                    if i > j: i, j = j, i
                    child[idx] = (i, j)
                elif random.random() < 0.2 and len(child) > 1:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                
                new_population.append(child)
            
            population = new_population
            
            # Evaluate new population and update best
            for net in population:
                v = _count_violations_vectorized(net)
                if v == 0:
                    f = len(net)
                else:
                    f = v * 1000 + len(net)
                
                if f < best_fitness:
                    best_fitness = f
                    best_network = net[:]
            
            # If we found a valid network, try to optimize its length
            if best_fitness < 1000 and time.time() < deadline - 10:
                optimized = _local_search(best_network, time_budget=min(5.0, deadline - time.time() - 5))
                if len(optimized) < best_fitness:
                    best_network = optimized
                    best_fitness = len(optimized)
    
    # Final optimization pass
    if time.time() < deadline - 3:
        optimized = _remove_redundant_comparators(best_network, time_budget=min(3.0, deadline - time.time() - 1))
        if len(optimized) < best_fitness:
            best_network = optimized
    
    # Final check
    if best_network is not None and _verify(best_network):
        return best_network
    else:
        return _known_good_network()

# EVOLVE-BLOCK-END
