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
    """Apply a list of comparators to a vector of bitmask states."""
    for i, j in network:
        states = _apply_comparator_vectorized(states, i, j)
    return states

def _is_sorted_vectorized(states):
    """Check if bitmask states are sorted (all 0s then all 1s).
    Returns a boolean array.
    """
    sorted_states = set()
    for num_ones in range(N + 1):
        if num_ones == 0:
            sorted_states.add(0)
        else:
            mask = ((1 << num_ones) - 1) << (N - num_ones)
            sorted_states.add(mask)
    
    result = np.zeros(len(states), dtype=bool)
    for idx, state in enumerate(states):
        if state in sorted_states:
            result[idx] = True
    
    return result

def _count_violations_vectorized(network):
    """Count the number of inputs for which the network fails to sort."""
    all_states = np.arange(1 << N, dtype=np.int64)
    result_states = _apply_network_vectorized(all_states, network)
    sorted_mask = _is_sorted_vectorized(result_states)
    violations = np.sum(~sorted_mask)
    return int(violations)

def _verify(network):
    """Verify that the network sorts all inputs. Returns True if valid."""
    return _count_violations_vectorized(network) == 0

def _remove_redundant_comparators(network, time_budget=5.0):
    """Remove redundant comparators from the network."""
    if not network:
        return network
    
    if not _verify(network):
        return network

    start_time = time.time()
    kept = network[:]
    
    changed = True
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        for idx in range(len(kept)):
            if (time.time() - start_time) >= time_budget:
                break
            
            candidate = kept[:idx] + kept[idx+1:]
            if _verify(candidate):
                kept = candidate
                changed = True
                break
    
    return kept

def _known_good_network():
    """Return a known good sorting network for 16 wires (OET, 120 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _get_initial_networks():
    """Generate diverse initial networks including a known good one and random structures."""
    networks = []
    
    # 1. Known good OET
    networks.append(_known_good_network())
    
    # 2. Batcher's Odd-Even Merge Sort (approx 119 comparators)
    networks.append(_build_batcher_oms(16))
    
    # 3. Bitonic Sort (120 comparators)
    networks.append(_build_bitonic(16))
    
    # 4. Random Layer-based Networks (varying depths)
    random.seed(42)
    for depth in [8, 10, 12]:
        net = []
        for _ in range(depth):
            wires = list(range(N))
            random.shuffle(wires)
            for k in range(N // 2):
                i, j = wires[2*k], wires[2*k+1]
                if i > j: i, j = j, i
                net.append((i, j))
        networks.append(net)
        
    # 5. Hybrid: OET with random perturbations
    for _ in range(3):
        net = _known_good_network()[:]
        for _ in range(5):
            idx = random.randint(0, len(net)-1)
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            net[idx] = (i, j)
        networks.append(net)
        
    return networks

def _build_batcher_oms(n):
    """Batcher's Odd-Even Merge Sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    left = _build_batcher_oms(k)
    right = _build_batcher_oms(n - k)
    right = [(i + k, j + k) for i, j in right]
    
    comparators = left + right
    
    # Merge step
    comparators += _batcher_merge(n)
    
    return comparators

def _batcher_merge(n):
    """Merge step for Batcher's OMS on n wires (two sorted halves of n/2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    comparators = []
    
    # Step 1: Compare (i, i+k) for i in 0..k-1
    for i in range(k):
        comparators.append((i, i + k))
    
    # Step 2: Recursively merge even and odd subsequences
    # Even positions: 0, 2, 4, ...
    # Odd positions: 1, 3, 5, ...
    
    # We need to map the recursive merge indices back to the original wire indices.
    # The recursive merge operates on k elements.
    # Even subsequence: wires 0, 2, 4, ..., n-2
    # Odd subsequence: wires 1, 3, 5, ..., n-1
    
    even_wires = list(range(0, n, 2))
    odd_wires = list(range(1, n, 2))
    
    # Generate merge network for k wires
    sub_merge = _batcher_merge(k)
    
    for i, j in sub_merge:
        # Map to even wires
        comparators.append((even_wires[i], even_wires[j]))
        # Map to odd wires
        comparators.append((odd_wires[i], odd_wires[j]))
        
    return comparators

def _build_bitonic(n):
    """Bitonic Sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    left = _build_bitonic(k)
    right = _build_bitonic(k)
    right = [(i + k, j + k) for i, j in right]
    
    comparators = left + right
    
    # Bitonic merge
    comparators += _bitonic_merge(n)
    
    return comparators

def _bitonic_merge(n):
    """Bitonic merge step for n wires."""
    if n <= 1:
        return []
    
    comparators = []
    # Compare (i, i + n/2) for i in 0..n/2-1
    # Wait, standard bitonic merge is:
    # for k in range(n//2, 0, -1):
    #   for i in range(0, n-k):
    #     compare(i, i+k)
    
    for k in range(n // 2, 0, -1):
        for i in range(0, n - k):
            comparators.append((i, i + k))
            
    return comparators

def _local_search(network, time_budget=2.0):
    """Perform local search to improve the network."""
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
    
    # Generate initial diverse population
    initial_networks = _get_initial_networks()
    
    best_network = None
    best_fitness = float('inf')
    
    # Evaluate initial networks
    for net in initial_networks:
        v = _count_violations_vectorized(net)
        if v == 0:
            f = len(net)
        else:
            f = v * 1000 + len(net)
            
        if f < best_fitness:
            best_fitness = f
            best_network = net[:]
            
    # If no valid network found yet (shouldn't happen with OET), fallback
    if best_network is None:
        best_network = _known_good_network()
        best_fitness = len(best_network)
    
    # Local search on the best initial network
    if time.time() < deadline - 10:
        optimized = _local_search(best_network, time_budget=min(10.0, deadline - time.time() - 5))
        if len(optimized) < best_fitness:
            best_network = optimized
            best_fitness = len(best_network)
    
    # Genetic Algorithm
    if time.time() < deadline - 15:
        pop_size = 30
        
        # Initialize population
        population = [best_network[:]]
        for _ in range(pop_size - 1):
            # Mix of random and mutated best
            if random.random() < 0.3 and len(best_network) > 0:
                net = best_network[:]
                for _ in range(random.randint(1, 3)):
                    if len(net) > 0:
                        idx = random.randint(0, len(net) - 1)
                        i, j = random.sample(range(N), 2)
                        if i > j: i, j = j, i
                        net[idx] = (i, j)
            else:
                # Random layer-based
                num_layers = random.randint(8, 12)
                net = []
                for _ in range(num_layers):
                    wires = list(range(N))
                    random.shuffle(wires)
                    for k in range(N // 2):
                        i, j = wires[2*k], wires[2*k+1]
                        if i > j: i, j = j, i
                        net.append((i, j))
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
            
            # Elitism
            best_net = min(population, key=lambda x: (_count_violations_vectorized(x) * 1000 + len(x)))
            new_population.append(best_net[:])
            
            # Crossover and Mutation
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break
                    
                parent1 = select()
                parent2 = select()
                
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
