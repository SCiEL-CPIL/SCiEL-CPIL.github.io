"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 60  # seconds, with safety margin

# EVOLVE-BLOCK-START

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    # Virtual indices 0..k_half-1 map to 0, 2, ..., k-2
    # Virtual indices k_half..k-1 map to k, k+2, ..., 2k-2
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    # Virtual indices 0..k_half-1 map to 1, 3, ..., k-1
    # Virtual indices k_half..k-1 map to k+1, k+3, ..., 2k-1
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    
    # Merge the two sorted halves
    merge = _merge(k)
    
    return left + right + merge

def _verify_network(comparators, n=N, time_limit=None, start_time=None):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle."""
    if start_time is None:
        start_time = time.time()
    if time_limit is not None and time.time() - start_time > time_limit:
        return None  # Indicate timeout/uncertain
        
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if wires != sorted(wires):
            return False
    return True

def _fitness(comparators, n=N, time_limit=None, start_time=None):
    """
    Calculate fitness: number of unsorted boolean inputs.
    Lower is better. 0 means valid sorting network.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is not None and time.time() - start_time > time_limit:
        return None
        
    unsorted_count = 0
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if wires != sorted(wires):
            unsorted_count += 1
            # Early exit if we've already found many unsorted cases
            # and time is running out? No, just count.
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
    return unsorted_count

def _remove_redundant(comparators, n=N, time_limit=50):
    """
    Greedy removal of redundant comparators.
    """
    start_time = time.time()
    current = list(comparators)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        removed = False
        for idx in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            
            candidate = current[:idx] + current[idx+1:]
            # Quick check with a subset first?
            # Full verification is expensive (65k * len(comps))
            # But len is small (~60). 65k * 60 = 3.9M ops. Fast enough in C/NumPy?
            # Python loops are slow. Let's optimize verification.
            
            if _verify_network(candidate, n, time_limit - elapsed, start_time):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    return current

def _genetic_search(initial_network, n=N, time_limit=30):
    """
    Genetic algorithm to find a smaller network.
    Population of networks.
    Fitness: -len(network) if valid, else -len + large_penalty * unsorted_count.
    Actually, we want to minimize length subject to validity.
    Let's use: fitness = len(network) + 1000 * unsorted_count.
    """
    start_time = time.time()
    
    # Initialize population
    # Start with the initial network and some random mutations
    pop_size = 50
    population = []
    
    # Add initial
    population.append(initial_network[:])
    
    # Add random networks (random comparators)
    # Or mutations of initial
    for _ in range(pop_size - 1):
        net = initial_network[:]
        # Random mutations: swap, delete, insert
        for _ in range(random.randint(0, 5)):
            op = random.choice(['swap', 'delete', 'insert'])
            if op == 'swap' and len(net) > 1:
                i, j = random.sample(range(len(net)), 2)
                net[i], net[j] = net[j], net[i]
            elif op == 'delete' and len(net) > 1:
                i = random.randint(0, len(net) - 1)
                net.pop(i)
            elif op == 'insert':
                i = random.randint(0, n - 1)
                j = random.randint(i + 1, n - 1)
                pos = random.randint(0, len(net))
                net.insert(pos, (i, j))
        population.append(net)
    
    best_network = initial_network[:]
    best_fitness = len(initial_network) + 1000 * _fitness(initial_network, n, time_limit, start_time)
    
    # If initial is valid, fitness is just len
    if _verify_network(initial_network, n, time_limit, start_time):
        best_fitness = len(initial_network)
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        if time.time() - start_time > time_limit:
            break
            
        # Evaluate fitness for population
        # We need to be careful with time.
        # Full fitness evaluation is expensive.
        # Let's only evaluate a subset or use a faster check.
        
        # Sort population by fitness (approximate)
        # To save time, we can just pick the best few and mutate them.
        
        # Selection: Keep the best half
        # We need fitness scores.
        # Let's compute fitness for all, but with a strict time check.
        
        scored_pop = []
        for net in population:
            # Check if valid first (fast if we cache? No, stateless)
            # If valid, fitness = len(net)
            # If not, fitness = len(net) + 1000 * unsorted_count
            # But calculating unsorted_count is same cost as verification.
            
            # Optimization: Check validity. If valid, fitness is known.
            # If not, we don't strictly need the count, just a penalty.
            # But count helps guide.
            
            # Let's just check validity.
            # If valid, fitness = len(net)
            # If invalid, fitness = 10000 (large penalty)
            # This might be too coarse.
            
            # Better: Use a sample of inputs to estimate unsorted count.
            # Sample 1000 random masks.
            
            sample_unsorted = 0
            sample_size = 1000
            for _ in range(sample_size):
                mask = random.getrandbits(n)
                wires = [(mask >> i) & 1 for i in range(n)]
                for i, j in net:
                    if wires[i] > wires[j]:
                        wires[i], wires[j] = wires[j], wires[i]
                if wires != sorted(wires):
                    sample_unsorted += 1
            
            # Estimate total unsorted
            # If sample is 0, likely valid. Verify fully.
            if sample_unsorted == 0:
                if _verify_network(net, n, time_limit - (time.time() - start_time), start_time):
                    fitness = len(net)
                else:
                    # Rare case: sample missed the unsorted ones
                    fitness = len(net) + 10000
            else:
                # Scale up
                estimated_unsorted = (sample_unsorted / sample_size) * (1 << n)
                fitness = len(net) + estimated_unsorted
            
            scored_pop.append((fitness, net))
            
            if time.time() - start_time > time_limit:
                break
        
        if not scored_pop:
            break
            
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_fitness:
            best_fitness = scored_pop[0][0]
            best_network = scored_pop[0][1][:]
            
            # If valid, we can try to remove redundants
            if best_fitness == len(best_network):
                 # Valid network found
                 pass
                 
        # Create next generation
        new_population = []
        
        # Elitism: Keep top 5
        for i in range(min(5, len(scored_pop))):
            new_population.append(scored_pop[i][1][:])
            
        # Crossover and Mutation
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
                
            # Pick parents from top 20
            parent1 = scored_pop[random.randint(0, min(19, len(scored_pop)-1))][1]
            parent2 = scored_pop[random.randint(0, min(19, len(scored_pop)-1))][1]
            
            # Crossover: Cut and splice
            # Since networks are sequences of comparators, simple cut might break structure.
            # But comparators are commutative in effect if they are on disjoint wires? No.
            # Order matters.
            # Let's just take a random prefix of P1 and suffix of P2.
            
            cut1 = random.randint(0, len(parent1))
            cut2 = random.randint(0, len(parent2))
            
            child = parent1[:cut1] + parent2[cut2:]
            
            # Mutation
            for _ in range(random.randint(1, 3)):
                op = random.choice(['swap', 'delete', 'insert'])
                if op == 'swap' and len(child) > 1:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                elif op == 'delete' and len(child) > 1:
                    i = random.randint(0, len(child) - 1)
                    child.pop(i)
                elif op == 'insert':
                    i = random.randint(0, n - 1)
                    j = random.randint(i + 1, n - 1)
                    pos = random.randint(0, len(child))
                    child.insert(pos, (i, j))
                    
            new_population.append(child)
            
        population = new_population

    return best_network

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    Each comparator is a tuple (i, j) with 0 <= i < j < N.
    """
    start_time = time.time()
    
    # 1. Generate Batcher's network
    network = _batcher(N)
    
    # 2. Try to remove redundant comparators (Greedy)
    # This is the most reliable way to reduce from 63 to ~60
    network = _remove_redundant(network, N, time_limit=20)
    
    # 3. If time remains, try Genetic Search to find even smaller or different topologies
    elapsed = time.time() - start_time
    if elapsed < 40: # 20s left for GA
        # GA might find a better one, but it's risky.
        # The greedy removal is usually sufficient to get to 60-62.
        # The optimal is 60.
        # Let's try GA with the current best as seed.
        ga_network = _genetic_search(network, N, time_limit=40 - elapsed)
        
        # Compare
        # Verify GA result
        if _verify_network(ga_network, N, time_limit=5, start_time=start_time):
            if len(ga_network) < len(network):
                network = ga_network
        else:
            # GA result invalid, stick with greedy
            pass
            
    # Final check
    if not _verify_network(network, N, time_limit=5, start_time=start_time):
        # Fallback to Batcher if something went wrong
        network = _batcher(N)
        
    return network

# EVOLVE-BLOCK-END
