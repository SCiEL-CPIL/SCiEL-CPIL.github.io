"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

N = 18

# EVOLVE-BLOCK-START
def _apply_network_to_mask(network, mask):
    """Apply network to a boolean mask, return resulting mask.
    Uses bit operations for speed."""
    wires = mask  # Represent as integer where bit i is wire i
    for i, j in network:
        # Compare wires i and j
        vi = (wires >> i) & 1
        vj = (wires >> j) & 1
        if vi > vj:
            # Swap: set i to vj (0), j to vi (1)
            wires &= ~(1 << i)  # clear bit i
            wires |= (1 << j)    # set bit j
        # else: i=0, j=1 already correct
    return wires

def _is_correct_fast(network, sample_size=262144):
    """Check if network sorts all boolean inputs using bit-parallel approach.
    Returns True if correct for all 2^N inputs."""
    total = 1 << N
    # We can check all 2^18 = 262144 inputs
    for mask in range(total):
        result = _apply_network_to_mask(network, mask)
        # Check if result is sorted: all 1s should be in lower positions? 
        # Wait: compare-exchange puts min on i, max on j.
        # For boolean: if wire[i]=1 and wire[j]=0, swap so wire[i]=0, wire[j]=1.
        # So after sorting, all 0s should come before all 1s.
        # Sorted means: there exists k such that bits 0..k-1 are 0, bits k..N-1 are 1.
        # Equivalently: the result should be of the form 0...01...1
        # Check: find first 1, then all bits after must be 1
        # Or: count of 1s = c, then result should have bits N-c..N-1 set to 1, rest 0.
        c = bin(result).count('1')
        if c == 0:
            expected = 0
        elif c == N:
            expected = (1 << N) - 1
        else:
            expected = ((1 << c) - 1) << (N - c)
        if result != expected:
            return False
    return True

def _is_correct_sampled(network, samples=5000):
    """Faster check using random sampling for quick fitness evaluation."""
    for _ in range(samples):
        mask = random.randint(0, (1 << N) - 1)
        result = _apply_network_to_mask(network, mask)
        c = bin(result).count('1')
        if c == 0:
            expected = 0
        elif c == N:
            expected = (1 << N) - 1
        else:
            expected = ((1 << c) - 1) << (N - c)
        if result != expected:
            return False
    return True

def _depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    wire_last_step = [0] * N
    max_depth = 0
    for i, j in network:
        d = max(wire_last_step[i], wire_last_step[j]) + 1
        max_depth = max(max_depth, d)
        wire_last_step[i] = d
        wire_last_step[j] = d
    return max_depth

def _batcher_bitonic_sort(n):
    """Generate Batcher's bitonic sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Bitonic sort:
    # 1. Recursively sort first half ascending, second half descending
    # 2. Bitonic merge
    half = n // 2
    comps = []
    
    # Sort first half (ascending)
    comps.extend(_batcher_bitonic_sort(half))
    # Sort second half (descending) - we can sort ascending then reverse comparator order
    # Actually for bitonic sort, we need one half ascending, one descending
    # Simple approach: sort both ascending, then the second half is "descending" in the sense
    # that we'll handle it in the merge. Actually the standard way:
    # - Sort first half ascending
    # - Sort second half descending (by sorting ascending and swapping all comparators' direction,
    #   but since our comparators are directional (min to i, max to j), we need to be careful)
    
    # Let me use a different approach: build bitonic sort properly
    # For n=2^k, bitonic sort has log2(n) stages
    # Stage s (s=1 to log2(n)): 
    #   For each block of size 2^s, do a bitonic merge
    
    # Actually, let's implement it recursively properly:
    # bitonic_sort(n):
    #   if n == 1: return []
    #   first_half = bitonic_sort(n/2)  # ascending
    #   second_half = bitonic_sort_desc(n/2)  # descending
    #   merge = bitonic_merge(n)
    #   return first_half + second_half + merge
    #
    # But our comparators are directional. Let's think differently.
    # 
    # Standard bitonic sort network for n=2^k:
    # For k from 1 to log2(n):
    #   For each block of size 2^k:
    #     Do a bitonic merge of size 2^k
    #
    # Bitonic merge of size m (m power of 2):
    #   For s from log2(m) down to 1:
    #     For i from 0 to m - 2^s - 1:
    #       Compare (i, i + 2^s)
    
    # Let's implement bitonic merge first
    def _bitonic_merge(m, offset=0):
        """Bitonic merge for m elements starting at offset."""
        if m <= 1:
            return []
        comps = []
        logm = int(math.log2(m))
        for s in range(logm, 0, -1):
            step = 1 << s
            for i in range(0, m - step):
                comps.append((offset + i, offset + i + step))
        return comps
    
    # Bitonic sort:
    # Stage 1: sort blocks of size 2 (ascending)
    # Stage 2: sort blocks of size 4 (bitonic merge)
    # ...
    # Stage log2(n): sort all n elements
    
    comps = []
    for k in range(1, int(math.log2(n)) + 1):
        block_size = 1 << k
        for offset in range(0, n, block_size):
            # Within each block, do a bitonic merge
            # But first, we need the block to be bitonic
            # The standard bitonic sort works by:
            # - At stage k, each block of size 2^k is bitonic
            # - We merge adjacent blocks of size 2^(k-1) to form bitonic blocks of size 2^k
            
            # Actually, the simpler way:
            # For each block of size 2^k, apply bitonic merge
            # But we need to ensure the input is bitonic
            
            # Let's use the standard recursive definition instead:
            pass
    
    # Simpler: just use the known structure
    # For n=18, we can pad to 32 or use a different approach
    # Let's just return a valid network using odd-even transposition as fallback
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps

def _random_network_near_base(base, num_add=5, num_remove=3):
    """Generate a random network by modifying a base network."""
    network = list(base)
    # Add random comparators
    for _ in range(num_add):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(network))
        network.insert(pos, (i, j))
    # Remove random comparators
    for _ in range(num_remove):
        if network:
            pos = random.randint(0, len(network) - 1)
            network.pop(pos)
    return network

def _crossover(parent1, parent2):
    """Crossover: take prefix from parent1, suffix from parent2."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    # Choose a crossover point
    min_len = min(len(parent1), len(parent2))
    if min_len == 0:
        return list(parent1) if parent1 else list(parent2)
    cut = random.randint(0, min_len)
    child = list(parent1[:cut]) + list(parent2[cut:])
    return child

def _mutation(network, rate=0.2):
    """Mutate network with add, remove, or swap operations."""
    network = list(network)
    # Add a random comparator
    if random.random() < rate:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(network))
        network.insert(pos, (i, j))
    # Remove a random comparator
    if random.random() < rate and network:
        pos = random.randint(0, len(network) - 1)
        network.pop(pos)
    # Swap two adjacent comparators
    if random.random() < rate and len(network) >= 2:
        i = random.randint(0, len(network) - 2)
        network[i], network[i + 1] = network[i + 1], network[i]
    # Duplicate a random comparator
    if random.random() < rate * 0.5 and network:
        i = random.randint(0, len(network) - 1)
        pos = random.randint(0, len(network))
        network.insert(pos, network[i])
    return network

def construct():
    """Construct a sorting network using an improved genetic algorithm."""
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for safety
    
    # Initialize population with diverse valid networks
    seed = _odd_even_transposition(N)
    
    # Generate several base networks
    bases = [seed]
    # Add some variations
    for _ in range(5):
        bases.append(_random_network_near_base(seed, num_add=random.randint(0, 10), 
                                                 num_remove=random.randint(0, 5)))
    
    # Initialize population
    population = []
    for base in bases:
        population.append(list(base))
        # Add mutated versions
        for _ in range(2):
            population.append(_mutation(list(base), rate=0.3))
    
    # Ensure population size
    while len(population) < 30:
        base = random.choice(bases)
        population.append(_random_network_near_base(base, 
                                                     num_add=random.randint(0, 8),
                                                     num_remove=random.randint(0, 4)))
    
    best_network = seed
    best_size = len(seed)
    
    # Verify seed is correct using fast method
    if _is_correct_fast(seed):
        best_size = len(seed)
    else:
        # Shouldn't happen, but fallback
        best_size = len(seed) + 1000
    
    pop_size = 30
    generation = 0
    
    while time.time() - start_time < timeout:
        generation += 1
        
        # Evaluate fitness: use sampled check for speed, then verify best candidates
        scored = []
        for net in population:
            net_size = len(net)
            if net_size < best_size:
                # Quick sampled check
                if _is_correct_sampled(net, samples=2000):
                    # Full check for promising candidates
                    if _is_correct_fast(net):
                        scored.append((net_size, net))
                        best_size = net_size
                        best_network = net
                    else:
                        scored.append((net_size + 500, net))
                else:
                    scored.append((net_size + 500, net))
            else:
                scored.append((net_size, net))
        
        # Sort by fitness
        scored.sort(key=lambda x: x[0])
        population = [net for _, net in scored]
        
        # Create next generation
        next_gen = []
        # Elitism: keep the best 6
        elite_count = min(6, len(population))
        for net in population[:elite_count]:
            next_gen.append(list(net))
        
        # Crossover and mutation
        while len(next_gen) < pop_size:
            # Tournament selection
            p1 = random.choice(population[:15])
            p2 = random.choice(population[:15])
            child = _crossover(p1, p2)
            child = _mutation(child, rate=0.25)
            next_gen.append(child)
        
        population = next_gen
        
        # Early exit if we found a very good network
        if best_size <= 72:
            break
        
        # If time is running out, do a final verification
        if time.time() - start_time > timeout - 5:
            break
    
    # Final verification of best network
    if not _is_correct_fast(best_network):
        # Fallback to seed if best is invalid
        if _is_correct_fast(seed):
            return seed
        return best_network
    
    return best_network
# EVOLVE-BLOCK-END
