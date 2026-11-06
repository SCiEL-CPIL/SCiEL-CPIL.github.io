"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 18

# EVOLVE-BLOCK-START
def _apply_network(network, inputs):
    """Apply a compare-exchange network to a list of inputs.
    Returns the sorted list if the network is correct for this input."""
    wires = list(inputs)
    for i, j in network:
        if wires[i] > wires[j]:
            wires[i], wires[j] = wires[j], wires[i]
    return wires

def _is_correct_fast(network, max_checks=None):
    """Check if the network sorts all 2^N boolean inputs using fast bit manipulation."""
    if max_checks is None:
        max_checks = 1 << N
    
    for mask in range(max_checks):
        # Extract bits for each wire
        wires = [0] * N
        for i in range(N):
            wires[i] = (mask >> i) & 1
        
        # Apply network
        for i, j in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        
        # Check sorted: all 0s before all 1s
        # Count number of 1s
        num_ones = sum(wires)
        # Check if first (N - num_ones) are 0 and last num_ones are 1
        for i in range(N - num_ones):
            if wires[i] != 0:
                return False
        for i in range(N - num_ones, N):
            if wires[i] != 1:
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
    import math
    
    def bitonic_merge(start, size, direction):
        """Generate comparators for bitonic merge of 'size' elements starting at 'start'.
        direction: True for ascending, False for descending."""
        if size <= 1:
            return []
        
        comps = []
        half = size // 2
        
        # Recursively merge the two halves
        comps.extend(bitonic_merge(start, half, direction))
        comps.extend(bitonic_merge(start + half, half, not direction))
        
        # Cross-comparisons
        for i in range(half):
            if direction:
                comps.append((start + i, start + half + i))
            else:
                comps.append((start + half + i, start + i))
        
        return comps
    
    def bitonic_sort_network(start, size, direction):
        """Generate comparators for bitonic sort of 'size' elements starting at 'start'.
        direction: True for ascending, False for descending."""
        if size <= 1:
            return []
        
        if size == 2:
            if direction:
                return [(start, start + 1)]
            else:
                return [(start + 1, start)]
        
        half = size // 2
        
        # Sort first half ascending, second half descending (to form bitonic sequence)
        comps = []
        comps.extend(bitonic_sort_network(start, half, True))
        comps.extend(bitonic_sort_network(start + half, half, False))
        
        # Merge the bitonic sequence
        comps.extend(bitonic_merge(start, size, direction))
        
        return comps
    
    # For n=18, we need to handle non-power-of-2
    # Use the next power of 2 (32) and then trim
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    
    # Generate for next_pow2 wires
    full_network = bitonic_sort_network(0, next_pow2, True)
    
    # Filter to only include comparators with indices < n
    filtered = [(i, j) for i, j in full_network if i < n and j < n]
    
    return filtered

def _odd_even_transposition(n):
    """Generate odd-even transposition sort network for n wires."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _remove_redundant(network, max_time=2.0):
    """Try to remove redundant comparators from a valid network."""
    start_time = time.time()
    network = list(network)
    
    while time.time() - start_time < max_time:
        removed = False
        for idx in range(len(network)):
            # Try removing this comparator
            candidate = network[:idx] + network[idx+1:]
            if _is_correct_fast(candidate, max_checks=4096):
                # Do a full check to be safe
                if _is_correct_fast(candidate):
                    network = candidate
                    removed = True
                    break
        
        if not removed:
            break
    
    return network

def _crossover(parent1, parent2):
    """Crossover two networks: take a prefix from one and suffix from the other."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    cut = random.randint(0, min(len(parent1), len(parent2)))
    child = list(parent1[:cut]) + list(parent2[cut:])
    return child

def _mutation(network, rate=0.15):
    """Mutate a network by adding, removing, or swapping comparators."""
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
    
    # Swap two adjacent comparators (more likely to preserve validity)
    if random.random() < rate * 0.5 and len(network) >= 2:
        i = random.randint(0, len(network) - 2)
        network[i], network[i+1] = network[i+1], network[i]
    
    return network

def construct():
    """Construct a sorting network using a genetic algorithm with local search."""
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for safety
    
    # Initialize population with known good networks
    population = []
    
    # 1. Odd-even transposition (seed)
    seed = _odd_even_transposition(N)
    population.append(seed)
    
    # 2. Batcher's bitonic sort (smaller than seed)
    bitonic = _batcher_bitonic_sort(N)
    population.append(bitonic)
    
    # 3. Add some variants
    for _ in range(18):
        # Start from a random base
        base = random.choice([seed, bitonic])
        net = list(base)
        
        # Add some random comparators
        for _ in range(random.randint(0, 10)):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            net.append((i, j))
        
        population.append(net)
    
    # Find the best initial network
    best_network = None
    best_size = float('inf')
    
    for net in population:
        if _is_correct_fast(net):
            if len(net) < best_size:
                best_size = len(net)
                best_network = net
    
    if best_network is None:
        best_network = seed
        best_size = len(seed)
    
    # Local search on the best network
    best_network = _remove_redundant(best_network, max_time=3.0)
    best_size = len(best_network)
    
    pop_size = 20
    
    # Main GA loop
    generation = 0
    while time.time() - start_time < timeout:
        generation += 1
        
        # Evaluate fitness - only do full check for promising candidates
        scored = []
        for net in population:
            net_len = len(net)
            if net_len < best_size:
                # Do a quick check first
                if _is_correct_fast(net, max_checks=2048):
                    # Do full check
                    if _is_correct_fast(net):
                        scored.append((net_len, net))
                        best_size = net_len
                        best_network = net
                    else:
                        scored.append((net_len + 500, net))
                else:
                    scored.append((net_len + 500, net))
            else:
                scored.append((net_len, net))
        
        # Sort by fitness
        scored.sort(key=lambda x: x[0])
        population = [net for _, net in scored]
        
        # Create next generation
        next_gen = []
        
        # Elitism: keep the best few
        for net in population[:5]:
            next_gen.append(list(net))
        
        # Crossover and mutation
        while len(next_gen) < pop_size:
            # Select parents with bias toward better networks
            p1 = random.choice(population[:10])
            p2 = random.choice(population[:10])
            child = _crossover(p1, p2)
            child = _mutation(child, rate=0.15)
            next_gen.append(child)
        
        population = next_gen
        
        # Local search on the current best every few generations
        if generation % 3 == 0 and time.time() - start_time < timeout - 5:
            improved = _remove_redundant(best_network, max_time=2.0)
            if len(improved) < best_size:
                best_network = improved
                best_size = len(best_network)
        
        # Early exit if we found a very good network
        if best_size <= 72:
            break
    
    # Final local search
    if time.time() - start_time < timeout - 2:
        best_network = _remove_redundant(best_network, max_time=3.0)
    
    return best_network
# EVOLVE-BLOCK-END
