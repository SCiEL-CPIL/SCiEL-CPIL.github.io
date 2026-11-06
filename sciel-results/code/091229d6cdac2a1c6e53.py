"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

def _apply_network(net, inputs):
    """Apply sorting network to inputs (list of 0/1 or values)."""
    arr = list(inputs)
    for i, j in net:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    return arr

def _is_valid_network(net):
    """Check if network sorts all 2^N boolean inputs using 0-1 principle."""
    n = len(net)
    if n > 4096:
        return False
    # Check all 2^N boolean inputs
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(net, inputs)
        # Check if sorted (non-decreasing)
        for k in range(N - 1):
            if result[k] > result[k + 1]:
                return False
    return True

def _network_depth(net):
    """Compute parallel depth of network."""
    if not net:
        return 0
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

def _mutate(net, n_wires, mutation_rate=0.3):
    """Apply random mutations to network."""
    net = list(net)
    for _ in range(len(net)):
        if random.random() < mutation_rate:
            op = random.choice(['add', 'remove', 'swap', 'shift'])
            if op == 'add' and len(net) < 4096:
                i = random.randint(0, n_wires - 1)
                j = random.randint(i + 1, n_wires - 1)
                pos = random.randint(0, len(net))
                net.insert(pos, (i, j))
            elif op == 'remove' and len(net) > 1:
                pos = random.randint(0, len(net) - 1)
                net.pop(pos)
            elif op == 'swap' and len(net) > 1:
                i = random.randint(0, len(net) - 1)
                j = random.randint(0, len(net) - 1)
                net[i], net[j] = net[j], net[i]
            elif op == 'shift' and len(net) > 1:
                i = random.randint(0, len(net) - 1)
                j = random.randint(0, len(net) - 1)
                net[i], net[j] = net[j], net[i]
    return net

def _crossover(net1, net2, n_wires):
    """Crossover two networks."""
    if not net1 or not net2:
        return list(net1 or net2)
    cut1 = random.randint(0, len(net1))
    cut2 = random.randint(0, len(net2))
    child = net1[:cut1] + net2[cut2:]
    return child

def _seed_oe_transposition(n_wires):
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(n_wires)
            for i in range(phase % 2, n_wires - 1, 2)]

def _seed_batcher(n_wires):
    """Simple Batcher-like network (odd-even merge sort network)."""
    def oe_merge(a, b, k):
        """Odd-even merge of two sorted lists of length k (power of 2)."""
        if k == 1:
            return []
        half = k // 2
        comps = []
        # First, merge pairs
        for i in range(half):
            comps.append((a + i, b + i))
        # Recursively merge odd and even subsequences
        comps += oe_merge(a, b, half)  # This is simplified; actual Batcher is more complex
        return comps
    
    # Use a simpler approach: just return odd-even transposition for now
    return _seed_oe_transposition(n_wires)

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    random.seed(42)
    
    best_net = None
    best_score = float('inf')
    
    # Start with seed networks
    seeds = [
        _seed_oe_transposition(N),
    ]
    
    # Try some known small networks by construction
    # Bitonic sort network for 16 wires, then extend
    def bitonic_sort(n):
        """Generate bitonic sort network for n wires (power of 2)."""
        if n <= 1:
            return []
        half = n // 2
        comps = []
        # Recursively sort two halves
        comps += bitonic_sort(half)
        # Shift second half by half
        comps2 = [(i + half, j + half) for i, j in bitonic_sort(half)]
        comps += comps2
        # Merge the two sorted halves
        comps += bitonic_merge(0, n)
        return comps
    
    def bitonic_merge(start, length):
        """Merge two sorted sequences of length length/2 starting at start."""
        if length <= 1:
            return []
        half = length // 2
        comps = []
        for i in range(half):
            comps.append((start + i, start + i + half))
        # Recursively merge
        comps += bitonic_merge(start, half)
        comps += bitonic_merge(start + half, half)
        return comps
    
    # Bitonic sort for 16 wires
    bitonic_16 = bitonic_sort(16)
    # Extend to 18 wires by adding comparators for wires 16, 17
    # Simple approach: add comparators to integrate wires 16, 17
    extended_bitonic = list(bitonic_16)
    # Add comparators to sort the last 2-3 wires with the rest
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            if (i, i+1) not in extended_bitonic:
                extended_bitonic.append((i, i+1))
    
    seeds.append(extended_bitonic)
    
    # Initialize population
    population = []
    for seed in seeds:
        if _is_valid_network(seed):
            population.append(seed)
            score = len(seed)
            if score < best_score:
                best_score = score
                best_net = list(seed)
    
    if not population:
        # Fallback: return odd-even transposition
        return _seed_oe_transposition(N)
    
    # Genetic algorithm
    pop_size = min(20, max(5, len(population)))
    while len(population) < pop_size:
        parent = random.choice(population)
        child = _mutate(parent, N, mutation_rate=0.5)
        population.append(child)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate new population members
        new_population = []
        for net in population:
            if _is_valid_network(net):
                new_population.append(net)
                score = len(net)
                if score < best_score:
                    best_score = score
                    best_net = list(net)
        
        if not new_population:
            break
            
        population = new_population
        
        # If we've found a very good network, try to refine it
        if best_score <= 80:
            # Focus on local search around best
            new_population = []
            for _ in range(pop_size):
                if random.random() < 0.7:
                    parent = best_net
                else:
                    parent = random.choice(population)
                child = _mutate(parent, N, mutation_rate=0.2)
                new_population.append(child)
            population = new_population
        else:
            # Standard GA step
            new_population = []
            for _ in range(pop_size):
                op = random.random()
                if op < 0.6:
                    # Crossover
                    p1 = random.choice(population)
                    p2 = random.choice(population)
                    child = _crossover(p1, p2, N)
                    child = _mutate(child, N, mutation_rate=0.1)
                elif op < 0.8:
                    # Mutation
                    parent = random.choice(population)
                    child = _mutate(parent, N, mutation_rate=0.3)
                else:
                    # Copy best
                    child = list(best_net)
                new_population.append(child)
            population = new_population
    
    # Final: try local search on best network to remove redundant comparators
    if best_net is not None:
        improved = list(best_net)
        # Try removing each comparator
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for idx in range(len(improved) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                test_net = improved[:idx] + improved[idx+1:]
                if _is_valid_network(test_net):
                    improved = test_net
                    changed = True
                    if len(improved) < best_score:
                        best_score = len(improved)
                        best_net = list(improved)
    
    if best_net is not None:
        return best_net
    else:
        return _seed_oe_transposition(N)
