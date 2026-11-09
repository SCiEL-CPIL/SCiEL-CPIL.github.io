"""Genetic algorithm for size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 82.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _precompute_targets():
    """Precompute expected sorted states for all 2^N boolean inputs."""
    targets = np.zeros(1 << N, dtype=np.int32)
    for mask in range(1 << N):
        ones = bin(mask).count('1')
        if ones == 0:
            targets[mask] = 0
        elif ones == N:
            targets[mask] = (1 << N) - 1
        else:
            targets[mask] = ((1 << ones) - 1) << (N - ones)
    return targets

def _simulate_network_bitparallel(net, targets):
    """
    Simulate a network on all 2^N inputs using bit-parallel NumPy.
    Returns the number of unsorted inputs (0 means valid sorting network).
    """
    if len(net) == 0:
        # Check if all inputs are already sorted (only true for N<=1)
        if N <= 1:
            return 0
        return (1 << N) - 1
    
    # state[i] = the value on wire i for all 2^N inputs
    # We represent this as: for each wire i, a bit array of length 2^N
    # where bit k is 1 if input k has a 1 on wire i.
    # Initially, wire i has bit i set for input k iff k has bit i set.
    
    # For input k (0 to 2^N-1), wire i gets value (k >> i) & 1
    # So state[i][k] = (k >> i) & 1
    
    # We can build this efficiently:
    # state[i] is a uint32 array of length 2^N where state[i][k] = (k >> i) & 1
    
    # Actually, let's use a different representation:
    # For each of the 2^N inputs, we have an N-bit state.
    # We process all inputs in parallel.
    
    # inputs[k] = k (the initial state for input k)
    inputs = np.arange(1 << N, dtype=np.int32)
    
    for i, j in net:
        # For each input, if wire i has 1 and wire j has 0, swap them
        # bit_i = (inputs >> i) & 1
        # bit_j = (inputs >> j) & 1
        # condition: bit_i == 1 and bit_j == 0
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        cond = (bit_i == 1) & (bit_j == 0)
        
        # If cond: clear bit i, set bit j
        # inputs = (inputs & ~(1 << i)) | (1 << j) for those where cond is true
        mask_i = 1 << i
        mask_j = 1 << j
        inputs = np.where(cond, (inputs & ~mask_i) | mask_j, inputs)
    
    # Count how many inputs are not sorted
    unsorted = np.sum(inputs != targets)
    return int(unsorted)

def _simulate_network_sampled(net, num_samples=5000, rng=None):
    """
    Sampled simulation for quick fitness evaluation.
    Returns the number of unsorted sampled inputs.
    """
    if rng is None:
        rng = random.Random(42)
    
    unsorted_count = 0
    for _ in range(num_samples):
        mask = rng.getrandbits(N)
        state = mask
        for i, j in net:
            if (state & (1 << i)) and not (state & (1 << j)):
                state = (state & ~(1 << i)) | (1 << j)
        
        ones = bin(state).count('1')
        if ones == 0:
            expected = 0
        elif ones == N:
            expected = (1 << N) - 1
        else:
            expected = ((1 << ones) - 1) << (N - ones)
        
        if state != expected:
            unsorted_count += 1
    
    return unsorted_count

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _bubble_sort_network():
    """Generate a bubble sort network (adjacent comparators, multiple passes)."""
    net = []
    for _ in range(N):
        for i in range(N - 1):
            net.append((i, i + 1))
    return net

def _bitonic_sort_network():
    """Generate a bitonic sort network."""
    def bitonic_merge(start, length, direction):
        """Generate comparators for bitonic merge of a segment."""
        comps = []
        if length == 1:
            return comps
        half = length // 2
        # Merge two bitonic halves
        for i in range(half):
            if direction == 'up':
                # Compare and swap so smaller goes to lower index
                comps.append((start + i, start + i + half))
            else:
                # Compare and swap so larger goes to lower index
                comps.append((start + i + half, start + i))
        
        # Recursively merge the two halves
        comps.extend(bitonic_merge(start, half, 'up'))
        comps.extend(bitonic_merge(start + half, half, 'down'))
        return comps
    
    def bitonic_sort(start, length, direction):
        """Generate comparators for bitonic sort of a segment."""
        comps = []
        if length <= 1:
            return comps
        half = length // 2
        # Sort first half up, second half down (to create bitonic sequence)
        comps.extend(bitonic_sort(start, half, 'up'))
        comps.extend(bitonic_sort(start + half, half, 'down'))
        # Merge the bitonic sequence
        comps.extend(bitonic_merge(start, length, direction))
        return comps
    
    return bitonic_sort(0, N, 'up')

def _random_network(size, rng):
    """Generate a random network of given size."""
    net = []
    for _ in range(size):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        net.append((i, j))
    return net

def _crossover(net1, net2, rng):
    """Uniform crossover between two networks."""
    length = max(len(net1), len(net2))
    child = []
    for i in range(length):
        if rng.random() < 0.5:
            if i < len(net1):
                child.append(net1[i])
            else:
                child.append((rng.randint(0, N - 2), rng.randint(0, N - 1)))
                # Ensure i < j
                a, b = child[-1]
                if a > b:
                    a, b = b, a
                if a == b:
                    b = min(a + 1, N - 1)
                child[-1] = (a, b)
        else:
            if i < len(net2):
                child.append(net2[i])
            else:
                child.append((rng.randint(0, N - 2), rng.randint(0, N - 1)))
                a, b = child[-1]
                if a > b:
                    a, b = b, a
                if a == b:
                    b = min(a + 1, N - 1)
                child[-1] = (a, b)
    return child

def _mutate(net, rng, mutation_rate=0.1):
    """Mutate a network by randomly changing, adding, or removing comparators."""
    child = list(net)
    
    # Randomly modify existing comparators
    for i in range(len(child)):
        if rng.random() < mutation_rate:
            a = rng.randint(0, N - 2)
            b = rng.randint(a + 1, N - 1)
            child[i] = (a, b)
    
    # Randomly add comparators
    if rng.random() < mutation_rate:
        a = rng.randint(0, N - 2)
        b = rng.randint(a + 1, N - 1)
        child.append((a, b))
    
    # Randomly remove comparators
    if len(child) > 1 and rng.random() < mutation_rate:
        idx = rng.randint(0, len(child) - 1)
        child.pop(idx)
    
    return child

def _evaluate_fitness(net, targets, use_full=False, num_samples=2000, rng=None):
    """
    Evaluate the fitness of a network.
    Returns (unsorted_count, is_valid).
    """
    if use_full:
        unsorted = _simulate_network_bitparallel(net, targets)
        return unsorted, (unsorted == 0)
    else:
        unsorted = _simulate_network_sampled(net, num_samples=num_samples, rng=rng)
        # Estimate full unsorted count
        estimated_full = unsorted * (1 << N) // num_samples
        return estimated_full, (unsorted == 0)

def _genetic_algorithm(seed_net, targets, deadline):
    """
    Run a genetic algorithm to find a smaller sorting network.
    """
    rng = random.Random(42)
    
    # Initialize population
    pop_size = 30
    population = []
    
    # Add seed network
    population.append(list(seed_net))
    
    # Add some known constructions
    bubble = _bubble_sort_network()
    if len(bubble) < len(seed_net):
        population.append(list(bubble))
    
    bitonic = _bitonic_sort_network()
    population.append(list(bitonic))
    
    # Add random networks of various sizes
    for size in [60, 70, 80, 90, 100, 120]:
        population.append(_random_network(size, rng))
    
    # Pad population with mutated copies
    while len(population) < pop_size:
        parent = rng.choice(population)
        population.append(_mutate(parent, rng, mutation_rate=0.15))
    
    # Evaluate initial fitness (sampled)
    fitnesses = []
    for net in population:
        fitness, _ = _evaluate_fitness(net, targets, use_full=False, num_samples=1000, rng=rng)
        fitnesses.append(fitness)
    
    best_net = None
    best_fitness = float('inf')
    best_len = float('inf')
    
    # Find initial best
    for i, net in enumerate(population):
        if fitnesses[i] == 0:
            if len(net) < best_len:
                best_len = len(net)
                best_net = list(net)
    
    if best_net is None:
        # Use the seed as fallback
        best_net = list(seed_net)
        best_len = len(seed_net)
    
    generation = 0
    no_improvement_count = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Selection: keep top half
        indexed_fitnesses = list(zip(fitnesses, range(len(population))))
        indexed_fitnesses.sort(key=lambda x: (x[0], len(population[x[1]])))
        
        survivors = [population[i] for _, i in indexed_fitnesses[:pop_size // 2]]
        
        # Create new population
        new_population = list(survivors)
        
        while len(new_population) < pop_size and time.time() < deadline:
            # Crossover
            if len(population) >= 2 and rng.random() < 0.7:
                idx1, idx2 = rng.sample(range(len(population)), 2)
                child = _crossover(population[idx1], population[idx2], rng)
            else:
                # Mutation
                parent = rng.choice(population)
                child = _mutate(parent, rng, mutation_rate=0.1)
            
            # Evaluate new individual (sampled)
            fitness, is_valid = _evaluate_fitness(child, targets, use_full=False, num_samples=1000, rng=rng)
            
            if is_valid and len(child) < best_len:
                # Full verification
                unsorted = _simulate_network_bitparallel(child, targets)
                if unsorted == 0:
                    best_net = list(child)
                    best_len = len(child)
                    no_improvement_count = 0
                else:
                    fitness = unsorted
            elif not is_valid:
                # Keep the sampled fitness
                pass
            
            new_population.append(child)
        
        population = new_population
        
        # Re-evaluate fitnesses
        fitnesses = []
        for net in population:
            fitness, _ = _evaluate_fitness(net, targets, use_full=False, num_samples=1000, rng=rng)
            fitnesses.append(fitness)
        
        # Check for improvement
        current_best_in_pop = min(fitnesses)
        if current_best_in_pop == 0:
            # Find the shortest valid network in population
            for i, net in enumerate(population):
                if fitnesses[i] == 0:
                    if len(net) < best_len:
                        unsorted = _simulate_network_bitparallel(net, targets)
                        if unsorted == 0:
                            best_net = list(net)
                            best_len = len(net)
                            no_improvement_count = 0
                            break
        
        if no_improvement_count > 5:
            # Inject more diversity
            for _ in range(5):
                if time.time() < deadline:
                    size = rng.randint(50, 150)
                    population.append(_random_network(size, rng))
                    fitness, _ = _evaluate_fitness(population[-1], targets, use_full=False, num_samples=1000, rng=rng)
                    fitnesses.append(fitness)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
    
    return best_net, best_len

def _improve_by_removal(net, targets, deadline):
    """
    Try to improve a network by removing redundant comparators.
    """
    current = list(net)
    best = list(current)
    best_len = len(current)
    
    # First, do a full validation
    unsorted = _simulate_network_bitparallel(current, targets)
    if unsorted != 0:
        return current
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        if not current:
            break
        
        # Try removing each comparator in random order
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            
            # Quick sampled check first
            sampled_unsorted = _simulate_network_sampled(candidate, num_samples=2000, rng=rng)
            if sampled_unsorted > 0:
                continue
            
            # Full check
            unsorted = _simulate_network_bitparallel(candidate, targets)
            if unsorted == 0:
                current = candidate
                best = list(current)
                best_len = len(current)
                improved = True
                break
        
        if not improved:
            # Try some random perturbations
            for _ in range(5):
                if time.time() > deadline:
                    break
                new_net = list(current)
                if len(new_net) >= 2:
                    i, j = rng.sample(range(len(new_net)), 2)
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                    sampled_unsorted = _simulate_network_sampled(new_net, num_samples=2000, rng=rng)
                    if sampled_unsorted == 0:
                        unsorted = _simulate_network_bitparallel(new_net, targets)
                        if unsorted == 0:
                            current = new_net
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                            improved = True
                            break
    
    return best

def construct():
    start_time = time.time()
    
    # Precompute targets
    targets = _precompute_targets()
    
    # Start with the seed network
    seed_net = _odd_even_transposition()
    
    # Verify seed
    unsorted = _simulate_network_bitparallel(seed_net, targets)
    if unsorted != 0:
        raise ValueError("Seed network is invalid")
    
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Strategy 1: Run genetic algorithm
    if time.time() < DEADLINE - 10:
        ga_deadline = time.time() + min(40, DEADLINE - time.time() - 5)
        ga_net, ga_len = _genetic_algorithm(seed_net, targets, ga_deadline)
        if ga_net is not None and ga_len < best_len:
            best_net = ga_net
            best_len = ga_len
    
    # Strategy 2: Try bitonic sort network
    if time.time() < DEADLINE - 15:
        bitonic = _bitonic_sort_network()
        unsorted = _simulate_network_bitparallel(bitonic, targets)
        if unsorted == 0 and len(bitonic) < best_len:
            best_net = list(bitonic)
            best_len = len(bitonic)
        
        # Try to improve bitonic
        if time.time() < DEADLINE - 10:
            improved = _improve_by_removal(bitonic, targets, deadline=time.time() + min(10, DEADLINE - time.time() - 5))
            if len(improved) < best_len:
                best_net = improved
                best_len = len(improved)
    
    # Strategy 3: Try bubble sort network
    if time.time() < DEADLINE - 10:
        bubble = _bubble_sort_network()
        unsorted = _simulate_network_bitparallel(bubble, targets)
        if unsorted == 0 and len(bubble) < best_len:
            best_net = list(bubble)
            best_len = len(bubble)
        
        if time.time() < DEADLINE - 5:
            improved = _improve_by_removal(bubble, targets, deadline=time.time() + min(5, DEADLINE - time.time() - 3))
            if len(improved) < best_len:
                best_net = improved
                best_len = len(improved)
    
    # Strategy 4: Final improvement on best network
    if time.time() < DEADLINE - 3:
        improved = _improve_by_removal(best_net, targets, deadline=time.time() + min(3, DEADLINE - time.time() - 2))
        if len(improved) < best_len:
            best_net = improved
            best_len = len(improved)
    
    # Final verification
    unsorted = _simulate_network_bitparallel(best_net, targets)
    if unsorted != 0:
        # Fallback to seed
        best_net = seed_net
    
    if len(best_net) > 4096:
        best_net = best_net[:4096]
    
    return best_net

# EVOLVE-BLOCK-END
