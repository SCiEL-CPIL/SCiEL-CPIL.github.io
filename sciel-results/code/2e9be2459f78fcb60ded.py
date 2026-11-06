<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute all Boolean inputs as a numpy array of shape (2^N, N)
# Each row is a bit-vector of length N
def _make_all_inputs():
    """Generate all 2^N Boolean inputs as uint8 array of shape (2^N, N)."""
    n_inputs = 1 << N
    # Use bitwise operations to generate all combinations
    # For each bit position k (0 to N-1), the bit is set in every other input
    # We want: input i has bit k = (i >> k) & 1
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comparators, inputs=None):
    """
    Simulate a compare-exchange network on all Boolean inputs.
    comparators: list of (i, j) pairs
    inputs: optional numpy array; if None, uses all 2^N Boolean inputs
    Returns the output array of shape (num_inputs, N).
    """
    if inputs is None:
        inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        # min goes to i, max goes to j
        # min(a,b) = a & b, max(a,b) = a | b for bits
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Check if the network sorts all Boolean inputs using the zero-one principle."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    # Check if each row is sorted: non-decreasing from left to right
    # For binary, sorted means all 0s then all 1s
    # Check: for each row, no 1 appears before a 0
    # Equivalent: for each adjacent pair in output, output[:,k] <= output[:,k+1]
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def random_network(num_comparators):
    """Generate a random network with the given number of comparators."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def improve_network(comparators, max_steps=5000, deadline=None):
    """
    Try to improve a network by removing redundant comparators or swapping them.
    Uses local search: try removing each comparator, check if still correct.
    """
    if deadline is None:
        deadline = time.time() + 60
    
    best = list(comparators)
    best_len = len(best)
    
    # First, try to remove redundant comparators
    # Iterate through and try removing each one
    i = 0
    while i < len(best):
        if time.time() > deadline - 5:
            break
        # Try removing comparator i
        candidate = best[:i] + best[i+1:]
        if is_sorted_network(candidate):
            best = candidate
            best_len = len(best)
            # Don't increment i, check same position again with new list
        else:
            i += 1
    
    # Try adding comparators at random positions to enable further removals
    # This is a simple hill-climbing step
    attempts = 0
    while time.time() < deadline - 5 and attempts < max_steps:
        attempts += 1
        # Pick a random position to insert a new comparator
        if not best:
            pos = 0
        else:
            pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Check if still sorted
        if is_sorted_network(candidate):
            # Now try to remove some comparators
            k = 0
            improved = True
            while k < len(candidate) and improved:
                improved = False
                # Try removing comparator k
                test = candidate[:k] + candidate[k+1:]
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                    # Don't increment k
                else:
                    k += 1
            if len(candidate) < len(best):
                best = candidate
                best_len = len(best)
    
    return best

def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Start with the seed odd-even transposition network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    best = list(seed)
    
    # Verify seed is correct
    if not is_sorted_network(best):
        raise ValueError("Seed network is not correct")
    
    # Try to improve by removing redundant comparators
    best = improve_network(best, deadline=deadline)
    
    # Try a few random restarts with smaller networks
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        # Generate a random network slightly smaller than current best
        target_len = max(60, len(best) - random.randint(5, 20))
        if target_len < len(best):
            candidate = random_network(target_len)
            # Try to fix it by adding necessary comparators
            # Use a heuristic: start from a known good structure
            # For now, just check if it works (unlikely for random)
            if is_sorted_network(candidate):
                best = candidate
    
    # Final attempt: try to further reduce the best network
    if time.time() < deadline - 5:
        best = improve_network(best, deadline=deadline)
    
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute all Boolean inputs as a numpy array of shape (2^N, N)
def _make_all_inputs():
    """Generate all 2^N Boolean inputs as uint8 array of shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comparators, inputs=None):
    """
    Simulate a compare-exchange network on all Boolean inputs.
    comparators: list of (i, j) pairs
    inputs: optional numpy array; if None, uses all 2^N Boolean inputs
    Returns the output array of shape (num_inputs, N).
    """
    if inputs is None:
        inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Check if the network sorts all Boolean inputs using the zero-one principle."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def _odd_even_merge_sort_network(n):
    """Generate an odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    
    comparators = []
    
    def merge(left, right, stride):
        """Merge two sorted sequences of length stride."""
        nonlocal comparators
        if stride == 1:
            return
        # Recursive merge
        merge(left, right, stride // 2)
        merge(left + stride // 2, right + stride // 2, stride // 2)
        # Cross comparators
        for i in range(stride // 2):
            comparators.append((left + i, right + i))
    
    def sort_network(size):
        """Generate sorting network for given size."""
        nonlocal comparators
        if size == 1:
            return
        half = size // 2
        sort_network(half)
        sort_network(size - half)
        # Merge the two halves
        merge(0, half, half)
    
    sort_network(n)
    return comparators

def _bitonic_sort_network(n):
    """Generate a bitonic sort network for n wires."""
    comparators = []
    
    def bitonic_merge(left, right, stride):
        """Bitonic merge of two sequences."""
        nonlocal comparators
        if stride == 1:
            return
        for i in range(stride):
            comparators.append((left + i, right + i))
        bitonic_merge(left, right, stride // 2)
        bitonic_merge(left + stride // 2, right + stride // 2, stride // 2)
    
    def bitonic_sort(size, start=0, direction=1):
        """Bitonic sort for given size."""
        nonlocal comparators
        if size <= 1:
            return
        half = size // 2
        # Sort first half ascending, second half descending
        bitonic_sort(half, start, 1)
        bitonic_sort(size - half, start + half, -1)
        # Merge
        bitonic_merge(start, start + half, half)
    
    bitonic_sort(n)
    return comparators

def _batcher_odd_even_merge_sort(n):
    """Batcher's odd-even merge sort."""
    comparators = []
    
    def odd_even_merge(left, right, stride):
        """Odd-even merge."""
        nonlocal comparators
        if stride == 1:
            return
        odd_even_merge(left, right, stride // 2)
        odd_even_merge(left + stride // 2, right + stride // 2, stride // 2)
        for i in range(stride // 2):
            comparators.append((left + i, right + i))
    
    def sort(n_wires):
        nonlocal comparators
        if n_wires <= 1:
            return
        half = n_wires // 2
        sort(half)
        sort(n_wires - half)
        odd_even_merge(0, half, half)
    
    sort(n)
    return comparators

def _remove_redundant(comparators, deadline):
    """Try to remove redundant comparators from the network."""
    best = list(comparators)
    i = 0
    while i < len(best):
        if time.time() > deadline - 3:
            break
        candidate = best[:i] + best[i+1:]
        if is_sorted_network(candidate):
            best = candidate
        else:
            i += 1
    return best

def _crossover(parent1, parent2):
    """Uniform crossover between two networks."""
    if not parent1 or not parent2:
        return list(parent1 if parent1 else parent2)
    
    max_len = max(len(parent1), len(parent2))
    child = []
    for i in range(max_len):
        if random.random() < 0.5:
            if i < len(parent1):
                child.append(parent1[i])
        else:
            if i < len(parent2):
                child.append(parent2[i])
    return child

def _mutate(comparators, mutation_rate=0.1):
    """Mutate a network by randomly modifying comparators."""
    if not comparators:
        return comparators
    
    mutated = list(comparators)
    for i in range(len(mutated)):
        r = random.random()
        if r < mutation_rate * 0.5:
            # Remove comparator
            del mutated[i]
            i -= 1
        elif r < mutation_rate:
            # Replace with random comparator
            a = random.randint(0, N - 2)
            b = random.randint(a + 1, N - 1)
            mutated[i] = (a, b)
        elif r < mutation_rate * 1.5:
            # Swap with next comparator if exists
            if i + 1 < len(mutated):
                mutated[i], mutated[i+1] = mutated[i+1], mutated[i]
    return mutated

def _ga_search(initial_networks, deadline, pop_size=50):
    """Genetic algorithm to find minimal sorting network."""
    # Initialize population
    population = []
    for net in initial_networks:
        if is_sorted_network(net):
            population.append(list(net))
    
    # Add some mutated versions
    for _ in range(pop_size - len(population)):
        if population:
            parent = random.choice(population)
            child = _mutate(parent, 0.2)
            if is_sorted_network(child):
                population.append(child)
    
    if not population:
        return []
    
    best = min(population, key=len)
    
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Select parents (tournament selection)
        new_population = []
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            if len(population[p1_idx]) <= len(population[p2_idx]):
                parent1 = population[p1_idx]
            else:
                parent1 = population[p2_idx]
            
            p3_idx = random.randint(0, len(population) - 1)
            p4_idx = random.randint(0, len(population) - 1)
            if len(population[p3_idx]) <= len(population[p4_idx]):
                parent2 = population[p3_idx]
            else:
                parent2 = population[p4_idx]
            
            # Crossover
            child = _crossover(parent1, parent2)
            
            # Mutation
            child = _mutate(child, 0.1)
            
            # Validate
            if is_sorted_network(child):
                new_population.append(child)
        
        # Update population
        population = new_population
        
        # Update best
        current_best = min(population, key=len)
        if len(current_best) < len(best):
            best = current_best
            # Try to remove redundant
            best = _remove_redundant(best, deadline)
    
    return best

def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    random.seed(42)
    
    # Generate initial networks from different families
    initial_networks = []
    
    # 1. Odd-even transposition network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    initial_networks.append(seed)
    
    # 2. Odd-even merge sort
    try:
        oems = _odd_even_merge_sort_network(N)
        if is_sorted_network(oems):
            initial_networks.append(oems)
    except:
        pass
    
    # 3. Bitonic sort
    try:
        bitonic = _bitonic_sort_network(N)
        if is_sorted_network(bitonic):
            initial_networks.append(bitonic)
    except:
        pass
    
    # 4. Batcher's odd-even merge
    try:
        batcher = _batcher_odd_even_merge_sort(N)
        if is_sorted_network(batcher):
            initial_networks.append(batcher)
    except:
        pass
    
    # First, try to reduce each initial network
    reduced_networks = []
    for net in initial_networks:
        reduced = _remove_redundant(net, deadline)
        reduced_networks.append(reduced)
    
    # Find the best initial
    best = min(reduced_networks, key=len)
    
    # Run genetic algorithm
    if time.time() < deadline - 10:
        ga_result = _ga_search(reduced_networks, deadline, pop_size=30)
        if ga_result and len(ga_result) < len(best):
            best = ga_result
    
    # Final reduction
    best = _remove_redundant(best, deadline)
    
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
