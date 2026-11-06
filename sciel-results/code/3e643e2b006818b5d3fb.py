import numpy as np
import time

N = 16
# Precompute all 2^16 zero-one inputs
# Shape: (65536, 16)
# We can generate this efficiently
all_inputs = np.unpackbits(
    np.arange(2**N, dtype=np.uint8).reshape(-1, 2**N // 8), 
    axis=1, 
    count=8
).reshape(-1, N)
# Note: np.arange(2**16) creates numbers 0 to 65535.
# We need bits of these numbers.
# Actually, simpler:
# inputs = np.zeros((2**N, N), dtype=np.uint8)
# for i in range(2**N):
#     for j in range(N):
#         inputs[i, j] = (i >> j) & 1
# But vectorized is better.
# Let's use the unpackbits method correctly.
# np.arange(2**16, dtype=np.uint16)
# We need 16 bits per number.
# np.unpackbits expects uint8. So we need 2 bytes per number.
# Let's just generate it via broadcasting or simple loop if N is small, but 65536*16 is 1M ops, trivial.
# Vectorized generation:
# bits = np.arange(2**N, dtype=np.uint16)[:, None]
# masks = (1 << np.arange(N, dtype=np.uint16))[None, :]
# all_inputs = (bits & masks).astype(np.uint8)

# Correct vectorized generation:
bits = np.arange(2**N, dtype=np.uint16)[:, None]
masks = (1 << np.arange(N, dtype=np.uint16))[None, :]
all_inputs = (bits & masks).astype(np.uint8)

def simulate(network, inputs):
    """
    Simulate the sorting network on all zero-one inputs.
    network: list of (i, j) tuples
    inputs: numpy array of shape (65536, 16)
    Returns: boolean array of shape (65536,) indicating if each output is sorted.
    """
    if not network:
        # Check if inputs are already sorted (only the all-zeros or specific cases? No, we need to check if the network sorts ANY input. 
        # If network is empty, it only sorts inputs that are already sorted.
        # We check if the output (which is same as input) is sorted.
        return np.all(inputs[:, :-1] <= inputs[:, 1:], axis=1)
    
    # Copy inputs to avoid modifying original
    state = inputs.copy()
    
    for i, j in network:
        # Compare-exchange: min to i, max to j
        # We need to handle the case where i > j? The task says i < j.
        # min_val = np.minimum(state[:, i], state[:, j])
        # max_val = np.maximum(state[:, i], state[:, j])
        # state[:, i] = min_val
        # state[:, j] = max_val
        
        # In-place update is tricky if we do min then max using updated values.
        # But min(state[i], state[j]) and max(state[i], state[j]) are independent of the order of assignment if we store them first.
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
        
    # Check if sorted
    return np.all(state[:, :-1] <= state[:, 1:], axis=1)

def is_valid(network):
    """
    Check if the network sorts all zero-one inputs.
    """
    sorted_flags = simulate(network, all_inputs)
    return np.all(sorted_flags)

def get_comparator_count(network):
    return len(network)

def construct():
    start_time = time.time()
    deadline = 85.0 # 5s safety margin
    
    # Initial valid network: Odd-even transposition
    initial_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best_network = initial_network
    best_count = len(best_network)
    
    # GA Parameters
    pop_size = 50
    max_generations = 5000
    mutation_rate = 0.05
    crossover_rate = 0.7
    
    # Represent network as bit vector
    # Possible comparators: (i, j) for 0 <= i < j < 16
    # Total pairs: 16*15/2 = 120
    # We need to map pair index to (i, j) and vice versa
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_to_idx = {p: idx for idx, p in enumerate(pairs)}
    num_pairs = len(pairs)
    
    def network_to_bits(net):
        bits = np.zeros(num_pairs, dtype=np.uint8)
        for p in net:
            bits[pair_to_idx[p]] = 1
        return bits
    
    def bits_to_network(bits):
        net = []
        for idx, b in enumerate(bits):
            if b:
                net.append(pairs[idx])
        return net
    
    # Initialize population
    # Start with the initial network and mutate
    pop = []
    initial_bits = network_to_bits(initial_network)
    pop.append(initial_bits)
    for _ in range(pop_size - 1):
        b = initial_bits.copy()
        # Randomly remove some comparators
        for idx in range(num_pairs):
            if b[idx] and np.random.random() < 0.3:
                b[idx] = 0
        pop.append(b)
        
    # Evaluate initial population
    fitnesses = []
    for bits in pop:
        net = bits_to_network(bits)
        if is_valid(net):
            fitnesses.append(len(net))
        else:
            fitnesses.append(float('inf'))
            
    best_fitness = min(fitnesses)
    if best_fitness < best_count:
        best_count = best_fitness
        best_network = bits_to_network(pop[np.argmin(fitnesses)])
        
    for gen in range(max_generations):
        if time.time() > deadline:
            break
            
        # Selection: Tournament
        new_pop = []
        for _ in range(pop_size):
            # Pick 3 random parents
            parents = np.random.choice(pop_size, 3, replace=False)
            parent_idx = parents[np.argmin([fitnesses[p] for p in parents])]
            parent = pop[parent_idx]
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Pick another parent
                mate_idx = np.random.choice(pop_size)
                if mate_idx == parent_idx:
                    continue
                mate = pop[mate_idx]
                # Uniform crossover
                mask = np.random.randint(0, 2, num_pairs)
                child = np.where(mask, parent, mate)
            else:
                child = parent.copy()
                
            # Mutation
            # Flip bits with some probability
            # We want to explore removing comparators (1->0) and adding (0->1)
            # But removing is more likely to be valid if we start from valid? No, removing breaks validity.
            # Adding restores validity but increases count.
            # We want to find valid networks with fewer comparators.
            # So we should bias mutation towards removal?
            # If we remove and it becomes invalid, fitness is inf.
            # If we add, it might become valid.
            # Let's just do random bit flips.
            for idx in range(num_pairs):
                if np.random.random() < mutation_rate:
                    child[idx] = 1 - child[idx]
                    
            new_pop.append(child)
            
        # Evaluate new population
        for i, bits in enumerate(new_pop):
            net = bits_to_network(bits)
            if is_valid(net):
                new_fitness = len(net)
            else:
                new_fitness = float('inf')
                
            # Elitism: Keep the best individual from previous generation
            # Actually, simpler: just replace and track global best
            fitnesses[i] = new_fitness
            pop[i] = bits
            
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_network = net
                if best_fitness < best_count:
                    best_count = best_fitness
                    
    return best_network

# Pre-compute inputs at module level to save time in construct
# Note: all_inputs is already defined above
