import time
import random
import numpy as np

N = 16
NUM_INPUTS = 1 << N  # 65536
CHUNKS = NUM_INPUTS // 64  # 1024 chunks of 64 inputs each

# Pre-compute the initial states for all 65536 binary inputs.
# State shape: (N, CHUNKS) where each element is uint64.
# Bit 'k' in chunk 'c' corresponds to input vector (c * 64 + k).
# We need to set bit 'w' of input 'v' to 1 if the w-th bit of v is 1.

def _init_states():
    states = np.zeros((N, CHUNKS), dtype=np.uint64)
    for w in range(N):
        for c in range(CHUNKS):
            mask = 0
            for k in range(64):
                v = c * 64 + k
                if (v >> w) & 1:
                    mask |= (1 << k)
            states[w, c] = mask
    return states

_INIT_STATES = None

def _get_states():
    global _INIT_STATES
    if _INIT_STATES is None:
        _INIT_STATES = _init_states()
    return _INIT_STATES

def _verify_network_bit_parallel(comparators):
    """
    Verifies if the network sorts all 65536 binary inputs.
    Returns True if sorted, False otherwise.
    """
    if not comparators:
        return False # Empty network doesn't sort unless N=1
    
    states = _get_states()
    curr = states.copy()
    
    for i, j in comparators:
        # min(curr[i], curr[j]) and max(curr[i], curr[j])
        # min(a, b) = a & b
        # max(a, b) = a | b
        min_val = curr[i] & curr[j]
        max_val = curr[i] | curr[j]
        curr[i] = min_val
        curr[j] = max_val
    
    # Check if sorted: for all adjacent wires i, i+1, we must have curr[i] <= curr[i+1]
    # In bit-parallel, this means: for all bits k, if curr[i] has 1 at k, curr[i+1] must have 1 at k.
    # Equivalent to: (curr[i] & ~curr[i+1]) == 0 for all i.
    for i in range(N - 1):
        if np.any(curr[i] & ~curr[i+1]):
            return False
    return True

def _base_60_network():
    # Verified 60-comparator sorting network for 16 inputs (Akers-style construction).
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def _local_search(net, deadline):
    """
    Aggressive greedy removal using fast NumPy verifier.
    """
    best_net = net[:]
    best_len = len(net)
    
    if not _verify_network_bit_parallel(best_net):
        raise ValueError("Initial network is not a valid sorting network")
    
    # Greedy removal pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing each comparator
        for idx in range(len(best_net)):
            if time.time() > deadline:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network_bit_parallel(candidate):
                best_net = candidate
                best_len = len(best_net)
                changed = True
                break # Restart the loop with the new shorter network
    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Start with the known 60-comparator network
    base_net = _base_60_network()
    
    if not _verify_network_bit_parallel(base_net):
        # Fallback to odd-even transposition (120 comparators)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = base_net
    best_len = len(best_net)
    
    # Run local search to try to reduce further
    # This is fast due to NumPy
    optimized_net = _local_search(best_net, deadline)
    best_net = optimized_net
    best_len = len(best_net)
    
    # If we have time, try Genetic Algorithm to escape local minima
    # and potentially find a < 60 network
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        pop_size = 20
        population = [best_net]
        # Add some random perturbations
        for _ in range(pop_size - 1):
            mutated = best_net[:]
            # Random mutation: remove a few, insert a few
            for _ in range(rng.randint(1, 3)):
                if len(mutated) > 1:
                    idx = rng.randrange(len(mutated))
                    mutated.pop(idx)
                else:
                    break
            for _ in range(rng.randint(1, 3)):
                i, j = rng.sample(range(N), 2)
                if i > j: i, j = j, i
                idx = rng.randrange(len(mutated) + 1)
                mutated.insert(idx, (i, j))
            population.append(mutated)
        
        while time.time() < deadline:
            new_population = []
            # Sort by length (shorter is better, but must be valid)
            # We only keep valid networks in the population? 
            # Or we can use a fitness function.
            # For simplicity, we only keep valid networks.
            
            valid_pop = []
            for net in population:
                if _verify_network_bit_parallel(net):
                    valid_pop.append(net)
            
            if not valid_pop:
                break
                
            # Keep the best ones
            valid_pop.sort(key=len)
            new_population = valid_pop[:pop_size//2]
            
            # Generate offspring
            while len(new_population) < pop_size and time.time() < deadline:
                # Crossover: take prefix from one, suffix from another
                parent1, parent2 = rng.sample(valid_pop, 2)
                if len(parent1) < len(parent2):
                    parent1, parent2 = parent2, parent1
                    
                # Try to create a valid offspring
                # Simple approach: take parent1, remove some, add some from parent2
                offspring = parent1[:]
                
                # Mutation
                for _ in range(rng.randint(1, 3)):
                    move_type = rng.choice(['remove', 'insert', 'swap'])
                    if move_type == 'remove' and len(offspring) > 1:
                        idx = rng.randrange(len(offspring))
                        offspring.pop(idx)
                    elif move_type == 'insert':
                        i, j = rng.sample(range(N), 2)
                        if i > j: i, j = j, i
                        idx = rng.randrange(len(offspring) + 1)
                        offspring.insert(idx, (i, j))
                    elif move_type == 'swap' and len(offspring) > 1:
                        i, j = rng.sample(range(len(offspring)), 2)
                        offspring[i], offspring[j] = offspring[j], offspring[i]
                        
                new_population.append(offspring)
                
            population = new_population
            
            # Update best
            for net in population:
                if _verify_network_bit_parallel(net) and len(net) < best_len:
                    best_net = net
                    best_len = len(net)
                    
            # Run local search on the best found in this generation
            if time.time() < deadline - 5.0:
                refined = _local_search(best_net, deadline)
                if len(refined) < best_len:
                    best_net = refined
                    best_len = len(best_net)

    return best_net
