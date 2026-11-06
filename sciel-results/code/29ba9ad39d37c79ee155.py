import time
import random

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Local Search with aggressive add-remove
    current_network = best_network[:]
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            # Try removals in random order to escape local minima
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    def add_and_optimize(current_net):
        """Add a random comparator, then aggressively remove redundancies."""
        new_i = random.randint(0, N - 1)
        new_j = random.randint(0, N - 1)
        if new_i > new_j:
            new_i, new_j = new_j, new_i
        if new_i == new_j:
            return current_net
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        # Aggressively remove redundancies from the enlarged network
        optimized = try_remove_all(candidate)
        return optimized

    while time.time() < deadline:
        # Phase 1: Aggressively remove
        optimized = try_remove_all(current_network)
        if len(optimized) < len(current_network):
            current_network = optimized
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        
        if time.time() > deadline:
            break
        
        # Occasionally do a more thorough removal pass
        if random.random() < 0.1:
            optimized = try_remove_all(current_network)
            if len(optimized) < len(current_network):
                current_network = optimized
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
        
        # Phase 2: Mutate - try different strategies
        strategy = random.random()
        
        if strategy < 0.25:
            # Add and optimize (most likely to find improvements)
            new_net = add_and_optimize(current_network)
            if len(new_net) <= len(current_network):
                current_network = new_net
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
        
        elif strategy < 0.45:
            # Move a comparator to a different position
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    comp = candidate.pop(i)
                    candidate.insert(j, comp)
                    if is_valid(candidate):
                        optimized = try_remove_all(candidate)
                        if len(optimized) <= len(current_network):
                            current_network = optimized
                            if len(current_network) < len(best_network):
                                best_network = current_network[:]
        
        elif strategy < 0.65:
            # Swap two comparators
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if is_valid(candidate):
                        # Try to remove redundancies after swap
                        optimized = try_remove_all(candidate)
                        if len(optimized) <= len(current_network):
                            current_network = optimized
                            if len(current_network) < len(best_network):
                                best_network = current_network[:]
        
        elif strategy < 0.8:
            # Replace one comparator
            if len(current_network) > 0:
                i = random.randint(0, len(current_network) - 1)
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    candidate = current_network[:]
                    candidate[i] = (new_i, new_j)
                    if is_valid(candidate):
                        optimized = try_remove_all(candidate)
                        if len(optimized) <= len(current_network):
                            current_network = optimized
                            if len(current_network) < len(best_network):
                                best_network = current_network[:]
        
        else:
            # Targeted mutation: try adding comparators between non-adjacent wires
            # that are likely to need sorting, then optimize
            additions = random.randint(1, 2)
            candidate = current_network[:]
            for _ in range(additions):
                # Bias toward longer-distance comparators which are often needed
                dist = random.choice([2, 3, 4, 5, 6, 7, 8])
                new_i = random.randint(0, N - 1 - dist)
                new_j = new_i + dist
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (new_i, new_j))
            
            if is_valid(candidate):
                optimized = try_remove_all(candidate)
                if len(optimized) <= len(current_network):
                    current_network = optimized
                    if len(current_network) < len(best_network):
                        best_network = current_network[:]

    return best_network
# EVOLVE-BLOCK-END
