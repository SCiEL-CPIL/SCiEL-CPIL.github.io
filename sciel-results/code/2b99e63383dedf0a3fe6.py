import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

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

    random.seed(123)
    
    # 3. Simulated Annealing with aggressive add-remove
    current_network = best_network[:]
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
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
        
        optimized = try_remove_all(candidate)
        return optimized

    # Simulated annealing parameters
    initial_temp = 50.0
    cooling_rate = 0.995
    min_temp = 0.1
    current_temp = initial_temp

    while time.time() < deadline and current_temp > min_temp:
        optimized = try_remove_all(current_network)
        if len(optimized) < len(current_network):
            current_network = optimized
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        
        if time.time() > deadline:
            break
        
        strategy = random.random()
        
        if strategy < 0.25:
            new_net = add_and_optimize(current_network)
            new_len = len(new_net)
            cur_len = len(current_network)
            # Accept worse solutions with probability based on temperature
            if new_len <= cur_len or random.random() < math.exp(-(new_len - cur_len) / current_temp):
                current_network = new_net
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
        
        elif strategy < 0.45:
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    comp = candidate.pop(i)
                    candidate.insert(j, comp)
                    if is_valid(candidate):
                        optimized = try_remove_all(candidate)
                        new_len = len(optimized)
                        cur_len = len(current_network)
                        if new_len <= cur_len or random.random() < math.exp(-(new_len - cur_len) / current_temp):
                            current_network = optimized
                            if len(current_network) < len(best_network):
                                best_network = current_network[:]
        
        elif strategy < 0.65:
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if is_valid(candidate):
                        optimized = try_remove_all(candidate)
                        new_len = len(optimized)
                        cur_len = len(current_network)
                        if new_len <= cur_len or random.random() < math.exp(-(new_len - cur_len) / current_temp):
                            current_network = optimized
                            if len(current_network) < len(best_network):
                                best_network = current_network[:]
        
        elif strategy < 0.8:
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
                        new_len = len(optimized)
                        cur_len = len(current_network)
                        if new_len <= cur_len or random.random() < math.exp(-(new_len - cur_len) / current_temp):
                            current_network = optimized
                            if len(current_network) < len(best_network):
                                best_network = current_network[:]
        
        else:
            additions = random.randint(1, 2)
            candidate = current_network[:]
            for _ in range(additions):
                dist = random.choice([2, 3, 4, 5, 6, 7, 8])
                new_i = random.randint(0, N - 1 - dist)
                new_j = new_i + dist
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (new_i, new_j))
            
            if is_valid(candidate):
                optimized = try_remove_all(candidate)
                new_len = len(optimized)
                cur_len = len(current_network)
                if new_len <= cur_len or random.random() < math.exp(-(new_len - cur_len) / current_temp):
                    current_network = optimized
                    if len(current_network) < len(best_network):
                        best_network = current_network[:]
        
        # Cool down
        current_temp *= cooling_rate

    # Final polish: try to remove more comparators from best_network
    if time.time() < deadline:
        polished = try_remove_all(best_network)
        if len(polished) < len(best_network):
            best_network = polished

    return best_network
# EVOLVE-BLOCK-END
