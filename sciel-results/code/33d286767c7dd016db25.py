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
    
    # 2. Precompute initial state masks for fast verification
    # For each wire j, mask has bit k set if input k has 1 on wire j
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask
    
    # Precompute: for each wire position, what should the final sorted state be?
    # After sorting, wire k should have 1s for the top 2^(N-1-k) inputs
    # i.e., inputs where the bit at position (N-1-k) is 1 (since sorted descending)
    # Actually for ascending sort: wire 0 gets smallest, wire N-1 gets largest
    # Wire k should have 1s for inputs that rank >= k
    # In boolean case, wire k has 1 iff at least (N-k) of the N bits are 1
    # This is complex; let's just check the monotonicity condition
    
    def is_valid_fast(network):
        """Fast verification using precomputed initial states."""
        state = initial_state[:]  # Copy list
        for i, j in network:
            si = state[i]
            sj = state[j]
            state[i] = si & sj
            state[j] = si | sj
        
        # Check: for each adjacent pair, state[k] should be subset of state[k+1]
        # i.e., state[k] & ~state[k+1] == 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid_fast(best_network):
        return best_network

    random.seed(123)
    
    # 3. Optimized removal: try removals more efficiently with early termination
    def try_remove_all(current_net, max_rounds=3):
        if time.time() > deadline:
            return current_net
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
            n = len(current_net)
            # Try removal in random order
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                # Quick check: can we remove this comparator?
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid_fast(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    # Add a comparator biased toward long distances, then optimize
    def add_and_optimize(current_net):
        # Bias toward longer distances (more useful for sorting)
        r = random.random()
        if r < 0.3:
            dist = random.randint(2, 4)
        elif r < 0.6:
            dist = random.randint(5, 8)
        else:
            dist = random.randint(9, 16)
        
        new_i = random.randint(0, N - 1 - dist)
        new_j = new_i + dist
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid_fast(candidate):
            return current_net
        
        optimized = try_remove_all(candidate, max_rounds=2)
        return optimized

    current_network = try_remove_all(best_network, max_rounds=5)
    
    # Simulated annealing parameters
    initial_temp = 8.0
    cooling_rate = 0.997
    min_temp = 0.01
    
    temperature = initial_temp
    
    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break
        
        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)
        
        if strategy < 0.30:
            # Add a biased comparator and optimize
            new_net = add_and_optimize(current_network)
            if len(new_net) <= len(current_network):
                current_network = new_net
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
                temperature *= cooling_rate
                continue
        
        elif strategy < 0.45:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)
        
        elif strategy < 0.60:
            # Move a comparator to a different position
            if n > 2:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
        
        elif strategy < 0.75:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif strategy < 0.90:
            # Replace a comparator with a nearby wire pair
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                new_i = max(0, min(N - 2, old_i + random.randint(-3, 3)))
                new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-2, 2)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)
        
        else:
            # Reorder a segment
            if n > 3:
                start = random.randint(0, n - 4)
                end = random.randint(start + 2, n - 1)
                segment = neighbor[start:end+1]
                random.shuffle(segment)
                neighbor[start:end+1] = segment
        
        # Check validity
        if not is_valid_fast(neighbor):
            temperature *= cooling_rate
            continue
        
        # Light cleanup
        neighbor = try_remove_all(neighbor, max_rounds=1)
        
        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost
        
        if delta <= 0:
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        else:
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor
        
        temperature *= cooling_rate
    
    # Final aggressive cleanup
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            final_network = try_remove_all(current_network, max_rounds=5)
            if len(final_network) < len(best_network):
                best_network = final_network[:]
                current_network = best_network[:]
    
    return best_network
# EVOLVE-BLOCK-END
