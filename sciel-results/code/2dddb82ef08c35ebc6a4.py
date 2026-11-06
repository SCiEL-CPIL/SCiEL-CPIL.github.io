import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Precompute initial state masks for fast verification
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask
    
    # Precompute sorted check masks: for each position k, we need state[k] <= state[k+1]
    # This means (state[k] & ~state[k+1]) == 0 for all k
    
    def is_valid_fast(network):
        """Fast verification using precomputed initial state."""
        state = initial_state[:]  # Copy the initial state
        for i, j in network:
            si = state[i]
            sj = state[j]
            state[i] = si & sj
            state[j] = si | sj
        
        # Check if sorted: for each k, state[k] must be subset of state[k+1]
        # i.e., state[k] & ~state[k+1] == 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    def is_valid(network):
        """Verification with early exit and local variable optimization."""
        state = initial_state[:]
        for i, j in network:
            si = state[i]
            sj = state[j]
            state[i] = si & sj
            state[j] = si | sj
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Aggressive removal function with multiple removal attempts
    def try_remove_all(current_net, max_rounds=3, time_limit=None):
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > time_limit:
                break
            removed = False
            rounds += 1
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    def try_remove_multiple(current_net, num_remove=1, time_limit=None):
        """Try to remove multiple comparators at once for faster convergence."""
        if num_remove <= 0 or len(current_net) <= num_remove:
            return current_net
        
        best_candidate = current_net[:]
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        # Try single removals first
        for idx in indices[:min(50, len(indices))]:
            if time.time() > time_limit:
                break
            candidate = current_net[:idx] + current_net[idx+1:]
            if is_valid(candidate):
                best_candidate = candidate
                break
        
        # If we found a removal, try to remove more
        if len(best_candidate) < len(current_net):
            remaining_indices = list(range(len(best_candidate)))
            random.shuffle(remaining_indices)
            for idx in remaining_indices[:min(30, len(remaining_indices))]:
                if time.time() > time_limit:
                    break
                candidate = best_candidate[:idx] + best_candidate[idx+1:]
                if is_valid(candidate):
                    best_candidate = candidate
                    break
        
        return best_candidate

    # Initial aggressive cleanup
    current_network = try_remove_all(best_network, max_rounds=15, time_limit=deadline)
    if len(current_network) < len(best_network):
        best_network = current_network[:]
    
    # Additional cleanup with multiple removals
    if time.time() < deadline:
        current_network = try_remove_multiple(current_network, num_remove=2, time_limit=deadline)
        if len(current_network) < len(best_network):
            best_network = current_network[:]

    # Simulated Annealing parameters
    initial_temp = 8.0
    cooling_rate = 0.999
    min_temp = 0.005
    
    temperature = initial_temp
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)
        
        if strategy < 0.20:
            # Add a random comparator - biased toward adjacent wires
            new_i = random.randint(0, N - 2)
            new_j = new_i + 1
            # With some probability, use non-adjacent wires
            if random.random() < 0.3:
                new_j = random.randint(new_i + 1, N - 1)
            pos = random.randint(0, len(neighbor))
            neighbor.insert(pos, (new_i, new_j))
        
        elif strategy < 0.40:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)
        
        elif strategy < 0.55:
            # Move a comparator to a different position
            if n > 2:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
        
        elif strategy < 0.70:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif strategy < 0.85:
            # Replace a comparator with a nearby wire pair
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                # Bias toward adjacent or near-adjacent wires
                if random.random() < 0.5:
                    new_i = max(0, min(N - 2, old_i + random.randint(-2, 2)))
                    new_j = new_i + 1
                else:
                    new_i = max(0, min(N - 2, old_i + random.randint(-3, 3)))
                    new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-2, 2)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)
        
        else:
            # Reorder a segment of the network
            if n > 3:
                start = random.randint(0, n - 4)
                end = random.randint(start + 2, n - 1)
                segment = neighbor[start:end+1]
                random.shuffle(segment)
                neighbor[start:end+1] = segment
        
        # Check validity
        if not is_valid(neighbor):
            temperature *= cooling_rate
            stall_count += 1
            continue
        
        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost
        
        if delta <= 0:
            # Accept improvement
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
            stall_count = 0
        else:
            # Accept worse solution with probability exp(-delta/T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor
                    stall_count += 1
                else:
                    stall_count += 1
            else:
                stall_count += 1
        
        # If stalled too long, restart from best
        if stall_count > max_stall:
            current_network = best_network[:]
            stall_count = 0
            # Increase temperature slightly to escape local minimum
            temperature = max(temperature, 5.0)
        
        temperature *= cooling_rate
    
    # Final cleanup with aggressive removal
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            final_network = try_remove_all(current_network, max_rounds=15, time_limit=deadline)
            if len(final_network) < len(best_network):
                best_network = final_network[:]
                current_network = best_network[:]
            else:
                # Try multiple removal
                final_network = try_remove_multiple(current_network, num_remove=2, time_limit=deadline)
                if len(final_network) < len(best_network):
                    best_network = final_network[:]
                    current_network = best_network[:]
                else:
                    break
    
    return best_network
# EVOLVE-BLOCK-END
