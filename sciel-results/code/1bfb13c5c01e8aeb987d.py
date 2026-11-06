import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Generate seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification using bit-vector simulation
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
    
    # 3. Aggressive removal function
    def try_remove_all(current_net, max_rounds=5):
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
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
    
    # 4. Mutation strategies
    def add_random_and_optimize(current_net):
        """Add a random comparator, then aggressively remove redundancies."""
        new_i = random.randint(0, N - 2)
        gap = random.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 7])
        new_j = min(N - 1, new_i + gap)
        if new_i >= new_j:
            return current_net
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        return try_remove_all(candidate, max_rounds=3)
    
    def add_targeted_and_optimize(current_net):
        """Add a comparator between non-adjacent wires, then optimize."""
        dist = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        if dist >= N:
            dist = N - 1
        new_i = random.randint(0, N - 1 - dist)
        new_j = new_i + dist
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        return try_remove_all(candidate, max_rounds=3)
    
    def replace_and_optimize(current_net):
        """Replace a random comparator with a new one, then optimize."""
        if not current_net:
            return current_net
        idx = random.randint(0, len(current_net) - 1)
        new_i = random.randint(0, N - 2)
        gap = random.choice([1, 1, 2, 2, 3, 4, 5, 6])
        new_j = min(N - 1, new_i + gap)
        if new_i >= new_j:
            return current_net
            
        candidate = current_net[:]
        candidate[idx] = (new_i, new_j)
        
        if not is_valid(candidate):
            return current_net
        
        return try_remove_all(candidate, max_rounds=3)
    
    def move_and_optimize(current_net):
        """Move a comparator to a different position, then optimize."""
        if len(current_net) < 2:
            return current_net
        i = random.randint(0, len(current_net) - 1)
        j = random.randint(0, len(current_net) - 1)
        if i == j:
            return current_net
        
        candidate = current_net[:]
        comp = candidate.pop(i)
        if i < j:
            j -= 1
        candidate.insert(j, comp)
        
        if not is_valid(candidate):
            return current_net
        
        return try_remove_all(candidate, max_rounds=2)
    
    def swap_and_optimize(current_net):
        """Swap two comparators, then optimize."""
        if len(current_net) < 2:
            return current_net
        i = random.randint(0, len(current_net) - 1)
        j = random.randint(0, len(current_net) - 1)
        if i == j:
            return current_net
        
        candidate = current_net[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        
        if not is_valid(candidate):
            return current_net
        
        return try_remove_all(candidate, max_rounds=2)
    
    # 5. Simulated Annealing
    current_network = try_remove_all(best_network, max_rounds=10)
    current_cost = len(current_network)
    best_cost = current_cost
    best_network = current_network[:]
    
    T = 3.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Track stagnation for restarts
    stagnation_count = 0
    max_stagnation = 200
    restart_count = 0
    max_restarts = 5
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with new temperature
            T = 3.0
            restart_count += 1
            if restart_count > max_restarts:
                break
            # Perturb current solution
            for _ in range(random.randint(3, 8)):
                current_network = add_random_and_optimize(current_network)
                current_cost = len(current_network)
        
        # Choose mutation strategy
        strategy = random.random()
        
        if strategy < 0.25:
            new_net = add_random_and_optimize(current_network)
        elif strategy < 0.45:
            new_net = add_targeted_and_optimize(current_network)
        elif strategy < 0.60:
            new_net = replace_and_optimize(current_network)
        elif strategy < 0.75:
            new_net = move_and_optimize(current_network)
        else:
            new_net = swap_and_optimize(current_network)
        
        new_cost = len(new_net)
        
        # Acceptance decision
        if new_cost <= current_cost:
            current_network = new_net
            current_cost = new_cost
            stagnation_count = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_network = current_network[:]
        else:
            delta = new_cost - current_cost
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_network = new_net
                    current_cost = new_cost
                    stagnation_count += 1
                else:
                    stagnation_count += 1
        
        # Check for stagnation
        if stagnation_count > max_stagnation:
            # Restart from best with perturbation
            current_network = best_network[:]
            current_cost = best_cost
            for _ in range(random.randint(5, 10)):
                current_network = add_random_and_optimize(current_network)
                current_cost = len(current_network)
            stagnation_count = 0
        
        T *= cooling_rate
    
    # Final aggressive cleanup
    if time.time() < deadline - 2.0:
        final_network = try_remove_all(current_network, max_rounds=15)
        if len(final_network) < best_cost:
            best_cost = len(final_network)
            best_network = final_network[:]
    
    # Multiple final cleanup passes
    passes = 0
    while time.time() < deadline - 1.0 and passes < 10:
        passes += 1
        final_network = try_remove_all(best_network, max_rounds=10)
        if len(final_network) < best_cost:
            best_cost = len(final_network)
            best_network = final_network[:]
        else:
            break
    
    return best_network
# EVOLVE-BLOCK-END
