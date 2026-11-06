import time
import random
import math

N = 17
NUM_INPUTS = 1 << N
FULL_MASK = (1 << NUM_INPUTS) - 1

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # Precompute the initial state masks for all 2^N inputs
    # state[j] is a bitmask of length 2^N, where bit k is 1 if input k has a 1 on wire j
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
            # Early exit: if any wire is all-zeros or all-ones, it's "settled"
            # but we still need to check the rest. Actually, we can't early exit
            # from simulation, but we can check monotonicity incrementally.
            # Better: check if the pair (i,j) is already in correct order
            # to skip useless work. But the main speedup is in the final check.
        
        # Check if sorted: state[k] should be a subset of state[k+1]
        # i.e., state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    def is_valid_fast(network):
        """Faster validity check with incremental monotonicity tracking."""
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Final check
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    current_network = best_network[:]
    current_cost = len(current_network)
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 5.0  # Higher initial temperature to encourage exploration
    T_min = 0.01
    cooling_rate = 0.995
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            # Try removals in random order
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

    def targeted_swap(current_net):
        """Try to replace a comparator with a more useful one."""
        if len(current_net) == 0:
            return None
        
        # Pick a random comparator to potentially replace
        idx = random.randint(0, len(current_net) - 1)
        old_i, old_j = current_net[idx]
        
        # Generate candidate replacements
        candidates = []
        # Try swapping with neighbors
        for new_i in range(N):
            for new_j in range(new_i + 1, N):
                if (new_i, new_j) == (old_i, old_j):
                    continue
                # Only consider "useful" swaps: ones that might help sorting
                # Heuristic: prefer comparators that are "active" (wires differ)
                candidates.append((new_i, new_j))
        
        random.shuffle(candidates)
        
        # Try up to 10 random replacements
        for new_i, new_j in candidates[:10]:
            if time.time() > deadline:
                return None
            candidate = current_net[:idx] + [(new_i, new_j)] + current_net[idx+1:]
            if is_valid(candidate):
                # Check if it leads to removal opportunities
                optimized = try_remove_all(candidate)
                return optimized
        
        return None

    def grow_and_shrink(current_net, num_adds):
        """Add random comparators, then aggressively remove redundancies."""
        candidate = current_net[:]
        for _ in range(num_adds):
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return None
        
        optimized = try_remove_all(candidate)
        return optimized

    def mutate(current_net):
        """Multi-strategy mutation: grow+shrink, targeted swap, or pure removal."""
        r = random.random()
        if r < 0.4:
            # Grow and shrink
            num_adds = random.choice([1, 1, 2, 2, 3])
            return grow_and_shrink(current_net, num_adds)
        elif r < 0.7:
            # Targeted swap: replace one comparator
            return targeted_swap(current_net)
        elif r < 0.85:
            # Try removing two comparators at once
            if len(current_net) < 2:
                return None
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 2)
            if i2 >= i1:
                i2 += 1
            # Remove both (handle index adjustment)
            if i1 < i2:
                candidate = current_net[:i1] + current_net[i1+1:i2] + current_net[i2+1:]
            else:
                candidate = current_net[:i2] + current_net[i2+1:i1] + current_net[i1+1:]
            if is_valid(candidate):
                return try_remove_all(candidate)
            return None
        else:
            # Grow by 2, then shrink
            return grow_and_shrink(current_net, 2)

    # Main Search Loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # 1. Aggressively reduce current network
        optimized = try_remove_all(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost
        
        if time.time() > deadline:
            break

        # 2. Mutation Phase
        new_net = mutate(current_network)
        
        if new_net is not None:
            new_cost = len(new_net)
            delta = new_cost - current_cost
            
            # Acceptance criterion
            if delta <= 0:
                accept = True
            else:
                # Accept worse solutions with probability exp(-delta/T)
                accept_prob = math.exp(-delta / T)
                accept = random.random() < accept_prob
            
            if accept:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
        
        # Cool down
        T = max(T * cooling_rate, T_min)

    return best_network
# EVOLVE-BLOCK-END
