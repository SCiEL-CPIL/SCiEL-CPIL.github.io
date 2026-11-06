import time
import random
import math
import numpy as np

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
    
    # 2. Fast verification using numpy
    # Precompute initial state as numpy array of uint64
    # Each wire holds a 131072-bit value (2048 uint64 words)
    WORDS = (NUM_INPUTS + 63) // 64  # 2048 words
    
    def make_initial_state():
        """Create initial state: wire j has bit k set if input k has bit j = 1"""
        state = np.zeros((N, WORDS), dtype=np.uint64)
        for j in range(N):
            for k in range(NUM_INPUTS):
                if (k >> j) & 1:
                    state[j, k // 64] |= (1 << (k % 64))
        return state

    initial_state = make_initial_state()
    
    def is_valid(network):
        """Fast verification using numpy vectorized operations"""
        state = initial_state.copy()
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Check sorted: for each adjacent pair, state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if np.any(state[k] & ~state[k + 1]):
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(123)
    
    # 3. Simulated Annealing with aggressive optimization
    current_network = best_network[:]
    
    def try_remove_all(current_net, max_rounds=3):
        """Aggressively try to remove any redundant comparators."""
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

    # Simulated annealing parameters
    initial_temp = 5.0
    cooling_rate = 0.997
    min_temp = 0.01
    
    temperature = initial_temp
    current_network = try_remove_all(current_network, max_rounds=5)
    
    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)
        
        if strategy < 0.25:
            # Add a random comparator
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i != new_j:
                pos = random.randint(0, len(neighbor))
                neighbor.insert(pos, (new_i, new_j))
        
        elif strategy < 0.45:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)
        
        elif strategy < 0.65:
            # Move a comparator to a nearby position (local rearrangement)
            if n > 2:
                i = random.randint(0, n - 1)
                offset = random.choice([-2, -1, 1, 2])
                j = i + offset
                if 0 <= j < n and j != i:
                    comp = neighbor.pop(i)
                    neighbor.insert(j, comp)
        
        elif strategy < 0.8:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        else:
            # Replace a comparator with a nearby wire pair
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                # Try to replace with a similar comparator (close wires)
                new_i = max(0, min(N - 2, old_i + random.randint(-2, 2)))
                new_j = max(1, min(N - 1, new_i + 1 + random.randint(0, 3)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)
        
        # Check validity
        if not is_valid(neighbor):
            temperature *= cooling_rate
            continue
        
        # Light cleanup: only one round of removal to save time
        neighbor = try_remove_all(neighbor, max_rounds=1)
        
        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost
        
        if delta <= 0:
            # Accept improvement
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        else:
            # Accept worse solution with probability exp(-delta/T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor
        
        temperature *= cooling_rate
    
    # Final cleanup with more aggressive removal
    if time.time() < deadline:
        final_network = try_remove_all(current_network, max_rounds=10)
        if len(final_network) < len(best_network):
            best_network = final_network[:]
    
    return best_network
# EVOLVE-BLOCK-END
