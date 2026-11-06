"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 80.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs as bitmasks
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Sorted values: bitmask x is sorted (0s then 1s) iff x = 2^N - 2^k for k in [0, N]
    sorted_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.int64)
    
    def is_sorted(inputs):
        """Check if all inputs in the array are sorted (bitmask form)."""
        count = 0
        for s in sorted_values:
            count += np.sum(inputs == s)
        return count == len(inputs)
    
    def apply_network(inputs, comparators):
        """Apply a list of comparators to an array of bitmasks."""
        for (i, j) in comparators:
            mask_i = (inputs >> i) & 1
            mask_j = (inputs >> j) & 1
            swap_mask = mask_j & (~mask_i)
            inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
        return inputs
        
    def fast_verify(network):
        """Verify if the network sorts all inputs."""
        if not network:
            return False
        test_inputs = all_inputs.copy()
        result = apply_network(test_inputs, network)
        return is_sorted(result)

    # Start with odd-even transposition network
    initial_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    if not fast_verify(best_network):
        return best_network

    rng = random.Random(42)
    
    # Simulated Annealing
    current_network = list(initial_network)
    current_count = len(current_network)
    
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start_time < DEADLINE - 5:
        # Generate a neighbor
        move_type = rng.random()
        neighbor = list(current_network)
        
        if move_type < 0.3:
            # Remove a comparator
            if len(neighbor) > 1:
                idx = rng.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
        elif move_type < 0.6:
            # Swap adjacent comparators
            if len(neighbor) > 1:
                idx = rng.randint(0, len(neighbor) - 2)
                neighbor[idx], neighbor[idx+1] = neighbor[idx+1], neighbor[idx]
        elif move_type < 0.85:
            # Insert a new comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            idx = rng.randint(0, len(neighbor))
            neighbor.insert(idx, (i, j))
        else:
            # Move a comparator to a different position
            if len(neighbor) > 2:
                idx1 = rng.randint(0, len(neighbor) - 1)
                idx2 = rng.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
        
        # Verify neighbor
        if not fast_verify(neighbor):
            T *= cooling
            continue
            
        new_count = len(neighbor)
        delta = new_count - current_count
        
        # Acceptance probability
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_network = neighbor
            current_count = new_count
            
            if new_count < best_count:
                best_network = list(current_network)
                best_count = new_count
                
                # Greedy removal pass
                improved = True
                while improved and time.time() - start_time < DEADLINE - 5:
                    improved = False
                    for idx in range(len(best_network) - 1, -1, -1):
                        if time.time() - start_time > DEADLINE - 5:
                            break
                        candidate = best_network[:idx] + best_network[idx+1:]
                        if fast_verify(candidate):
                            best_network = candidate
                            best_count = len(best_network)
                            improved = True
                            current_network = list(best_network)
                            current_count = best_count
                            break
                            
        T *= cooling
        
    return best_network
# EVOLVE-BLOCK-END
