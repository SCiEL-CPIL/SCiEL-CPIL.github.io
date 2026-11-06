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
    # For N=18, 2^18 = 262144 inputs
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Sorted values: bitmask x is sorted (0s then 1s) iff x = 2^N - 2^k for k in [0, N]
    # i.e., the set of 1-bits is a suffix
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
    
    # Start with odd-even transposition network
    initial_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(initial_network)
    best_count = len(initial_network)
    
    # Verify initial network
    if not is_sorted(apply_network(all_inputs.copy(), initial_network)):
        # Shouldn't happen, but return as-is if broken
        return initial_network
    
    rng = random.Random(42)
    
    # Greedy removal: try to remove each comparator one by one
    network = list(initial_network)
    
    changed = True
    while changed:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if len(candidate) < len(network):
                # Quick check on a subset first
                subset_size = min(4096, 1 << N)
                test_inputs = all_inputs[:subset_size].copy()
                result = apply_network(test_inputs, candidate)
                if is_sorted(result):
                    # Full check
                    full_result = apply_network(all_inputs.copy(), candidate)
                    if is_sorted(full_result):
                        network = candidate
                        changed = True
                        best_network = list(network)
                        best_count = len(network)
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Try swapping adjacent comparators to enable further removals
    for iteration in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        improved = False
        for idx in range(len(network) - 1):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try swapping network[idx] and network[idx+1]
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            
            # Check if swapped network is still valid
            full_result = apply_network(all_inputs.copy(), candidate)
            if is_sorted(full_result):
                # Try to remove comparators from this swapped network
                temp_network = list(candidate)
                temp_changed = True
                while temp_changed:
                    temp_changed = False
                    for tidx in range(len(temp_network) - 1, -1, -1):
                        if time.time() - start_time > DEADLINE - 5:
                            break
                        t_candidate = temp_network[:tidx] + temp_network[tidx+1:]
                        if len(t_candidate) < len(temp_network):
                            subset_size = min(4096, 1 << N)
                            test_inputs = all_inputs[:subset_size].copy()
                            result = apply_network(test_inputs, t_candidate)
                            if is_sorted(result):
                                full_result = apply_network(all_inputs.copy(), t_candidate)
                                if is_sorted(full_result):
                                    temp_network = t_candidate
                                    temp_changed = True
                    if time.time() - start_time > DEADLINE - 5:
                        break
                
                if len(temp_network) < len(network):
                    network = temp_network
                    improved = True
                    best_network = list(network)
                    best_count = len(network)
        if not improved:
            break
    
    # Random restarts with perturbation if time remains
    if time.time() - start_time < DEADLINE - 10:
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Create a perturbed version of the best network
            perturbed = list(best_network)
            # Randomly swap a few comparators
            for _ in range(3):
                if len(perturbed) > 1:
                    i = rng.randint(0, len(perturbed) - 2)
                    perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            
            # Check if perturbed is valid
            full_result = apply_network(all_inputs.copy(), perturbed)
            if not is_sorted(full_result):
                continue
            
            # Try to remove comparators
            temp_network = list(perturbed)
            temp_changed = True
            while temp_changed:
                temp_changed = False
                for tidx in range(len(temp_network) - 1, -1, -1):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    t_candidate = temp_network[:tidx] + temp_network[tidx+1:]
                    if len(t_candidate) < len(temp_network):
                        subset_size = min(4096, 1 << N)
                        test_inputs = all_inputs[:subset_size].copy()
                        result = apply_network(test_inputs, t_candidate)
                        if is_sorted(result):
                            full_result = apply_network(all_inputs.copy(), t_candidate)
                            if is_sorted(full_result):
                                temp_network = t_candidate
                                temp_changed = True
                if time.time() - start_time > DEADLINE - 5:
                    break
            
            if len(temp_network) < best_count:
                best_network = list(temp_network)
                best_count = len(temp_network)
    
    return best_network
# EVOLVE-BLOCK-END
