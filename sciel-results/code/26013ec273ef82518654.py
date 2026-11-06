"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 80.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _batcher_odd_merge(n, m):
    """
    Returns comparators to merge two sorted lists of sizes n and m.
    The first list occupies indices 0..n-1, the second occupies indices n..n+m-1.
    Uses Batcher's Odd-Merge algorithm.
    """
    if n == 0 or m == 0:
        return []
    if n == 1 and m == 1:
        return [(0, 1)]
    
    n_odd = n // 2
    n_even = n - n_odd
    m_odd = m // 2
    m_even = m - m_odd
    
    # Merge A_odd (size n_odd) with B_even (size m_even)
    merge_odd = _batcher_odd_merge(n_odd, m_even)
    
    def map_odd(pos):
        if pos < n_odd:
            return 2 * pos + 1
        else:
            return n + 2 * (pos - n_odd)
    
    mapped_odd = [(map_odd(i), map_odd(j)) for (i, j) in merge_odd]
    
    # Merge A_even (size n_even) with B_odd (size m_odd)
    merge_even = _batcher_odd_merge(n_even, m_odd)
    
    def map_even(pos):
        if pos < n_even:
            return 2 * pos
        else:
            return n + 2 * (pos - n_even) + 1
    
    mapped_even = [(map_even(i), map_even(j)) for (i, j) in merge_even]
    
    # Final comparators
    total = n + m
    final_comps = [(2 * i, 2 * i + 1) for i in range(total // 2)]
    
    return mapped_odd + mapped_even + final_comps


def _batcher_merge_sort(n):
    """Returns a list of comparators for sorting n wires using Batcher's Merge Sort."""
    if n <= 1:
        return []
    
    k = n // 2
    left = _batcher_merge_sort(k)
    right = [(i + k, j + k) for (i, j) in _batcher_merge_sort(n - k)]
    merge = _batcher_odd_merge(k, n - k)
    
    return left + right + merge


def _odd_even_transposition_network(n):
    """Generate an odd-even transposition sorting network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


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
    
    # Start with Batcher's Odd-Merge Sort network
    initial_network = _batcher_merge_sort(N)
    
    # Verify initial network
    if not initial_network or not is_sorted(apply_network(all_inputs.copy(), initial_network)):
        # Fall back to odd-even transposition
        initial_network = _odd_even_transposition_network(N)
    
    best_network = list(initial_network)
    best_count = len(best_network)
    
    rng = random.Random(42)
    
    # Greedy removal: try to remove each comparator one by one
    network = list(best_network)
    
    for pass_num in range(8):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        changed = True
        while changed:
            changed = False
            # Randomize the order of removal attempts
            indices = list(range(len(network) - 1, -1, -1))
            if pass_num % 2 == 1:
                indices.reverse()
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > DEADLINE - 5:
                    break
                if idx >= len(network):
                    continue
                    
                # Try removing network[idx]
                candidate = network[:idx] + network[idx+1:]
                
                # Full check on all 262144 inputs
                full_result = apply_network(all_inputs.copy(), candidate)
                if is_sorted(full_result):
                    network = candidate
                    changed = True
                    if len(network) < best_count:
                        best_network = list(network)
                        best_count = len(network)
    
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
                            full_result = apply_network(all_inputs.copy(), t_candidate)
                            if is_sorted(full_result):
                                temp_network = t_candidate
                                temp_changed = True
                    if time.time() - start_time > DEADLINE - 5:
                        break
                
                if len(temp_network) < len(network):
                    network = temp_network
                    improved = True
                    if len(network) < best_count:
                        best_network = list(network)
                        best_count = len(network)
        if not improved:
            break
    
    # Random restarts with perturbation if time remains
    if time.time() - start_time < DEADLINE - 10:
        for restart in range(15):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Create a perturbed version of the best network
            perturbed = list(best_network)
            # Randomly swap a few comparators
            for _ in range(rng.randint(1, 5)):
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
