import numpy as np
import time

N = 16

# Precompute sorted values for all 2^N inputs
# Input is a bitmask. Sorted output has 0s at lower indices, 1s at higher indices.
_masks = np.arange(1 << N, dtype=np.uint32)
_bit_counts = np.array([bin(i).count('1') for i in range(1 << N)], dtype=np.uint8)
# sorted_val = ((1 << k) - 1) << (N - k)
# Note: 1 << k can overflow uint32 if k is large? No, k <= 16.
# 1 << 16 is 65536, fits in uint32.
_sorted_vals = ((1 << _bit_counts) - 1) << (N - _bit_counts)
_sorted_vals = _sorted_vals.astype(np.uint32)

def _is_valid(net):
    """
    Check if the network sorts all inputs using the Zero-One Principle.
    Uses NumPy for vectorized simulation.
    """
    if not net:
        return N <= 1
    x = _masks.copy()
    for i, j in net:
        bi = (x >> i) & 1
        bj = (x >> j) & 1
        swap_mask = bi > bj
        if np.any(swap_mask):
            x = np.where(swap_mask, x ^ (1 << i) | (1 << j), x)
    return np.all(x == _sorted_vals)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit
    
    # Initial network: Odd-Even Transposition (120 comparators)
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    
    while time.time() < deadline:
        improved = False
        
        # 1. Greedy Pruning
        for k in range(len(net)):
            if time.time() > deadline:
                break
            candidate = net[:k] + net[k+1:]
            if _is_valid(candidate):
                net = candidate
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Random Walk on Equivalent Networks (Independent Swaps)
        # Find indices of adjacent independent comparators
        swap_indices = []
        for k in range(len(net) - 1):
            i1, j1 = net[k]
            i2, j2 = net[k+1]
            # Independent if wires are disjoint
            if i1 != i2 and i1 != j2 and j1 != i2 and j1 != j2:
                swap_indices.append(k)
                
        if swap_indices:
            # Pick a random swap
            k = swap_indices[np.random.randint(len(swap_indices))]
            net[k], net[k+1] = net[k+1], net[k]
            # Continue loop to try pruning again
        else:
            # No independent adjacent swaps available.
            # Try a random move (non-adjacent or dependent)
            # This requires verification.
            if len(net) > 0:
                idx = np.random.randint(len(net))
                comp = net[idx]
                rest = net[:idx] + net[idx+1:]
                new_idx = np.random.randint(len(rest) + 1)
                candidate = rest[:new_idx] + [comp] + rest[new_idx:]
                if _is_valid(candidate):
                    net = candidate
                    # Continue loop
                else:
                    # If we can't find a valid move, we are stuck.
                    # Break to return current best.
                    break
            else:
                break

    return net
