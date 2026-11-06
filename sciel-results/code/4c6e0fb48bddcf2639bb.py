import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Initial odd-even transposition network: 120 comparators
    initial_net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)
    
    # Fast rejection subset
    fast_indices = np.random.choice(2**N, size=1000, replace=False).astype(np.uint32)
    fast_inputs = inputs[fast_indices]
    fast_targets = targets[fast_indices]

    def verify_full(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    def verify_fast(net):
        masks = fast_inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, fast_targets)
        
    def verify(net):
        if verify_fast(net):
            return verify_full(net)
        return False

    # Verify initial
    if not verify(initial_net):
        return initial_net

    current_net = initial_net
    best_net = initial_net
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    
    random.seed(42)
    
    # Strategy: Local search with moves: Delete 1, Swap 2, Delete 2
    # We prioritize moves that reduce length.
    
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 1:
            break
        
        # Try to find an improving move
        improved = False
        
        # Move 1: Try to delete a single comparator
        # Sample a subset of indices to keep time bounded
        # Increase sample size as we get smaller? Or keep fixed?
        sample_size = min(50, net_len)
        indices = random.sample(range(net_len), sample_size)
        
        for idx in indices:
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                improved = True
                break
        
        if improved:
            continue
            
        # Move 2: Try to swap two comparators (reordering)
        # This helps if the network is correct but not minimal, and reordering allows deletion later
        if net_len > 2:
            i1 = random.randint(0, net_len - 1)
            i2 = random.randint(0, net_len - 1)
            if i1 != i2:
                candidate = current_net.copy()
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                if verify(candidate):
                    current_net = candidate
                    # Accept swap to potentially enable future deletions
                    pass
        
        # Move 3: Try to delete two comparators
        # This is more expensive but escapes local minima of single-deletion
        if net_len > 2:
            # Sample pairs
            i1 = random.randint(0, net_len - 1)
            i2 = random.randint(0, net_len - 1)
            if i1 != i2:
                if i1 > i2:
                    i1, i2 = i2, i1
                # Remove i2 first, then i1-1 (since index shifts)
                # Or construct new list
                candidate = current_net[:i1] + current_net[i1+1:i2] + current_net[i2+1:]
                if verify(candidate):
                    current_net = candidate
                    if len(current_net) < best_len:
                        best_net = current_net
                        best_len = len(current_net)
                    improved = True

        if not improved:
             # Stuck. Try a random perturbation: swap 3 random items?
             # Or just continue. The random nature of sampling acts as perturbation.
             pass

    return best_net
# EVOLVE-BLOCK-END
