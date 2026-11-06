"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 16 wires.
    Strategy: Start from Batcher's odd-even mergesort (60 comparators),
    then use randomized local search (removal + swap) to try to find
    a smaller valid network.
    """
    
    def batcher_oom(n):
        """Batcher's odd-even mergesort network for n=2^k wires."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        # Recursively sort halves
        left = batcher_oom(k)
        right = [(i + k, j + k) for (i, j) in left]
        
        # Odd-even merge
        def merge(a, b, size):
            """Merge two sorted sequences of 'size' elements starting at positions a and b."""
            if size == 1:
                return [(a, b)]
            half = size // 2
            result = []
            result.extend(merge(a, b, half))
            result.extend(merge(a + half, b + half, half))
            # Compare-exchange between the two halves
            for i in range(half):
                result.append((a + i, b + i))
            return result
        
        merged = merge(0, k, k)
        return left + right + merged
    
    # Generate Batcher's network for 16 wires
    batcher_net = batcher_oom(N)
    
    # Pre-compute all 2^16 inputs and their sorted targets
    inputs = np.arange(2**N, dtype=np.uint32)
    # Target for k ones: top k bits are 1, rest 0 (since min goes to lower wire)
    # Wait: compare-exchange writes min to wire i (lower index) and max to wire j (higher index).
    # So sorted output has 0s in lower wires and 1s in higher wires.
    # For k ones, wires 0..(N-k-1) are 0, wires (N-k)..(N-1) are 1.
    # That's: ((1 << k) - 1) << (N - k)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) for k in popcounts], dtype=np.uint32)
    
    def simulate(net):
        """Simulate the network on all 65536 inputs, return final state array."""
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            # Condition: wire i has 1, wire j has 0 -> swap
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return masks
    
    def verify(net):
        """Check if net sorts all inputs."""
        result = simulate(net)
        return np.array_equal(result, targets)
    
    # Verify Batcher's network
    if not verify(batcher_net):
        # Fallback: use odd-even transposition (should not happen)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    current_net = list(batcher_net)
    best_net = list(batcher_net)
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Phase 1: Aggressive removal search
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 1:
            break
        
        # Try to find removable comparators by testing random subsets
        found_removal = False
        max_attempts = max(10, net_len // 2)
        
        for _ in range(max_attempts):
            if time.time() - start_time > deadline - 5.0:
                break
            idx = random.randint(0, net_len - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                found_removal = True
                break
        
        if not found_removal:
            # Try swapping to escape local minimum
            if net_len > 2:
                i1 = random.randint(0, net_len - 1)
                i2 = random.randint(0, net_len - 1)
                if i1 != i2:
                    candidate = current_net.copy()
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                    if verify(candidate):
                        current_net = candidate
    
    # Phase 2: Polish - try systematic removals from the best network
    if time.time() - start_time < deadline - 2.0:
        polish_net = list(best_net)
        polish_len = len(polish_net)
        # Try removing each comparator in order, keep if valid
        for idx in range(polish_len - 1, -1, -1):
            if time.time() - start_time > deadline - 2.0:
                break
            candidate = polish_net[:idx] + polish_net[idx+1:]
            if verify(candidate):
                polish_net = candidate
                polish_len -= 1
        
        if len(polish_net) < best_len:
            best_net = polish_net
            best_len = len(best_net)
    
    return best_net
# EVOLVE-BLOCK-END
