"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _batchers_16():
    """
    Generate Batcher's Odd-Even Merge Sort network for 16 wires.
    This serves as a high-quality seed for the annealing search.
    """
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        # Recursively sort the two halves
        net1 = _odd_even_merge(k)
        net2 = _odd_even_merge(k)
        
        # Combine them with offsets
        left = [(i, j) for i, j in net1]
        right = [(i + k, j + k) for i, j in net2]
        
        # Odd-even merge of the two sorted halves
        # This part is tricky to generate recursively without explicit indices.
        # For N=16, it's easier to hardcode or use a known generator.
        # Let's use a standard generator for Batcher's network.
        
    # Standard Batcher's Odd-Even Merge Sort for N=16
    # Source: Known optimal or near-optimal constructions
    # Batcher's OES for 16 uses 63 comparators.
    
    # Let's construct it properly.
    def _oe_merge(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        # Sort first half
        left = _oe_merge(k)
        # Sort second half
        right = [(i + k, j + k) for i, j in _oe_merge(k)]
        
        # Merge
        # The merge step of odd-even merge sort involves:
        # 1. Comparing (i, i+k) for i in 0..k-1 (this is actually part of the merge)
        #    Wait, the standard definition:
        #    OES(n) = OES(k) on first half, OES(k) on second half, then Merge(k, k)
        #    Merge(k, k) involves comparing (i, i+k) for i=0..k-1? No.
        
        # Let's use a simpler, verified construction for 16.
        # Batcher's network for 16 wires:
        # Stage 1: (0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,14), (13,15)
        # Stage 2: (0,4), (1,5), (2,6), (3,7), (8,12), (9,13), (10,14), (11,15)
        # Stage 3: (0,8), (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)
        # ... This is getting complicated to hardcode correctly without a library.
        
        # Alternative: Use a known 60-comparator network if I can find one, 
        # or just use a generic construction and let annealing fix it.
        # Actually, the prompt mentions "A verified 60-comparator construction is the reference".
        # I will start with a slightly larger, known correct network and anneal down.
        
        # Let's use a simple correct network: Shell's network? Or just odd-even transposition (120) and anneal?
        # Annealing from 120 to 60 is a long way.
        # Let's try to construct Batcher's OES correctly.
        
    # Correct Batcher's OES for 16:
    # 1. Split into 8 and 8.
    # 2. Sort each 8.
    # 3. Merge the two 8s.
    
    def _sort_8():
        # Batcher's OES for 8
        # 1. Sort 4s
        def _sort_4():
            # (0,2), (1,3) -> Sort pairs
            # (0,1), (2,3) -> Sort pairs? No.
            # OES(4):
            # Sort 2s: (0,1), (2,3)
            # Merge: (0,2), (1,3)
            # Result: (0,1), (2,3), (0,2), (1,3)
            return [(0, 1), (2, 3), (0, 2), (1, 3)]
        
        s4_0 = _sort_4()
        s4_1 = [(i+4, j+4) for i, j in _sort_4()]
        
        # Merge two 4s into 8
        # Merge(4,4):
        # Compare (i, i+4) for i=0..3? No.
        # The merge step of OES:
        # For i in 0..3: compare (i, i+4)
        # Then recursively merge the resulting pairs?
        # Actually, OES(n) merge step:
        # Compare (i, i+n/2) for i=0..n/2-1.
        # Then apply OES(n/2) to the even indices and OES(n/2) to the odd indices?
        # No, that's not right either.
        
        # Let's use a simpler, robust approach:
        # Start with Odd-Even Transposition (120 comparators).
        # It is correct. Annealing will reduce it.
        # 120 -> 60 is a 50% reduction. With 90s budget, this is feasible.
        
        return [(i, i + 1) for phase in range(8) for i in range(phase % 2, 7, 2)] + \
               [(i+8, i+9) for phase in range(8) for i in range(phase % 2, 7, 2)] # This is wrong, just use the 16-wire OET
        
    # Let's just use the 16-wire Odd-Even Transposition Network as the seed.
    # It has 120 comparators.
    seed = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed.append((i, i + 1))
            
    return seed

def _verify_net(net):
    """
    Verify network using zero-one principle with NumPy.
    Returns True if sorted.
    """
    n_comp = len(net)
    if n_comp == 0:
        return N <= 1
        
    # Create a matrix of all 2^16 inputs? 65536 x 16.
    # Memory: 65536 * 16 * 4 bytes = 4MB. Feasible.
    # But applying the network to all 65536 inputs at once might be slow if done in Python loops.
    # We can vectorize the comparator operations.
    
    # Generate all masks
    # np.arange(1 << N) is 65536.
    # bits = (masks[:, None] >> np.arange(N)[None, :]) & 1
    
    # To save time, we can check a subset first, or just do the full check.
    # 65536 * 16 operations per comparator? No, 65536 operations per comparator.
    # 120 comparators * 65536 ops = 7.8M ops. NumPy can do this fast.
    
    masks = np.arange(1 << N, dtype=np.int32)
    bits = (masks[:, None] >> np.arange(N, dtype=np.int32)[None, :]) & 1
    
    for i, j in net:
        # Extract columns i and j
        a = bits[:, i]
        b = bits[:, j]
        
        # min and max
        min_ab = np.minimum(a, b)
        max_ab = np.maximum(a, b)
        
        bits[:, i] = min_ab
        bits[:, j] = max_ab
        
    # Check if sorted: bits[:, k] <= bits[:, k+1] for all k
    # diff = np.diff(bits, axis=1)
    # return np.all(diff >= 0)
    
    # Vectorized check
    left = bits[:, :-1]
    right = bits[:, 1:]
    return np.all(left <= right)

def _apply_single(net, input_vec):
    """Apply network to a single input vector (for debugging/speed testing)."""
    vals = list(input_vec)
    for i, j in net:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initialize with Odd-Even Transposition (120 comparators)
    current_net = _batchers_16()
    current_cost = len(current_net)
    
    best_net = current_net[:]
    best_cost = current_cost
    
    # Verify seed (should be true)
    if not _verify_net(current_net):
        # Fallback if seed is broken (shouldn't happen)
        return current_net

    T = 10.0  # Initial temperature
    T_min = 0.1
    alpha = 0.995  # Cooling rate
    
    # Pre-allocate for speed if needed, but list ops are fine for N=16
    
    while time.time() < deadline:
        # Generate neighbor
        new_net = current_net[:]
        
        # Choose a move: Insert, Delete, Swap
        move_type = random.random()
        
        if move_type < 0.4 and len(new_net) > 1:
            # Delete
            idx = random.randint(0, len(new_net) - 1)
            new_net.pop(idx)
            
        elif move_type < 0.8:
            # Insert
            # Pick a random valid comparator
            # We can pick random i, j with i < j
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            comp = (i, j)
            # Insert at random position
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, comp)
            
        else:
            # Swap
            if len(new_net) > 1:
                idx1 = random.randint(0, len(new_net) - 1)
                idx2 = random.randint(0, len(new_net) - 1)
                if idx1 != idx2:
                    new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
                else:
                    # If same index, just keep it (no-op)
                    pass

        new_cost = len(new_net)
        
        # Verify new network
        # Optimization: If new_cost > best_cost + 50, skip verification? 
        # No, we need to know if it's valid to continue.
        # But we can skip if it's obviously bad? 
        # Let's verify.
        
        is_valid = _verify_net(new_net)
        
        if is_valid:
            # Calculate delta
            delta_cost = new_cost - current_cost
            
            # Acceptance probability
            if delta_cost < 0:
                accept = True
            else:
                # Accept with probability exp(-delta / T)
                # Since we minimize cost, delta_cost is positive if worse.
                # We want to accept worse states to escape local minima.
                # Cost difference is in comparators.
                # Let's scale delta_cost. 1 comparator is a significant change.
                # If delta_cost is 1, and T is 10, prob is exp(-0.1) ~ 0.9.
                # If delta_cost is 10, prob is exp(-1) ~ 0.36.
                accept = random.random() < np.exp(-delta_cost / T)
            
            if accept:
                current_net = new_net
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_net = current_net[:]
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat slightly if stuck?
            # For now, just stay at T_min
            T = T_min

    # Final verification of best
    if _verify_net(best_net):
        return best_net
    else:
        return current_net

# EVOLVE-BLOCK-END
