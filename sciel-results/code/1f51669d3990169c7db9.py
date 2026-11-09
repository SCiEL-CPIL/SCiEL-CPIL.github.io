import time
import numpy as np
import random

N = 17
TOTAL_MASKS = 1 << N  # 131072
DEADLINE_FRACTION = 0.95  # Safety margin

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_network(net, masks):
    """
    Returns the number of unsorted masks.
    0 means valid.
    """
    if len(net) == 0:
        # A mask is sorted if it is of the form 0...01...1
        # i.e., bit k is 1 for k < count_ones, 0 otherwise?
        # No, min to i, max to j. 
        # Input 01 (1). i=0, j=1. a=1, b=0. swap=1. x = 01 & ~1 | 1<<1 = 10 (2).
        # So 1s move to higher indices.
        # Sorted mask: 0...0 1...1.
        # Values: 2^k - 1 for k=0..N.
        # e.g. N=2. Sorted: 00(0), 01(1), 10(2), 11(3)? 
        # Wait. 
        # 00 -> 00. (0 ones). 2^0-1=0.
        # 01 -> 10? No.
        # Let's re-verify the sorting direction.
        # Comparator (i, j) with i < j.
        # min(x_i, x_j) goes to i.
        # max(x_i, x_j) goes to j.
        # So 0s go to lower indices, 1s go to higher indices.
        # Sorted output: 0...0 1...1.
        # Number of 1s is preserved.
        # If there are k ones, they are at indices N-k, ..., N-1.
        # Value = sum_{i=N-k}^{N-1} 2^i = 2^N - 2^{N-k} = 2^{N-k}(2^k - 1).
        # Let's check N=2.
        # k=0: 00 (0).
        # k=1: 01 (1). (One 1 at index 1).
        # k=2: 11 (3). (Two 1s at indices 0,1).
        # So sorted values are: 0, 1, 3.
        # My previous logic `2^k - 1` gave 0, 1, 3, 7... for N=2 it would be 0, 1, 3. 
        # Wait, 2^2-1 = 3. 2^1-1=1. 2^0-1=0.
        # So yes, the set of sorted values is { 2^k - 1 | k in 0..N }.
        
        sorted_vals = set([ (1 << k) - 1 for k in range(N+1) ])
        # Count masks NOT in this set
        return int(np.sum(~np.isin(masks, list(sorted_vals))))

    x = masks.astype(np.int32).copy()
    for i, j in net:
        a = (x >> i) & 1
        b = (x >> j) & 1
        swap = (a & ~b)
        x = (x & ~(swap << i)) | (swap << j)
    
    return int(np.sum(x != masks))

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    masks = np.arange(TOTAL_MASKS, dtype=np.int32)
    
    # Initialize with seed
    current_net = _build_seed()
    current_cost = _verify_network(current_net, masks)
    
    best_net = current_net[:]
    best_cost = current_cost
    
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    alpha = 0.98  # Cooling rate
    
    T = T_start
    steps = 0
    max_steps = 5000 # Safety cap, but time is the real limit
    
    # Precompute bit operations for speed? 
    # The bottleneck is _verify_network.
    # We can optimize _verify_network by using a more efficient representation?
    # N=17 fits in int32. 
    # Vectorized numpy is already quite fast.
    
    # To speed up verification, we can early exit if we detect obvious errors?
    # No, we need the count for cost.
    # But if cost > 0, we just need to know it's invalid.
    # Actually, for SA, we need the cost value to calculate delta.
    
    # Optimization: 
    # Instead of running the whole network on all 131k masks every time,
    # we can maintain a "dirty" set of masks?
    # No, that's complex.
    
    # Let's try to make the verification as fast as possible.
    # The current implementation is ~50-100ms.
    # 55s / 0.1s = 550 steps.
    # This is too few for SA to escape deep local minima.
    
    # We need a faster verification or a smarter search.
    
    # Alternative: Use a smaller subset of masks for the "cheap" cost function, 
    # and only verify on the full set when a better "cheap" cost is found?
    # Risk: The cheap cost might not correlate well with the full cost.
    # But for sorting networks, random masks are good indicators.
    
    # Let's use a subset of 4096 masks for the SA loop.
    # If a move improves the cost on the subset, we verify on the full set.
    # If it degrades the full set cost, we accept with probability exp(-delta/T).
    
    subset_size = 4096
    # Use a fixed random subset for consistency
    rng = np.random.RandomState(42)
    subset_indices = rng.choice(TOTAL_MASKS, size=subset_size, replace=False)
    subset_masks = masks[subset_indices]
    
    def _verify_subset(net):
        if len(net) == 0:
            sorted_vals = set([ (1 << k) - 1 for k in range(N+1) ])
            return int(np.sum(~np.isin(subset_masks, list(sorted_vals))))
        
        x = subset_masks.astype(np.int32).copy()
        for i, j in net:
            a = (x >> i) & 1
            b = (x >> j) & 1
            swap = (a & ~b)
            x = (x & ~(swap << i)) | (swap << j)
        return int(np.sum(x != subset_masks))

    # Initial state
    current_net = _build_seed()
    current_subset_cost = _verify_subset(current_net)
    current_full_cost = _verify_network(current_net, masks)
    
    best_net = current_net[:]
    best_full_cost = current_full_cost
    best_subset_cost = current_subset_cost
    
    T = T_start
    
    while time.time() < deadline and T > T_end:
        steps += 1
        
        # Generate a neighbor
        move_type = random.randint(0, 3)
        new_net = current_net[:]
        
        if move_type == 0: # Remove
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
        elif move_type == 1: # Add
            comp = random.choice(all_comps)
            idx = random.randint(0, len(new_net))
            new_net.insert(idx, comp)
        elif move_type == 2: # Swap
            if len(new_net) >= 2:
                i1 = random.randint(0, len(new_net) - 1)
                i2 = random.randint(0, len(new_net) - 1)
                if i1 != i2:
                    new_net[i1], new_net[i2] = new_net[i2], new_net[i1]
        else: # Move (Reorder)
            if len(new_net) >= 2:
                i1 = random.randint(0, len(new_net) - 1)
                i2 = random.randint(0, len(new_net) - 1)
                if i1 != i2:
                    elem = new_net.pop(i1)
                    new_net.insert(i2, elem)
        
        # Evaluate new state
        new_subset_cost = _verify_subset(new_net)
        
        # Calculate delta in subset cost
        delta_subset = new_subset_cost - current_subset_cost
        
        # Acceptance criterion
        if delta_subset <= 0:
            # Improvement or neutral on subset
            # Verify on full set to update best
            new_full_cost = _verify_network(new_net, masks)
            
            # Update current state
            current_net = new_net
            current_subset_cost = new_subset_cost
            current_full_cost = new_full_cost
            
            # Update best
            if new_full_cost < best_full_cost:
                best_full_cost = new_full_cost
                best_net = new_net[:]
                best_subset_cost = new_subset_cost
                
        else:
            # Degradation on subset
            # Accept with probability exp(-delta / T)
            # Note: delta_subset is in units of "unsorted masks in subset".
            # We need to scale this to a "temperature" compatible scale.
            # If subset size is 4096, max delta is 4096.
            # T is ~50. 
            # If delta is 10, exp(-10/50) = exp(-0.2) ~ 0.8. Good.
            # If delta is 100, exp(-100/50) = exp(-2) ~ 0.13.
            # This seems reasonable.
            
            if random.random() < np.exp(-delta_subset / T):
                # Accept move, but we don't necessarily need to verify full set immediately
                # unless it looks promising? 
                # For simplicity and correctness of the "current" state tracking, 
                # we should track the full cost if we accept.
                # But verifying full set is expensive.
                # We can lazily verify?
                # If we accept a bad move, we might wander.
                # Let's verify full set only if the subset cost is close to best_subset_cost?
                # No, that's complex.
                # Let's just accept and update current state.
                # We will verify full set only when we find a new subset best?
                # No, that breaks the "best_full_cost" tracking.
                
                # Compromise: 
                # We track current_full_cost lazily.
                # We only verify full set if:
                # 1. It's a new subset best.
                # 2. We have time to spare and want to check if we found a global best?
                
                # For now, let's just update current state without full verification.
                # This makes the search faster.
                current_net = new_net
                current_subset_cost = new_subset_cost
                # current_full_cost is now unknown/stale.
                # We mark it as "stale".
                # We will re-verify if we find a new subset best.
                
                # Actually, if we accept a move that increases subset cost, 
                # it's very unlikely to decrease full cost.
                # So we don't update best_full_cost.
                
                # But we need to know current_full_cost for future comparisons?
                # No, SA uses the current state's cost for delta calculation.
                # We are using subset cost for delta.
                # So we don't strictly need current_full_cost for the SA step.
                # We only need it to update best_net.
                
                # So:
                # - If subset cost improves: Verify full, update best if full improves.
                # - If subset cost worsens but accepted: Just update current state.
                
                pass # No full verification here
        
        T *= alpha
        
        # Periodic full verification of current state to ensure we aren't drifting too far?
        # Or just rely on the "improvement" triggers.
        
        # If we have a lot of time, we can do a "local search" refinement on the best found so far?
        # No, let's stick to SA.
        
        # Check if we should break early if T is very low and no improvements for a long time?
        # Not necessary, the loop ends at T_end or deadline.
        
    # Final check: ensure best_net is valid (it should be, as we only update best after full verification)
    # But just in case, return best_net.
    
    return best_net
