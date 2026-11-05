import time
import random

# Precomputation
N = 256

# Bitmask helpers
# We will use Python integers as bitmasks.
# Bit i-1 represents element i.

def elem_to_bit(e):
    return 1 << (e - 1)

# Precompute 4-AP conflicts and 3-AP gains for each element
# four_conflicts[x] = list of bitmasks representing the OTHER 3 elements in 4-APs containing x
# three_gains[x] = list of bitmasks representing the OTHER 2 elements in 3-APs containing x

four_conflicts = [[] for _ in range(N + 1)]
three_gains = [[] for _ in range(N + 1)]

# Generate 4-APs
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        # For each element in the AP, the other 3 form a conflict mask
        mask_a = elem_to_bit(b) | elem_to_bit(c) | elem_to_bit(e)
        mask_b = elem_to_bit(a) | elem_to_bit(c) | elem_to_bit(e)
        mask_c = elem_to_bit(a) | elem_to_bit(b) | elem_to_bit(e)
        mask_e = elem_to_bit(a) | elem_to_bit(b) | elem_to_bit(c)
        
        four_conflicts[a].append(mask_a)
        four_conflicts[b].append(mask_b)
        four_conflicts[c].append(mask_c)
        four_conflicts[e].append(mask_e)

# Generate 3-APs
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        mask_a = elem_to_bit(b) | elem_to_bit(c)
        mask_b = elem_to_bit(a) | elem_to_bit(c)
        mask_c = elem_to_bit(a) | elem_to_bit(b)
        
        three_gains[a].append(mask_a)
        three_gains[b].append(mask_b)
        three_gains[c].append(mask_c)

def is_4ap_free_addition_mask(A_mask, x):
    """Check if adding x to A_mask creates a 4-AP."""
    for m in four_conflicts[x]:
        if (A_mask & m) == m:
            return False
    return True

def delta_3aps_for_addition_mask(A_mask, x):
    """Calculate how many new 3-APs are created by adding x to A_mask."""
    count = 0
    for m in three_gains[x]:
        if (A_mask & m) == m:
            count += 1
    return count

def count_3aps_mask(A_mask):
    """Count distinct 3-APs in set represented by A_mask."""
    # Each 3-AP is counted 3 times (once for each element).
    total = 0
    for i in range(1, N + 1):
        if (A_mask >> (i - 1)) & 1:
            total += delta_3aps_for_addition_mask(A_mask, i)
    return total // 3

def greedy_initial_mask(seed_val=0):
    """Generate a 4-AP free set using a greedy approach with bitmask."""
    rng = random.Random(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    if seed_val % 2 == 0:
        rng.shuffle(candidates)
    
    for x in candidates:
        if is_4ap_free_addition_mask(A_mask, x):
            A_mask |= elem_to_bit(x)
    return A_mask

def local_search_mask(A_mask_init, deadline):
    """
    Perform local search to maximize 3-APs while maintaining 4-AP freedom using bitmasks.
    Optimized with incremental delta and validity tracking.
    """
    A_mask = A_mask_init
    current_score = count_3aps_mask(A_mask)
    
    # Initialize delta[x] and can_add[x]
    # delta[x] = gain of adding x to A
    # can_add[x] = True if adding x to A is 4-AP free
    delta = [0] * (N + 1)
    can_add = [False] * (N + 1)
    
    for x in range(1, N + 1):
        if (A_mask >> (x - 1)) & 1:
            delta[x] = 0 # Not used for x in A
            can_add[x] = False
        else:
            delta[x] = delta_3aps_for_addition_mask(A_mask, x)
            can_add[x] = is_4ap_free_addition_mask(A_mask, x)

    # Helper to update delta and can_add when adding/removing an element
    # This is expensive if done naively for all x, so we only do it when A changes.
    # Actually, we can update incrementally.
    # If we add y to A:
    #   For any x not in A:
    #     delta[x] increases by (number of 3-APs containing x and y and some z in A)
    #     can_add[x] might become False if x and y complete a 4-AP with two other elements in A.
    
    # Given the complexity of incremental updates for all x, and N=256 is small,
    # recalculating delta/can_add for all x after each move is O(N * degree) ~ 256 * 20 ~ 5000 ops.
    # This is fast enough for Python if we do it carefully or just recompute the arrays.
    
    def recompute_deltas():
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                delta[x] = 0
                can_add[x] = False
            else:
                delta[x] = delta_3aps_for_addition_mask(A_mask, x)
                can_add[x] = is_4ap_free_addition_mask(A_mask, x)

    recompute_deltas()

    while time.time() < deadline:
        improved = False
        
        # 1. Try Additions
        best_add_x = -1
        best_add_delta = 0
        
        for x in range(1, N + 1):
            if can_add[x] and delta[x] > best_add_delta:
                best_add_delta = delta[x]
                best_add_x = x
        
        if best_add_x != -1:
            A_mask |= elem_to_bit(best_add_x)
            current_score += best_add_delta
            improved = True
            recompute_deltas()
            continue
        
        # 2. Try Swaps
        # Find best (x_out, y_in) such that x_out in A, y_in not in A
        # new_score = current_score - delta_in_A(x_out) + delta_after_removal(y_in)
        # Note: delta[x] computed above is gain of ADDING x.
        # Loss of removing x is the number of 3-APs containing x in A.
        # This is exactly delta_3aps_for_addition_mask(A_mask, x) if x was NOT in A?
        # No. delta_3aps_for_addition_mask(A, x) counts 3-APs formed by x and 2 elements in A.
        # If x is in A, the "loss" is the number of 3-APs in A that contain x.
        # This is the same value: number of pairs {u,v} in A\{x} such that {x,u,v} is a 3-AP.
        # So loss(x) = delta_3aps_for_addition_mask(A_mask, x).
        
        # Let's precompute loss for all x in A
        loss = [0] * (N + 1)
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                loss[x] = delta_3aps_for_addition_mask(A_mask, x)
        
        best_swap = None
        best_swap_score = current_score
        
        # We need to be careful: if we remove x, the gain for y changes.
        # gain(y | A \ {x}) != gain(y | A).
        # However, if we just pick the best x and best y independently, we might be wrong.
        # But usually, the best move is a local improvement.
        # Let's iterate over all pairs (x, y) to be safe, but optimize the inner loop.
        
        # Optimization: Pre-compute potential gains for y assuming A is unchanged.
        # gain_approx[y] = delta[y] (which is gain(y | A)).
        # If we remove x, gain(y | A\x) = gain(y | A) - (1 if {x, y, z} is a 3-AP with z in A\{x,y} else 0)... 
        # This is getting complicated. 
        # Since N=256, N^2 = 65536. 
        # For each pair, checking 4-AP freedom and calculating exact gain is expensive.
        # Let's stick to the previous logic but use the precomputed loss[x].
        
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((A_mask >> (x - 1)) & 1):
                continue
            
            A_temp_mask = A_mask & ~elem_to_bit(x)
            
            # Calculate gain for each y in A_temp_mask
            # We can reuse the logic from before but it was slow.
            # Let's just iterate y.
            
            for y in range(1, N + 1):
                if time.time() > deadline:
                    break
                if y == x:
                    continue
                if (A_temp_mask >> (y - 1)) & 1:
                    continue # y is already in A (and y != x)
                
                # Check 4-AP freedom for adding y to A_temp_mask
                # This is the bottleneck.
                # We can check: is y 4-AP free in A? If not, maybe it is in A\x?
                # If y is not 4-AP free in A, it means there exists a 4-AP containing y and 3 elements in A.
                # If one of those 3 is x, then removing x might make y valid.
                
                valid = True
                for m in four_conflicts[y]:
                    if (A_temp_mask & m) == m:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate gain
                gain = 0
                for m in three_gains[y]:
                    if (A_temp_mask & m) == m:
                        gain += 1
                
                new_score = current_score - loss[x] + gain
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = (x, y)
            
        if best_swap:
            x, y = best_swap
            A_mask = (A_mask & ~elem_to_bit(x)) | elem_to_bit(y)
            current_score = best_swap_score
            improved = True
            recompute_deltas()
            continue
        
        if not improved:
            break
            
    return A_mask

def mask_to_list(A_mask):
    """Convert bitmask to sorted list of elements."""
    return [i for i in range(1, N + 1) if (A_mask >> (i - 1)) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s budget, 10s safety margin
    
    best_mask = 0
    best_score = -1
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial_mask(0))
    
    # Candidates 1-10: Randomized Greedies
    for i in range(1, 11):
        if time.time() > deadline - 5:
            break
        candidates.append(greedy_initial_mask(i))
        
    # Evaluate and optimize each candidate
    for idx, init_mask in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Quick score check
        s = count_3aps_mask(init_mask)
        
        # Run local search
        optimized_mask = local_search_mask(init_mask, deadline)
        
        opt_score = count_3aps_mask(optimized_mask)
        
        if opt_score > best_score:
            best_score = opt_score
            best_mask = optimized_mask
            
    if best_mask == 0:
        # Fallback
        best_mask = candidates[0]
        
    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END
