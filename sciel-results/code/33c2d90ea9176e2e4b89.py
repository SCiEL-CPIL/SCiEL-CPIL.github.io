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
    """
    A_mask = A_mask_init
    current_score = count_3aps_mask(A_mask)
    
    # Precompute list of elements in A for faster iteration
    def get_elements(mask):
        return [i for i in range(1, N + 1) if (mask >> (i - 1)) & 1]
    
    while time.time() < deadline:
        improved = False
        elems = get_elements(A_mask)
        
        # Try Additions
        best_add_x = -1
        best_add_delta = 0
        
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                continue
            if is_4ap_free_addition_mask(A_mask, x):
                delta = delta_3aps_for_addition_mask(A_mask, x)
                if delta > best_add_delta:
                    best_add_delta = delta
                    best_add_x = x
        
        if best_add_x != -1:
            A_mask |= elem_to_bit(best_add_x)
            current_score += best_add_delta
            improved = True
            continue
        
        # Try Swaps - only check promising swaps
        best_swap = None
        best_swap_score = current_score
        
        for x in elems:
            if time.time() > deadline:
                break
            
            # Calculate loss from removing x
            loss = delta_3aps_for_addition_mask(A_mask, x)
            
            # Temporarily remove x
            A_temp_mask = A_mask & ~elem_to_bit(x)
            
            # Only try adding elements that could improve
            for y in range(1, N + 1):
                if time.time() > deadline:
                    break
                if y == x:
                    continue
                if (A_temp_mask >> (y - 1)) & 1:
                    continue
                
                if not is_4ap_free_addition_mask(A_temp_mask, y):
                    continue
                
                gain = delta_3aps_for_addition_mask(A_temp_mask, y)
                
                new_score = current_score - loss + gain
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = (x, y)
            
        if best_swap:
            x, y = best_swap
            A_mask = (A_mask & ~elem_to_bit(x)) | elem_to_bit(y)
            current_score = best_swap_score
            improved = True
            continue
        
        if not improved:
            break
            
    return A_mask

def polish_mask(A_mask, deadline):
    """
    Final polish pass: try all single-element removals followed by best re-addition.
    This can escape local optima that swaps cannot.
    """
    current_score = count_3aps_mask(A_mask)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try removing each element and see if we can add back a better one
        for x in range(1, N + 1):
            if time.time() > deadline - 5:
                break
            if not ((A_mask >> (x - 1)) & 1):
                continue
            
            A_temp = A_mask & ~elem_to_bit(x)
            
            # Find best element to add to A_temp
            best_y = -1
            best_gain = 0
            for y in range(1, N + 1):
                if y == x:
                    continue
                if (A_temp >> (y - 1)) & 1:
                    continue
                if is_4ap_free_addition_mask(A_temp, y):
                    gain = delta_3aps_for_addition_mask(A_temp, y)
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
            
            if best_y != -1:
                # Check if this improves score
                # Loss from removing x
                loss = delta_3aps_for_addition_mask(A_mask, x)
                new_score = current_score - loss + best_gain
                if new_score > current_score:
                    A_mask = A_temp | elem_to_bit(best_y)
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            break
    
    return A_mask

def mask_to_list(A_mask):
    """Convert bitmask to sorted list of elements."""
    return [i for i in range(1, N + 1) if (A_mask >> (i - 1)) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s budget, 5s safety margin
    
    best_mask = 0
    best_score = -1
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial_mask(0))
    
    # Candidates 1-15: Randomized Greedies
    for i in range(1, 16):
        if time.time() > deadline - 10:
            break
        candidates.append(greedy_initial_mask(i))
        
    # Evaluate and optimize each candidate
    for idx, init_mask in enumerate(candidates):
        if time.time() > deadline - 15:
            break
            
        # Run local search with remaining time minus polish budget
        ls_deadline = deadline - 10
        optimized_mask = local_search_mask(init_mask, ls_deadline)
        
        # Polish pass
        polished_mask = polish_mask(optimized_mask, deadline)
        
        opt_score = count_3aps_mask(polished_mask)
        
        if opt_score > best_score:
            best_score = opt_score
            best_mask = polished_mask
            
    if best_mask == 0:
        # Fallback
        best_mask = candidates[0]
        
    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END
