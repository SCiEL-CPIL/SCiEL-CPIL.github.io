import time
import random
import sys

N = 256
DEADLINE_MARGIN = 10  # seconds before the 90s limit

# Bitmask helper
def bit(x):
    return 1 << (x - 1)

def in_mask(mask, x):
    return (mask >> (x - 1)) & 1

# Precomputation
# 1. 3-AP contributions
# For each x, we want to know which pairs (u, v) in S would form a 3-AP with x.
# Types:
# x is middle: (u, x, v) -> u + v = 2x
# x is left: (x, u, v) -> u = x+d, v = x+2d
# x is right: (u, v, x) -> u = x-2d, v = x-d

# We store for each x a list of "witness" pairs.
# If x is middle: witness is (u, v). We check if u in S and v in S.
# If x is left: witness is (u, v). We check if u in S and v in S.
# If x is right: witness is (u, v). We check if u in S and v in S.

# To speed up the check "u in S and v in S", we can precompute a 2D array? No, N=256, N^2=65536.
# We can precompute for each pair (u, v) a bitmask of which x's make a 3-AP with them?
# Actually, iterating through the small list of witnesses for x is faster.
# Max witnesses for x:
# Middle: ~128 pairs
# Left: ~64 pairs
# Right: ~64 pairs
# Total ~256 checks per add/remove. This is fast.

mid_witnesses = [[] for _ in range(N + 1)]
left_witnesses = [[] for _ in range(N + 1)]
right_witnesses = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # x is middle: a + c = 2x, a < x < c
    for a in range(1, x):
        c = 2 * x - a
        if c <= N and c > x:
            mid_witnesses[x].append((a, c))
    
    # x is left: x, b, c -> b = x+d, c = x+2d
    for d in range(1, (N - x) // 2 + 1):
        b = x + d
        c = x + 2 * d
        if c <= N:
            left_witnesses[x].append((b, c))
            
    # x is right: a, b, x -> b = x-d, a = x-2d
    for d in range(1, (x - 1) // 2 + 1):
        b = x - d
        a = x - 2 * d
        if a >= 1:
            right_witnesses[x].append((a, b))

# 2. 4-AP constraints
# For each x, store the other 3 elements of any 4-AP containing x.
four_ap_constraints = [[] for _ in range(N + 1)]

for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            # The 4-AP is (a, b, c, e)
            # If we add x, it's invalid if the other 3 are in S.
            for x in (a, b, c, e):
                others = [y for y in (a, b, c, e) if y != x]
                four_ap_constraints[x].append(others)

def count_3aps_from_mask(mask):
    """Calculate total 3-APs in the set represented by mask."""
    count = 0
    # To avoid double counting, we can sum over middle elements only?
    # No, a 3-AP (a,b,c) is uniquely identified by its middle b.
    # So we just iterate over all b in S and count pairs (a,c) in S with a+c=2b.
    for b in range(1, N + 1):
        if in_mask(mask, b):
            for a, c in mid_witnesses[b]:
                if in_mask(mask, a) and in_mask(mask, c):
                    count += 1
    return count

def is_valid_mask(mask):
    """Check if mask contains no 4-AP."""
    # For each x in S, check if any 4-AP constraint is satisfied.
    # This is O(|S| * K) where K is avg constraints.
    # Better: iterate over all 4-APs and check if all 4 are in mask.
    # Number of 4-APs is ~ O(N^2) = 65536.
    # Checking 65536 masks is fast.
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                if in_mask(mask, a) and in_mask(mask, b) and in_mask(mask, c) and in_mask(mask, e):
                    return False
    return True

def get_delta_3aps(x, mask):
    """Calculate change in 3-AP count if x is added to mask."""
    delta = 0
    # x as middle
    for a, c in mid_witnesses[x]:
        if in_mask(mask, a) and in_mask(mask, c):
            delta += 1
    # x as left
    for b, c in left_witnesses[x]:
        if in_mask(mask, b) and in_mask(mask, c):
            delta += 1
    # x as right
    for a, b in right_witnesses[x]:
        if in_mask(mask, a) and in_mask(mask, b):
            delta += 1
    return delta

def can_add(x, mask):
    """Check if adding x to mask creates a 4-AP."""
    for others in four_ap_constraints[x]:
        if all(in_mask(mask, y) for y in others):
            return False
    return True

def can_remove(x, mask):
    """Removing x never creates a 4-AP, so always valid."""
    return True

def local_search(start_mask, time_limit=5.0):
    """
    Local search using bitmask.
    Moves: Add, Remove, Swap.
    """
    start_time = time.time()
    current_mask = start_mask
    current_score = count_3aps_from_mask(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try Adding
        # Random order to avoid bias
        random.shuffle(all_elements)
        for x in all_elements:
            if in_mask(current_mask, x):
                continue
            if not can_add(x, current_mask):
                continue
            delta = get_delta_3aps(x, current_mask)
            new_score = current_score + delta
            if new_score > best_score:
                current_mask |= bit(x)
                current_score = new_score
                best_mask = current_mask
                best_score = new_score
                improved = True
                break
        
        if improved:
            continue
            
        # Try Removing
        # Only useful if we are stuck and want to change structure? 
        # Removing decreases score, so we only do it if it enables a better add/swap?
        # In standard hill climbing for maximization, we don't remove unless it's part of a swap.
        # But sometimes removing a "bad" element allows adding multiple "good" ones.
        # Let's stick to Swaps for structure change.
        
        # Try Swaps
        # Pick a random x not in mask, random y in mask
        in_mask_elems = [i for i in range(1, N+1) if in_mask(current_mask, i)]
        not_in_mask_elems = [i for i in range(1, N+1) if not in_mask(current_mask, i)]
        
        if not in_mask_elems or not not_in_mask_elems:
            break
            
        # Try a few random swaps
        for _ in range(50):
            x = random.choice(not_in_mask_elems)
            y = random.choice(in_mask_elems)
            
            # Check validity of swap: 
            # 1. Removing y is always valid regarding 4-APs.
            # 2. Adding x to (mask - y) must be valid.
            temp_mask = current_mask & ~bit(y)
            if not can_add(x, temp_mask):
                continue
            
            # Calculate score change
            # Score(mask) = Score(mask - y) + Delta_add(x, mask - y)
            # We need Score(mask - y).
            # Delta_remove(y, mask) = - get_delta_3aps(y, mask - y)? No.
            # Let's just calculate new score directly or incrementally.
            # Incremental is faster.
            # Delta_remove(y, mask) = - (number of 3-APs involving y that are fully in mask)
            # This is exactly get_delta_3aps(y, mask - y) ... wait.
            # get_delta_3aps(y, M) gives # APs added if y is added to M.
            # If we remove y from S, the loss is the # APs that have y and other 2 in S\{y}.
            # This is get_delta_3aps(y, S - {y}).
            
            mask_without_y = current_mask & ~bit(y)
            loss = get_delta_3aps(y, mask_without_y)
            gain = get_delta_3aps(x, mask_without_y)
            
            new_score = current_score - loss + gain
            
            if new_score > best_score:
                current_mask = mask_without_y | bit(x)
                current_score = new_score
                best_mask = current_mask
                best_score = new_score
                improved = True
                # Update lists for next iteration? 
                # Just break and re-calc in next loop to be safe/fast enough
                break
                
        if not improved:
            # If no improvement in Add or Swap, try a random perturbation?
            # Or break if stuck for a long time?
            # For now, if no improvement, we might be at a local optimum.
            # We can try a "kick": remove a random element and add a random one, even if score drops?
            # Simulated Annealing style.
            # Let's try a few random kicks to escape.
            kicked = False
            for _ in range(5):
                if not in_mask_elems or not not_in_mask_elems:
                    break
                x = random.choice(not_in_mask_elems)
                y = random.choice(in_mask_elems)
                temp_mask = current_mask & ~bit(y)
                if not can_add(x, temp_mask):
                    continue
                # Accept with some probability? 
                # For simplicity, let's just accept if it doesn't drop too much? 
                # Or just break out of local search and restart.
                # Given the time budget, restarting is safer.
                break
            
            if not improved and not kicked:
                # Stuck
                break
                
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    best_mask = 0
    best_score = -1
    
    # Strategy 1: Greedy construction
    # Add elements in order of "potential" (e.g., those that form many 3-APs)
    # Or just random order.
    # Let's try a few random greedy runs.
    
    for trial in range(5):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        random.seed(trial)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        mask = 0
        score = 0
        for x in order:
            if in_mask(mask, x):
                continue
            if can_add(x, mask):
                mask |= bit(x)
                score += get_delta_3aps(x, mask & ~bit(x)) # Careful: delta is based on mask BEFORE adding
                # Actually get_delta_3aps(x, mask) calculates APs formed with x and elements in mask.
                # So score += get_delta_3aps(x, mask) is correct if mask doesn't contain x.
        
        # Local search from this greedy result
        ls_mask, ls_score = local_search(mask, time_limit=min(5.0, DEADLINE_MARGIN - (time.time() - start_time)))
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
    # Strategy 2: Random Start + Local Search
    for trial in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        random.seed(100 + trial)
        # Random subset of size ~ N/2
        size = random.randint(100, 150)
        elems = random.sample(range(1, N + 1), size)
        mask = 0
        for x in elems:
            mask |= bit(x)
        
        # This random mask likely has 4-APs. 
        # We need to clean it up or start local search from a valid mask.
        # local_search assumes start_mask is valid? 
        # My local_search doesn't check validity of start_mask.
        # It assumes moves maintain validity.
        # If start_mask is invalid, the score is meaningless.
        # So we must start from a valid mask.
        
        # Let's just use the greedy mask from Strategy 1 if available, or a new greedy one.
        # Or, we can run a "repair" step: remove elements causing 4-APs.
        
        # Let's stick to valid starts.
        # Run a quick greedy to get a valid mask.
        order = list(range(1, N + 1))
        random.shuffle(order)
        valid_mask = 0
        for x in order:
            if can_add(x, valid_mask):
                valid_mask |= bit(x)
        
        ls_mask, ls_score = local_search(valid_mask, time_limit=min(2.0, DEADLINE_MARGIN - (time.time() - start_time)))
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
    # Final check
    if best_mask == 0:
        # Fallback
        best_mask = bit(1)
        
    # Convert mask to list
    result = []
    for x in range(1, N + 1):
        if in_mask(best_mask, x):
            result.append(x)
            
    return result

def validate(result):
    mask = 0
    for x in result:
        mask |= bit(x)
    return is_valid_mask(mask)

if __name__ == "__main__":
    result = construct()
    score = count_3aps_from_mask(0) # Just for structure, we don't use this in construct
    # Recalc score for printing
    m = 0
    for x in result:
        m |= bit(x)
    real_score = count_3aps_from_mask(m)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {real_score}")
    print(f"Valid (no 4-AP): {valid}")
