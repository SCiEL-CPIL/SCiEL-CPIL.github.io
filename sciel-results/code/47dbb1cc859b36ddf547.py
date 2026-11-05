import time
import random

N = 256

# Precomputation
# We represent the set as a bitmask (int).
# Bit i corresponds to element i+1 (since elements are 1..256).
# So bit 0 is element 1, bit 255 is element 256.

# Precompute for each element x (1..256):
# 1. The bitmask of elements that, if present along with x, would complete a 4-AP.
#    Actually, it's better to precompute the "conflict" mask for 4-APs.
#    If we add x, we must ensure that for every 4-AP containing x, the other 3 elements are NOT all present.
#    This is hard to check with a single mask AND operation because we need "NOT (A & B & C)".
#    
#    Alternative: Precompute for each x, the list of 4-APs it belongs to.
#    To check validity of adding x to mask M:
#    For each 4-AP (a,b,c,d) containing x:
#       Let others = {a,b,c,d} \ {x}
#       If (M & mask(others)) == mask(others):
#           Then adding x creates a 4-AP.
#    
#    To make this fast, we can precompute for each x a list of masks representing the "other 3 elements" of each 4-AP containing x.
#    Let's call this `conflict_masks_4[x]`. It's a list of integers (bitmasks).
#    Check: any((M & cm) == cm for cm in conflict_masks_4[x])
    
# 2. The bitmask of elements that form a 3-AP with x.
#    For counting 3-APs, we want to count how many 3-APs are fully contained in M.
#    Let's precompute `three_ap_masks`. A list of all 3-AP masks.
#    Counting is sum(1 for m in three_ap_masks if (M & m) == m).
#    This is O(#3-APs) which is ~33k. Too slow for every SA step if we do full recount.
#    
#    Better: Incremental count.
#    When adding x:
#    New 3-APs formed are those containing x where the other 2 elements are in M.
#    Let `partner_masks_3[x]` be a list of masks for the "other 2 elements" of each 3-AP containing x.
#    Delta = sum(1 for pm in partner_masks_3[x] if (M & pm) == pm)
#    
#    When removing x:
#    Lost 3-APs are those containing x where the other 2 elements are in M (before removal).
#    Delta = -sum(1 for pm in partner_masks_3[x] if (M & pm) == pm)
    
# Precomputation
conflict_masks_4 = [None] * (N + 1)  # 1-indexed
partner_masks_3 = [None] * (N + 1)

# Initialize
for i in range(1, N + 1):
    conflict_masks_4[i] = []
    partner_masks_3[i] = []

# Generate 4-APs
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        # Mask for the 4 elements
        # For each element in the AP, add the mask of the other 3 to its conflict list
        # a
        conflict_masks_4[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
        # b
        conflict_masks_4[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
        # c
        conflict_masks_4[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1)))
        # e
        conflict_masks_4[e].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

# Generate 3-APs
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        # For each element, add mask of the other 2 to partner list
        partner_masks_3[a].append((1 << (b - 1)) | (1 << (c - 1)))
        partner_masks_3[b].append((1 << (a - 1)) | (1 << (c - 1)))
        partner_masks_3[c].append((1 << (a - 1)) | (1 << (b - 1)))

def count_3aps(mask):
    """Count 3-APs in mask. Used for initialization and verification."""
    count = 0
    # Iterate over all 3-APs? No, we can use the partner lists if we iterate over set bits.
    # But for initial random sets, it's easier to just sum over all 3-AP masks.
    # Let's precompute all 3-AP masks as a list of integers.
    pass

# Actually, for the SA loop, we need fast delta calculation.
# For initialization, we can build the mask and count.
# Let's precompute a list of all 3-AP masks for the initial count and potential verification.
all_3ap_masks = []
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        all_3ap_masks.append(m)

def count_3aps_full(mask):
    count = 0
    for m in all_3ap_masks:
        if (mask & m) == m:
            count += 1
    return count

def is_4ap_free_addition(mask, x):
    """Check if adding x to mask creates a 4-AP."""
    bit_x = 1 << (x - 1)
    if mask & bit_x:
        return False # Already in set
    for cm in conflict_masks_4[x]:
        if (mask & cm) == cm:
            return False
    return True

def delta_3aps_addition(mask, x):
    """Calculate number of new 3-APs formed by adding x to mask."""
    delta = 0
    for pm in partner_masks_3[x]:
        if (mask & pm) == pm:
            delta += 1
    return delta

def delta_3aps_removal(mask, x):
    """Calculate number of 3-APs lost by removing x from mask."""
    delta = 0
    for pm in partner_masks_3[x]:
        if (mask & pm) == pm:
            delta += 1
    return delta

def greedy_initial(seed_val=0):
    """Generate a 4-AP free set using a greedy approach."""
    random.seed(seed_val)
    mask = 0
    candidates = list(range(1, N + 1))
    if seed_val % 2 == 0:
        random.shuffle(candidates)
    
    for x in candidates:
        if is_4ap_free_addition(mask, x):
            mask |= (1 << (x - 1))
    return mask

def simulated_annealing(initial_mask, start_time, deadline):
    """
    Perform Simulated Annealing to maximize 3-APs while maintaining 4-AP freedom.
    """
    current_mask = initial_mask
    current_score = count_3aps_full(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    # Temperature schedule
    T = 10.0
    cooling_rate = 0.999
    
    # We need to pick random moves.
    # Moves: Add x, Remove x, Swap x,y
    
    while time.time() < deadline:
        T *= cooling_rate
        if T < 0.01:
            break
            
        # Random move selection
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            x = random.randint(1, N)
            if current_mask & (1 << (x - 1)):
                continue
            if not is_4ap_free_addition(current_mask, x):
                continue
            
            delta = delta_3aps_addition(current_mask, x)
            new_mask = current_mask | (1 << (x - 1))
            new_score = current_score + delta
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        elif move_type < 0.8:
            # Remove
            # Pick a random element from the set
            # Efficient way: pick a random index, if not in set, retry?
            # Or maintain a list of set bits? 
            # Given N=256, we can just pick random x and check if in set.
            # If not in set, skip.
            x = random.randint(1, N)
            if not (current_mask & (1 << (x - 1))):
                continue
            
            delta = -delta_3aps_removal(current_mask, x)
            new_mask = current_mask & ~(1 << (x - 1))
            new_score = current_score + delta
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        else:
            # Swap
            # Pick two distinct random elements
            x = random.randint(1, N)
            y = random.randint(1, N)
            if x == y:
                continue
            
            bit_x = 1 << (x - 1)
            bit_y = 1 << (y - 1)
            
            # Case 1: x in set, y not in set
            if (current_mask & bit_x) and not (current_mask & bit_y):
                # Check if removing x and adding y is valid
                # First, check if adding y is valid in the context of (current_mask - x)
                temp_mask = current_mask & ~bit_x
                if not is_4ap_free_addition(temp_mask, y):
                    continue
                
                # Calculate delta
                # Remove x: lose delta_3aps_removal(current_mask, x)
                # Add y: gain delta_3aps_addition(temp_mask, y)
                delta_remove = -delta_3aps_removal(current_mask, x)
                delta_add = delta_3aps_addition(temp_mask, y)
                delta = delta_remove + delta_add
                
                new_mask = temp_mask | bit_y
                new_score = current_score + delta
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        
            # Case 2: y in set, x not in set
            elif (current_mask & bit_y) and not (current_mask & bit_x):
                temp_mask = current_mask & ~bit_y
                if not is_4ap_free_addition(temp_mask, x):
                    continue
                
                delta_remove = -delta_3aps_removal(current_mask, y)
                delta_add = delta_3aps_addition(temp_mask, x)
                delta = delta_remove + delta_add
                
                new_mask = temp_mask | bit_x
                new_score = current_score + delta
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        
            # Case 3: Both in set or both not in set -> No change in cardinality, but might change structure?
            # Actually, swapping two elements both in set or both out of set results in the same set.
            # So we skip these.
            
    return best_mask, best_score

def mask_to_list(mask):
    """Convert bitmask to sorted list of elements."""
    elements = []
    for i in range(N):
        if mask & (1 << i):
            elements.append(i + 1)
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s budget, 10s safety margin
    
    best_mask_overall = 0
    best_score_overall = -1
    
    # Generate diverse initial candidates
    # 1. Standard Greedy (1..256)
    # 2. Randomized Greedy (shuffled)
    
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial(0))
    
    # Candidates 1-5: Randomized Greedies
    for i in range(1, 6):
        if time.time() > deadline - 10: # Leave time for SA
            break
        candidates.append(greedy_initial(i))
        
    # Evaluate and optimize each candidate
    for idx, init_mask in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        # Run SA with a slice of the remaining time
        # Allocate time based on number of candidates
        # Let's just run until global deadline for the last candidate,
        # and use a fraction for others?
        # Simpler: Run SA for a fixed short time per candidate, then one long run for the best?
        # Or just run SA until deadline for each, but break if we find a very good one?
        # Given SA is fast, we can afford multiple runs.
        
        # Let's allocate 10s per candidate if < 6 candidates, else less.
        # Actually, let's just run SA until the global deadline for the *last* candidate,
        # and for previous ones, use a shorter time budget.
        
        if idx < len(candidates) - 1:
            sa_deadline = start_time + 80.0 - 10.0 # Leave 10s for the final best candidate's refinement
            # Actually, simpler: just run all of them with a time slice.
            # Let's do: 5s per candidate, then take the best and refine it for the rest of the time.
            sa_deadline = time.time() + 5.0
        else:
            sa_deadline = deadline
            
        sa_mask, sa_score = simulated_annealing(init_mask, start_time, sa_deadline)
        
        if sa_score > best_score_overall:
            best_score_overall = sa_score
            best_mask_overall = sa_mask
            
    # If we have time left, refine the best mask found
    if best_mask_overall != 0 and time.time() < deadline - 1:
        # Run SA again on the best mask found
        final_mask, final_score = simulated_annealing(best_mask_overall, start_time, deadline)
        if final_score > best_score_overall:
            best_mask_overall = final_mask
            best_score_overall = final_score
            
    if best_mask_overall == 0:
        # Fallback
        best_mask_overall = candidates[0]
        
    return mask_to_list(best_mask_overall)

# EVOLVE-BLOCK-END
