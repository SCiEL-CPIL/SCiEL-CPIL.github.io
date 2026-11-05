import time
import random
import sys

N = 256

# Precomputation for fast bitmask operations
# We represent a set A as a bitmask of length N.
# Bit i-1 corresponds to element i.

# Precompute 4-APs and 3-APs
# For each element x, we store:
# 1. A list of masks representing the "other" elements in each 4-AP containing x.
#    If any of these masks is a subset of the current set (excluding x), adding x creates a 4-AP.
# 2. A list of masks representing the "other" elements in each 3-AP containing x.
#    The number of such masks that are subsets of the current set (excluding x) is the gain in 3-APs.

element_to_4ap_others = [[] for _ in range(N)]
element_to_3ap_others = [[] for _ in range(N)]

# Generate 4-APs
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        ma = 1 << (a - 1)
        mb = 1 << (b - 1)
        mc = 1 << (c - 1)
        me = 1 << (e - 1)
        
        element_to_4ap_others[a-1].append(mb | mc | me)
        element_to_4ap_others[b-1].append(ma | mc | me)
        element_to_4ap_others[c-1].append(ma | mb | me)
        element_to_4ap_others[e-1].append(ma | mb | mc)

# Generate 3-APs
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        ma = 1 << (a - 1)
        mb = 1 << (b - 1)
        mc = 1 << (c - 1)
        
        element_to_3ap_others[a-1].append(mb | mc)
        element_to_3ap_others[b-1].append(ma | mc)
        element_to_3ap_others[c-1].append(ma | mb)

def is_4ap_free(mask, x):
    """Check if adding x to mask results in a set with no 4-APs."""
    # x is 1-indexed in the problem, 0-indexed in our list logic.
    # mask does NOT contain x.
    idx = x - 1
    for other_mask in element_to_4ap_others[idx]:
        if (mask & other_mask) == other_mask:
            return False
    return True

def delta_3aps(mask, x):
    """Calculate the change in the number of 3-APs when adding x to mask."""
    idx = x - 1
    count = 0
    for other_mask in element_to_3ap_others[idx]:
        if (mask & other_mask) == other_mask:
            count += 1
    return count

def count_3aps(mask):
    """Count the total number of 3-APs in the set represented by mask."""
    # We can iterate through all 3-APs and check if all three elements are in the mask.
    # This is O(N^2) but N=256 is small.
    # However, for SA, we need fast delta updates.
    # We will maintain the score incrementally in the main loop.
    # This function is for initial evaluation and final verification.
    count = 0
    # Iterate over all possible 3-APs
    for d in range(1, N // 2 + 1):
        for start in range(1, N - 2 * d + 1):
            a, b, c = start, start + d, start + 2 * d
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))) and (mask & (1 << (c - 1))):
                count += 1
    return count

def mask_to_list(mask):
    """Convert bitmask to sorted list of elements."""
    return [i + 1 for i in range(N) if (mask >> i) & 1]

def greedy_initial(seed_val, time_budget=1.0):
    """Generate an initial 4-AP free set using a greedy approach."""
    random.seed(seed_val)
    mask = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    start = time.time()
    for x in candidates:
        if time.time() > start + time_budget:
            break
        if is_4ap_free(mask, x):
            mask |= (1 << (x - 1))
    return mask

def random_initial(seed_val, density=0.5):
    """Generate a random initial set, then repair it by removing elements that create 4-APs."""
    random.seed(seed_val)
    mask = 0
    # Start with a random set
    for i in range(1, N + 1):
        if random.random() < density:
            mask |= (1 << (i - 1))
    
    # Repair: remove elements that cause 4-APs
    # We need to be careful about the order.
    # A simple approach: iterate through elements, if adding it (or keeping it) causes a 4-AP, remove it.
    # Since we started with a random set, we can just check each element.
    # If an element is part of a 4-AP with other elements in the set, we might need to remove one of them.
    # To keep it simple and valid, we can just rebuild greedily from the random set?
    # Or, simpler: just use a very sparse random set that is likely 4-AP free.
    # Let's try a different strategy: Start empty, add random elements if valid.
    mask = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if is_4ap_free(mask, x):
            mask |= (1 << (x - 1))
    return mask

def simulated_annealing(initial_mask, deadline):
    """
    Run Simulated Annealing to maximize 3-APs while maintaining 4-AP freedom.
    """
    mask = initial_mask
    current_score = count_3aps(mask)
    best_mask = mask
    best_score = current_score
    
    # Temperature parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.99
    
    T = T_start
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_end:
            T = T_end
            
        # Choose a random move
        # 1. Add a random element
        # 2. Remove a random element
        # 3. Swap a random element with another
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random element
            # Find a random element not in mask
            # To do this efficiently, we can pick a random index and check if it's in mask.
            # Or maintain a list of available elements? That's expensive to update.
            # Let's just pick a random x in [1, N] and check.
            x = random.randint(1, N)
            if not (mask & (1 << (x - 1))):
                if is_4ap_free(mask, x):
                    new_mask = mask | (1 << (x - 1))
                    delta = delta_3aps(mask, x)
                    new_score = current_score + delta
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                    # Accept move?
                    # In SA, we accept worse moves with probability exp(-delta_E / T)
                    # Here, we want to MAXIMIZE score. So delta_E = -delta_score.
                    # If delta_score > 0, always accept.
                    # If delta_score < 0, accept with prob exp(delta_score / T).
                    if delta > 0:
                        mask = new_mask
                        current_score = new_score
                    else:
                        if random.random() < pow(2.718281828, delta / T):
                            mask = new_mask
                            current_score = new_score
                            
        elif move_type < 0.7:
            # Try to remove a random element
            # Pick a random element in mask
            # We can do this by picking a random index and checking, or using a list.
            # Let's use a list of elements in mask for efficiency?
            # Actually, N is small, so random sampling with rejection is fine.
            x = random.randint(1, N)
            if mask & (1 << (x - 1)):
                new_mask = mask & ~(1 << (x - 1))
                # Delta is negative of gain from adding x to new_mask
                # Gain from adding x to new_mask is delta_3aps(new_mask, x)
                # So loss from removing x from mask is -delta_3aps(new_mask, x)
                delta = -delta_3aps(new_mask, x)
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask
                if delta > 0:
                    mask = new_mask
                    current_score = new_score
                else:
                    if random.random() < pow(2.718281828, delta / T):
                        mask = new_mask
                        current_score = new_score

        else:
            # Try a swap: remove x, add y
            x = random.randint(1, N)
            if not (mask & (1 << (x - 1))):
                # x is not in mask, so we can't remove it.
                # Let's try to find an x in mask and a y not in mask.
                # To ensure we get a valid swap, we might need to retry.
                # For simplicity, let's just do add/remove for now.
                # Swaps are more complex to handle with SA acceptance criteria.
                pass
            else:
                # x is in mask. Find a random y not in mask.
                y = random.randint(1, N)
                if y == x or (mask & (1 << (y - 1))):
                    pass
                else:
                    # Check if swapping x for y is valid (no 4-APs)
                    mask_without_x = mask & ~(1 << (x - 1))
                    if is_4ap_free(mask_without_x, y):
                        new_mask = mask_without_x | (1 << (y - 1))
                        # Calculate delta
                        # Loss from removing x: -delta_3aps(mask_without_x, x)
                        # Gain from adding y: delta_3aps(mask_without_x, y)
                        loss_x = -delta_3aps(mask_without_x, x)
                        gain_y = delta_3aps(mask_without_x, y)
                        delta = loss_x + gain_y
                        new_score = current_score + delta
                        if new_score > best_score:
                            best_score = new_score
                            best_mask = new_mask
                        if delta > 0:
                            mask = new_mask
                            current_score = new_score
                        else:
                            if random.random() < pow(2.718281828, delta / T):
                                mask = new_mask
                                current_score = new_score
                            
        T *= cooling_rate
        
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_mask = None
    best_score = -1
    
    # Generate diverse initial candidates
    # 1. Greedy (deterministic)
    # 2. Greedy (randomized)
    # 3. Random (sparse)
    
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial(0, time_budget=0.5))
    
    # Candidates 1-5: Randomized Greedies
    for i in range(1, 6):
        if time.time() > deadline - 20:
            break
        candidates.append(greedy_initial(i, time_budget=0.5))
        
    # Candidates 6-10: Random Sparse (using the repair logic which is effectively greedy but with random order)
    for i in range(6, 11):
        if time.time() > deadline - 20:
            break
        candidates.append(random_initial(i))
        
    # Run Simulated Annealing on each candidate
    # Allocate time based on remaining time and number of candidates
    num_candidates = len(candidates)
    time_per_candidate = (deadline - time.time() - 5) / num_candidates if num_candidates > 0 else 0
    
    for idx, init_mask in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        search_deadline = min(deadline - 2, time.time() + time_per_candidate)
        
        # If time per candidate is too small, skip SA and just use the initial mask
        if search_deadline - time.time() < 1.0:
            optimized = init_mask
        else:
            optimized = simulated_annealing(init_mask, search_deadline)
            
        opt_score = count_3aps(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_mask = optimized
            
    # If we have time left, do a final polish on the best mask found
    if best_mask is not None and time.time() < deadline - 2:
        final_deadline = deadline - 1
        # Run SA again on the best mask found
        best_mask = simulated_annealing(best_mask, final_deadline)
        final_score = count_3aps(best_mask)
        if final_score > best_score:
            best_score = final_score
    
    if best_mask is None:
        best_mask = candidates[0] if candidates else greedy_initial(0)
    
    return mask_to_list(best_mask)

# Test/Validation (optional, for debugging)
if __name__ == "__main__":
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Cardinality: {len(result)}")
    print(f"First 10 elements: {result[:10]}")
    
    # Verify no 4-APs
    mask = 0
    for x in result:
        mask |= (1 << (x - 1))
        
    valid = True
    for d in range(1, N // 3 + 1):
        for start in range(1, N - 3 * d + 1):
            a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))) and (mask & (1 << (c - 1))) and (mask & (1 << (e - 1))):
                valid = False
                print(f"Found 4-AP: {a}, {b}, {c}, {e}")
                break
        if not valid:
            break
            
    print(f"Valid (no 4-APs): {valid}")
    print(f"Number of 3-APs: {count_3aps(mask)}")
