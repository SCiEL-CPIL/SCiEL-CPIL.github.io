import time
import random
import math

def construct():
    start_time = time.time()
    # Leave a safety margin to avoid being killed by the evaluator
    # The limit is 150s, we use 145s for computation
    time_limit = 145.0
    
    N = 512
    K = 32
    
    def score_mask(mask):
        """
        Count distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
        mask: int where bit (x-1) is set iff x is in the set.
        """
        count = 0
        # d can range from 1 to 255 (since max value is 512, a+2d <= 512 implies 2d < 512)
        # Actually if a=1, a+2d=512 => 2d=511 => d=255.5, so max integer d is 255.
        for d in range(1, 256):
            # Check if there exists a such that bit a is set, bit a+d is set, bit a+2d is set.
            # mask & (mask >> d) & (mask >> 2d) will have bit a set if the progression exists.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    # 1. Greedy Initialization
    # Start with an empty set and add elements one by one, choosing the one that maximizes the score.
    current_mask = 0
    current_set = set()
    
    # We can optimize the greedy step slightly by precomputing or just iterating.
    # 32 iterations * 512 candidates * 255 checks = ~4 million bit ops. Very fast.
    for _ in range(K):
        best_x = -1
        best_score = -1
        # Iterate through all possible values
        for x in range(1, N + 1):
            if x in current_set:
                continue
            # Create a temporary mask with x added
            temp_mask = current_mask | (1 << (x - 1))
            sc = score_mask(temp_mask)
            if sc > best_score:
                best_score = sc
                best_x = x
        
        if best_x != -1:
            current_set.add(best_x)
            current_mask |= (1 << (best_x - 1))
            
    current_score = best_score
    best_score = current_score
    best_mask = current_mask
    best_set = set(current_set)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - 5:
        return sorted(list(best_set))
    
    # 2. Simulated Annealing Polish
    # We will use the remaining time for SA.
    # T_start and T_end are chosen to allow exploration initially and exploitation later.
    # Delta is typically small (integer steps in score), so T=5.0 is a reasonable start.
    T_start = 10.0
    T_end = 0.1
    
    # Precompute list of set elements for faster random access?
    # Since K=32, list(current_set) is cheap.
    # However, for SA, we need to pick an element to remove and one to add.
    
    # We'll maintain a list of elements in the set for O(1) removal if we swap indices,
    # but since K is small, using a set and list conversion is acceptable if not in the innermost loop.
    # Actually, picking `to_remove` from `current_set` requires a list.
    # Picking `to_add` from 1..512 not in `current_set` requires rejection sampling or a list of non-members.
    # Rejection sampling is fine since density is 32/512 ~ 6%.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
        
        # Calculate progress and temperature
        # Ensure progress is between 0 and 1
        progress = elapsed / time_limit
        if progress > 1.0:
            progress = 1.0
            
        # Linear interpolation in log space for T
        # T = T_start * (T_end / T_start) ^ progress
        # Using pow is fine, or math.exp
        T = T_start * pow(T_end / T_start, progress)
        
        # Generate Move: Swap one element
        # Pick random element to remove
        # Converting set to list every step is O(K). 
        # To optimize, we can maintain a list `curr_list` parallel to `current_set`.
        # Let's do that for speed.
        
        # For now, let's just do it simply. If it's too slow, we optimize.
        # 32 elements is small.
        
        to_remove_list = list(current_set)
        to_remove = to_remove_list[random.randrange(K)]
        
        # Pick random element to add
        # Rejection sampling
        while True:
            to_add = random.randint(1, N)
            if to_add not in current_set:
                break
        
        # Update mask
        new_mask = current_mask
        new_mask &= ~(1 << (to_remove - 1))
        new_mask |= (1 << (to_add - 1))
        
        # Score new mask
        new_score = score_mask(new_mask)
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0:
            # Always accept improvement
            current_set.remove(to_remove)
            current_set.add(to_add)
            current_mask = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_mask = new_mask
                best_set = set(current_set)
        else:
            # Accept with probability exp(delta / T)
            # If delta == 0, prob = 1.
            if delta == 0:
                prob = 1.0
            else:
                # Avoid overflow/underflow issues, though delta is small
                prob = math.exp(delta / T)
            
            if random.random() < prob:
                current_set.remove(to_remove)
                current_set.add(to_add)
                current_mask = new_mask
                current_score = new_score

    # Return the best set found
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
