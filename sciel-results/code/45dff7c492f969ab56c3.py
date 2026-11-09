import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # Max d such that 1 + 2d <= 512 -> 2d <= 511 -> d <= 255

    def time_ok():
        return time.time() < deadline

    # Bitmask scoring
    # mask is an int where bit i is 1 if (i+1) is in the set
    def score_mask(mask):
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def get_random_mask():
        lst = random.sample(range(1, MAX_VAL + 1), N)
        return list_to_mask(lst)

    def simulated_annealing(start_mask, time_limit):
        """
        Runs SA for the given time limit.
        Returns best mask and score found.
        """
        current_mask = start_mask
        current_score = score_mask(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        # Parameters
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        # Precompute indices for random selection? 
        # We can just pick random positions in the list representation
        # But bitmask is faster. Let's stick to bitmask and random values.
        
        while time.time() < time_limit:
            if not time_ok():
                break
                
            # Generate a move: remove one element, add another
            # Find a random element in the mask
            # We can iterate or pick random bit
            # Picking random bit from 0 to 511 and checking if it's in mask is O(1) avg if sparse, but N=32 is dense enough?
            # Better: keep a list representation for move generation, or just pick random x in 1..512
            # If x is in mask, remove it. If not, add it.
            # But we must maintain exactly N elements.
            # So: pick x in mask, pick y not in mask. Swap.
            
            # To pick x in mask efficiently:
            # We can maintain a list of elements.
            # Let's convert to list once? No, that's slow if done every step.
            # Let's pick random x in 1..512. If in mask, it's a candidate for removal.
            # If not, it's a candidate for addition.
            # We need one removal and one addition.
            
            # Strategy:
            # 1. Pick random x in 1..512.
            # 2. If x in mask: 
            #    a. Pick random y in 1..512.
            #    b. If y not in mask and y != x:
            #       new_mask = mask ^ (1 << (x-1)) ^ (1 << (y-1))
            #       ...
            # This has a low probability of success if N is small relative to 512.
            # N=32, 512 values. Prob(x in mask) = 32/512 = 1/16.
            # Prob(y not in mask) = 15/16.
            # Success prob ~ 1/16 * 15/16 ~ 6%.
            # This is inefficient.
            
            # Better: Maintain a list of elements in the set.
            # But updating list on every move is O(N).
            # Or, we can just generate a random permutation of the set? No.
            
            # Let's use the list representation for the state in SA, and convert to mask for scoring.
            # Or, better:
            # Pick i from 0 to N-1.
            # old_val = lst[i]
            # Pick new_val from 1 to 512.
            # If new_val in set: skip or retry.
            # This requires checking membership. A set object is O(1).
            
            # Let's restart SA with list + set + mask.
            
            pass # Placeholder, will rewrite below

    # Rewritten SA with list/set/mask for efficient moves
    def sa_with_state(start_mask, time_limit):
        lst = mask_to_list(start_mask)
        s_set = set(lst)
        current_mask = start_mask
        current_score = score_mask(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.005
        
        while time.time() < time_limit:
            if not time_ok():
                break
                
            # Pick a random index
            i = random.randint(0, N - 1)
            old_val = lst[i]
            
            # Pick a random new value
            new_val = random.randint(1, MAX_VAL)
            
            if new_val == old_val or new_val in s_set:
                # Small penalty or just skip? Skipping is safer.
                # To avoid infinite loops if set is dense, but 32/512 is sparse.
                continue
            
            # Calculate new mask
            new_mask = current_mask
            new_mask &= ~(1 << (old_val - 1))
            new_mask |= (1 << (new_val - 1))
            
            # Score new mask
            new_score = score_mask(new_mask)
            
            # Acceptance
            delta = new_score - current_score
            if delta >= 0 or random.random() < pow(2.71828, delta / T):
                # Accept
                lst[i] = new_val
                s_set.discard(old_val)
                s_set.add(new_val)
                current_mask = new_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            
            # Cool down
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_mask, best_score

    # Main logic
    random.seed(42)
    
    best_mask_overall = 0
    best_score_overall = 0
    
    # Strategy 1: Interval Start
    # S = {1, ..., 32}
    mask1 = 0
    for i in range(1, 33):
        mask1 |= (1 << (i - 1))
    
    s1 = score_mask(mask1)
    if s1 > best_score_overall:
        best_score_overall = s1
        best_mask_overall = mask1
        
    # Run SA on this start
    time_limit_1 = time.time() + 30.0
    m1, s1_sa = sa_with_state(mask1, time_limit_1)
    if s1_sa > best_score_overall:
        best_score_overall = s1_sa
        best_mask_overall = m1
        
    # Strategy 2: Random Start
    if time_ok():
        mask2 = get_random_mask()
        s2 = score_mask(mask2)
        if s2 > best_score_overall:
            best_score_overall = s2
            best_mask_overall = mask2
            
        time_limit_2 = time.time() + 30.0
        m2, s2_sa = sa_with_state(mask2, time_limit_2)
        if s2_sa > best_score_overall:
            best_score_overall = s2_sa
            best_mask_overall = m2

    # Strategy 3: Greedy Construction Start
    if time_ok():
        # Build a set greedily
        s_greedy = set()
        mask_greedy = 0
        # Start with 1
        s_greedy.add(1)
        mask_greedy |= (1 << 0)
        
        while len(s_greedy) < N:
            best_candidate = -1
            best_candidate_score = -1
            # Try adding each unused number
            for x in range(1, MAX_VAL + 1):
                if x in s_greedy:
                    continue
                # Tentative mask
                new_mask = mask_greedy | (1 << (x - 1))
                new_score = score_mask(new_mask)
                if new_score > best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate = x
            
            if best_candidate > -1:
                s_greedy.add(best_candidate)
                mask_greedy |= (1 << (best_candidate - 1))
            else:
                # Fallback to random
                x = random.randint(1, MAX_VAL)
                while x in s_greedy:
                    x = random.randint(1, MAX_VAL)
                s_greedy.add(x)
                mask_greedy |= (1 << (x - 1))
                
        s3 = score_mask(mask_greedy)
        if s3 > best_score_overall:
            best_score_overall = s3
            best_mask_overall = mask_greedy
            
        # Polish with SA
        time_limit_3 = time.time() + 20.0
        m3, s3_sa = sa_with_state(mask_greedy, time_limit_3)
        if s3_sa > best_score_overall:
            best_score_overall = s3_sa
            best_mask_overall = m3

    # Strategy 4: More Random Restarts
    # Use remaining time
    if time_ok():
        # Split remaining time into chunks
        remaining = deadline - time.time()
        if remaining > 10:
            # Do 5 more random restarts
            for i in range(5):
                if not time_ok():
                    break
                chunk_time = (deadline - time.time()) / (5 - i) * 0.8 # Leave some buffer
                if chunk_time < 5:
                    chunk_time = 5
                    if time.time() + chunk_time > deadline:
                        break
                
                mask_r = get_random_mask()
                s_r = score_mask(mask_r)
                if s_r > best_score_overall:
                    best_score_overall = s_r
                    best_mask_overall = mask_r
                    
                m_r, s_r_sa = sa_with_state(mask_r, time.time() + chunk_time)
                if s_r_sa > best_score_overall:
                    best_score_overall = s_r_sa
                    best_mask_overall = m_r

    # Final Polish on the best found
    if time_ok() and best_mask_overall != 0:
        final_limit = deadline - 5.0 # 5s buffer
        m_final, s_final = sa_with_state(best_mask_overall, final_limit)
        if s_final > best_score_overall:
            best_score_overall = s_final
            best_mask_overall = m_final

    # Convert best mask to list
    best_list = mask_to_list(best_mask_overall)
    
    # Ensure exactly 32 distinct integers
    if len(best_list) != N:
        # Fallback to interval if something went wrong
        best_list = list(range(1, 33))
        
    return best_list

