import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 32
    MAXV = 512
    MAX_D = 255  # Max possible d is (512-1)/2 = 255.5 -> 255
    
    # Precompute valid 3-APs for each d?
    # For a given d, the APs are (a, a+d, a+2d) where 1 <= a and a+2d <= 512.
    # a ranges from 1 to 512 - 2d.
    
    def score_and_counts(S):
        """
        Calculate score and count_d for a set S.
        Returns (score, count_d_array)
        """
        S_set = set(S)
        count_d = np.zeros(MAX_D + 1, dtype=np.int32)
        score = 0
        for d in range(1, MAX_D + 1):
            cnt = 0
            # Iterate through possible starting points a
            # a can go up to MAXV - 2d
            max_a = MAXV - 2 * d
            # Optimization: Instead of iterating all a, we can just check if ANY AP exists.
            # But we need the count for delta updates.
            # So we must count all APs.
            # Number of APs for a given d is MAXV - 2d.
            # Sum over d=1..255 of (512-2d) is approx 255*512 - 2*255*256/2 = 130560 - 65280 = 65280.
            # This is small enough to do directly.
            for a in range(1, max_a + 1):
                if a in S_set and (a + d) in S_set and (a + 2 * d) in S_set:
                    cnt += 1
            if cnt > 0:
                score += 1
            count_d[d] = cnt
        return score, count_d

    def delta_add(y, S_set, count_d, d):
        """Check if adding y to S increases count_d[d]."""
        # y is first
        if y + d <= MAXV and (y + d) in S_set and (y + 2 * d) <= MAXV and (y + 2 * d) in S_set:
            return True
        # y is middle
        if y - d >= 1 and (y - d) in S_set and y + d <= MAXV and (y + d) in S_set:
            return True
        # y is last
        if y - 2 * d >= 1 and (y - 2 * d) in S_set and y - d >= 1 and (y - d) in S_set:
            return True
        return False

    def delta_remove(x, S_set, count_d, d):
        """Check if x is part of any AP with diff d in current S."""
        # x is first
        if x + d <= MAXV and (x + d) in S_set and (x + 2 * d) <= MAXV and (x + 2 * d) in S_set:
            return True
        # x is middle
        if x - d >= 1 and (x - d) in S_set and x + d <= MAXV and (x + d) in S_set:
            return True
        # x is last
        if x - 2 * d >= 1 and (x - 2 * d) in S_set and x - d >= 1 and (x - d) in S_set:
            return True
        return False

    def evaluate_move(remove_x, add_y, S_set, count_d, current_score):
        """
        Calculate the new score if we remove x and add y.
        Returns (new_score, valid)
        """
        if remove_x == add_y:
            return current_score, False
        
        # Calculate delta
        delta = 0
        
        # Effect of removing x
        for d in range(1, MAX_D + 1):
            if count_d[d] > 0:
                if delta_remove(remove_x, S_set, count_d, d):
                    # x is part of an AP with diff d
                    # count_d[d] will decrease by 1
                    if count_d[d] == 1:
                        # This AP was the only one, so d becomes uncovered
                        delta -= 1
        
        # Effect of adding y
        # Note: We must consider the state of S *after* removing x.
        # So S_set_temp = S_set - {x}
        S_set_temp = set(S_set)
        S_set_temp.remove(remove_x)
        
        for d in range(1, MAX_D + 1):
            # Check if y forms an AP with diff d in S_set_temp
            if delta_add(add_y, S_set_temp, count_d, d):
                # y forms an AP with diff d
                # We need to know if this AP was already counted in count_d[d] (which is for S_set)
                # Wait, count_d is for the current S_set (before move).
                # If y forms an AP in S_set_temp, does it mean count_d[d] increases?
                # Yes, because y is new.
                # But we must ensure that the AP formed by y was not already "covered" by the removal of x?
                # No, y is not in S_set, so no AP involving y existed in S_set.
                # So any AP formed by y in S_set_temp is a NEW AP.
                # However, we must check if d was *already* covered by other APs in S_set_temp.
                # If count_d[d] (from S_set) was > 0, and we removed x, did count_d[d] drop to 0?
                # We already accounted for the drop in the 'remove x' part.
                # Let's track the effective count for each d.
                
                # Actually, it's safer to just compute the new count for d if it's affected.
                # But iterating all d and checking is O(D).
                # Let's refine:
                # We want to know if the *set* of covered differences changes.
                
                # Let's just calculate the new count_d for affected d's.
                # This is getting complicated to do incrementally without errors.
                # Given D=255, maybe just recalculating the score for the whole set is faster than complex delta logic?
                # Recalculating score is O(N^2) = 32*32 = 1024 ops + set lookups.
                # Delta logic is O(D) = 255 ops + set lookups.
                # Delta logic should be faster.
                
                # Let's stick to a simpler delta:
                # 1. Identify all d where count changes.
                # 2. For those d, check if coverage status (count>0) changes.
                
                pass # Placeholder
        
        # Simpler approach: Just recalculate score for the new set?
        # With 150s, if we can do 100k moves, and each move takes 1ms, that's 100s.
        # Recalculation:
        # new_S = S - {x} + {y}
        # new_score = _count_progression_diffs(new_S)
        # This is robust. Let's optimize _count_progression_diffs.
        
        return current_score, True # Fallback

    # Let's use the robust recalculation with an optimized counter.
    def fast_score(S):
        S_set = set(S)
        diffs = set()
        S_list = list(S)
        for a in S_list:
            for b in S_list:
                d = b - a
                if d > 0:
                    if a + 2 * d in S_set:
                        diffs.add(d)
        return len(diffs)

    best_set = set(range(1, N + 1))
    best_score = fast_score(best_set)
    
    rng = random.Random(42)
    
    # Multiple restarts with different initializations
    initial_sets = [
        set(range(1, N + 1)),
        set(range(1, N + 1, 2)), # Odds
        set(range(2, 2*N + 1, 2)), # Evens
        set(rng.sample(range(1, MAXV + 1), N)),
        set(rng.sample(range(1, MAXV + 1), N)),
    ]
    
    for init in initial_sets:
        if time.time() > deadline:
            break
            
        current = set(init)
        current_score = fast_score(current)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline:
            # Pick a random element to remove
            x = rng.choice(list(current))
            # Pick a random element to add
            # To make it smarter, pick from a range near the existing elements?
            # Or just random 1..512
            y = rng.randint(1, MAXV)
            
            if x == y:
                continue
                
            new_set = set(current)
            new_set.remove(x)
            new_set.add(y)
            
            new_score = fast_score(new_set)
            
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current)
            else:
                delta = new_score - current_score
                if delta < 0:
                    # Accept with probability exp(delta / T)
                    if rng.random() < np.exp(delta / T):
                        current = new_set
                        current_score = new_score
                # else delta == 0, always accept to wander
                
            T *= cooling
            if T < T_min:
                T = T_min

    return sorted(list(best_set))
