import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = []
    
    def try_greedy(random_seed):
        random.seed(random_seed)
        selected = []
        sums = set()
        
        # We iterate through candidates 1 to 4096
        # To allow some diversity, we could shuffle, but for B3, 
        # smallest-first is usually best for density.
        # Let's stick to smallest-first but allow "skipping" via randomization?
        # Actually, just running standard greedy multiple times with different 
        # tie-breaking or slight perturbations is good.
        
        # Standard Greedy:
        for x in range(1, 4097):
            if time.time() > deadline:
                return selected
            
            # Check if x can be added
            # New sums:
            # 1. x + x + x
            # 2. x + x + s for s in selected
            # 3. x + s1 + s2 for s1, s2 in selected (s1 <= s2)
            
            new_sums = []
            
            # 1
            s1 = 3 * x
            if s1 in sums:
                continue
            new_sums.append(s1)
            
            # 2
            for s in selected:
                s2 = 2 * x + s
                if s2 in sums or s2 in new_sums:
                    # Check against new_sums is important for internal collisions
                    # But since we add to set later, we must check pairwise
                    # Optimization: just collect and check set at end?
                    # No, early exit is faster.
                    # However, checking `s2 in new_sums` is O(k).
                    # Let's use a temporary set for new sums?
                    pass
            # This logic is getting messy. Let's use a temp set for new sums.
            
            temp_new_sums = set()
            
            # 1
            val = 3 * x
            if val in sums or val in temp_new_sums:
                continue
            temp_new_sums.add(val)
            
            # 2
            for s in selected:
                val = 2 * x + s
                if val in sums or val in temp_new_sums:
                    continue # Invalid
                temp_new_sums.add(val)
            
            # 3
            # Pairs (s1, s2) from selected
            # Since selected is sorted, we can use combinations_with_replacement
            # But we can break early if sum is too big? No, sums are bounded by 3*4096.
            for i in range(len(selected)):
                s1 = selected[i]
                for j in range(i, len(selected)):
                    s2 = selected[j]
                    val = x + s1 + s2
                    if val in sums or val in temp_new_sums:
                        # Collision found
                        # We need to abort this candidate
                        # But we are in a loop. We need a flag or exception.
                        # Let's use a flag.
                        break
                    temp_new_sums.add(val)
                else:
                    continue
                break
            
            if len(temp_new_sums) == 1 + len(selected) + len(selected)*(len(selected)+1)//2:
                # All sums unique and disjoint from existing
                selected.append(x)
                sums.update(temp_new_sums)
        
        return selected

    # Run standard greedy first
    best_set = try_greedy(0)
    best_len = len(best_set)
    
    # Try some randomized variations?
    # For B3 sets, the greedy is quite deterministic.
    # However, we can try to skip some small numbers to see if it helps?
    # Usually, for Sidon sets, the greedy produces a near-optimal set.
    # The main gain here is going up to 4096 instead of 513 and using O(k^2) check.
    
    # If we have time, we can try to "repair" or improve?
    # Or just run greedy with different starting points?
    # e.g. start with {2, 3, ...}?
    
    # Let's just run the fast greedy. It should find a much larger set than the seed.
    # The seed found ~10-20 elements in range 512?
    # With range 4096 and fast check, we should get significantly more.
    
    # To be safe, let's add a simple randomized restart loop if time permits.
    # But standard greedy is likely the best single shot.
    
    # Let's verify the count.
    # A B3 set in [1, N] has size approx N^(1/3) * constant?
    # No, B_h sets have size O(N^(1/h)). For h=3, size is O(N^(1/3)).
    # 4096^(1/3) = 16.
    # Wait. The number of triple sums is ~ k^3/6.
    # The range of sums is 3 to 3*4096 = 12288.
    # So k^3/6 <= 12288 => k^3 <= 73728 => k <= 42.
    # So max size is around 40-50.
    
    # The seed stopped at 512. 512^(1/3) = 8. So seed likely found ~8-10 elements.
    # We can find ~40-50 elements.
    # This is a huge improvement.
    
    return best_set
