import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    n = 12
    best_max = 2048  # powers of 2
    best_set = [1 << i for i in range(n)]
    
    # Beam search
    beam = [([1], {0, 1}, {1})]  # (set, subset_sums, differences)
    # Actually, storing full sets of sums for beam width 10 and depth 12 is feasible.
    # 10 * 2^12 * 12 operations?
    
    # To speed up, we can just store the set and recompute sums? No, recompute is expensive.
    # We store the set of sums.
    
    # Initialize
    # S = {1}
    # Sums = {0, 1}
    # Diffs = {1}
    
    current_beam = [([1], {0, 1}, {1})]
    
    for step in range(1, n):
        next_beam = []
        for s, sums, diffs in current_beam:
            # Find candidates
            # Greedy: smallest x not in diffs
            # Random: x in [greedy, greedy + 50]
            
            # Optimization: We only need to check x up to some limit?
            # The max element in the best set is likely < 1000.
            # So we can limit search range.
            
            # Get sorted diffs to find smallest missing quickly?
            # Or just iterate x from 1 upwards.
            # Since we are in a beam, we might have many diffs.
            
            # Let's just try a range of candidates.
            # The greedy choice is the lower bound for this prefix.
            greedy_x = 1
            while greedy_x in diffs:
                greedy_x += 1
            
            # Try candidates from greedy_x to greedy_x + 100
            # And maybe some random larger ones?
            
            candidates = list(range(greedy_x, greedy_x + 50))
            random.shuffle(candidates)
            
            for x in candidates:
                if x in diffs:
                    continue
                
                # Check if x is distinct from existing elements
                if x in s:
                    continue
                
                # New sums
                new_sums = set(sums)
                for sum_val in sums:
                    new_sums.add(sum_val + x)
                
                # Check for collisions
                # If len(new_sums) != 2 * len(sums), there is a collision
                if len(new_sums) != 2 * len(sums):
                    continue
                
                # New diffs
                # Diffs of new_sums
                # This is O(N^2) where N = 2^step.
                # For step=11, N=2048. N^2 = 4M.
                # 10 beams * 50 candidates * 4M = 2B operations. Too slow in Python.
                
                # We need a faster way to check collisions or compute diffs.
                # Collision check:
                # sums and sums+x are disjoint iff x not in diffs.
                # We already checked x not in diffs.
                # So new_sums are distinct.
                
                # We need new_diffs for the next step.
                # new_diffs = diffs (from old sums)
                #             U {x} (diff between 0 and x, or sum and sum+x)
                #             U {y - z | y in sums+x, z in sums}
                #             U {y - z | y in sums, z in sums+x} (same as above)
                
                # The new differences are:
                # 1. Old diffs
                # 2. x (diff between 0 and x)
                # 3. |a - b| where a in sums, b in sums+x.
                #    b = s' + x. a - b = a - s' - x.
                #    This is equivalent to | (a - s') - x |.
                #    Let d = a - s'. d is in old_diffs U {0}.
                #    So new diffs include |d - x| for d in old_diffs U {0}.
                
                # So:
                new_diffs = set(diffs)
                new_diffs.add(x)
                for d in diffs:
                    new_diffs.add(abs(d - x))
                new_diffs.add(x) # d=0 case
                
                next_beam.append((s + [x], new_sums, new_diffs))
                
                # Check time
                if time.time() > deadline:
                    return best_set
            
            # Keep best 10 based on max element
            next_beam.sort(key=lambda item: max(item[0]))
            next_beam = next_beam[:10]
            
        current_beam = next_beam
        if not current_beam:
            break
            
    # Evaluate final beam
    for s, sums, diffs in current_beam:
        if len(s) == n:
            if max(s) < best_max:
                best_max = max(s)
                best_set = s
                
    return best_set
