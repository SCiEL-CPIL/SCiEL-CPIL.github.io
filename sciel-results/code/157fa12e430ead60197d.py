import random
import time
from itertools import combinations

def construct():
        n = 16
        k = 4
        # Generate all 4-subsets as bitmasks
        all_sets = []
        for comb in combinations(range(n), k):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            all_sets.append(mask)
        
        # Precompute intersection of every pair? 
        # There are 1820 sets. Pairs: ~1.6M.
        # Storing intersections might be too much memory or time.
        # Let's just compute on the fly but optimize the check.
        
        def is_sunflower(a, b, c):
            ab = a & b
            ac = a & c
            bc = b & c
            return ab == ac == bc
        
        def greedy(order):
            chosen = []
            # For speed, we can maintain a list of chosen masks
            for s in order:
                ok = True
                # Check against all pairs in chosen
                # If |chosen| is large, this is slow.
                # Optimization: If we add s, we only need to check pairs (a,b) in chosen
                # such that a,b,s form sunflower.
                # We can break early if found.
                for i in range(len(chosen)):
                    a = chosen[i]
                    for j in range(i+1, len(chosen)):
                        b = chosen[j]
                        if is_sunflower(a, b, s):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    chosen.append(s)
            return chosen

        best_family = []
        start_time = time.time()
        deadline = start_time + 50 # 50s limit
        
        # Run multiple random greedy passes
        while time.time() < deadline:
            random.shuffle(all_sets)
            family = greedy(all_sets)
            if len(family) > len(best_family):
                best_family = family
                # Update best
                if len(best_family) > 200: # Heuristic threshold
                    break # Or continue to see if we can do better
            
        # Local search on the best family
        # Try to add any set from all_sets not in best_family
        # If it conflicts, try swapping?
        # For now, just return the best greedy result.
        
        # Convert masks back to lists of 1..16
        result = []
        for mask in best_family:
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(i+1)
            result.append(elems)
            
        return result
