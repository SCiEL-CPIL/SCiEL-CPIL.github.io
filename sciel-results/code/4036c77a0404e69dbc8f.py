import itertools
import time
import random

def construct():
    N = 192
    candidates = list(range(N, 0, -1))
    
    best_size = 0
    best_set = set()
    start_time = time.time()
    timeout = 50  # Leave 10s buffer

    def greedy_solve(order):
        """
        Greedily build a set based on the given order of candidates.
        """
        selected = []
        sums_3 = set()
        
        # Pre-compute pairs? No, we add one by one.
        
        for x in order:
            # Check 1: Does x divide any existing sum of 3?
            valid = True
            for s in sums_3:
                if s % x == 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check 2: Do any existing y in selected divide (x + a + b)?
            # We need to check all pairs (a, b) from selected.
            # For each pair, sum = x + a + b.
            # Check if sum % y == 0 for any y in selected.
            
            # Optimization: 
            # Iterate y in selected.
            # For a fixed y, we need to check if there exist a, b in selected\{y} 
            # such that x + a + b = k*y.
            # a + b = k*y - x.
            
            # Simpler: Just iterate all pairs (a,b) and check all y.
            # But that is O(|S|^3).
            
            # Let's try the direct pair iteration.
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    s_new = x + a + b
                    # Check if any y in selected divides s_new
                    for y in selected:
                        if s_new % y == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Add x to selected
                selected.append(x)
                # Update sums_3
                # New sums are x + a + b for all a, b in selected (excluding x, which is new)
                # Wait, selected now contains x.
                # The new sums involve x and two OTHER elements.
                # The other elements are selected[:-1] (the old selected).
                old_selected = selected[:-1]
                for a, b in itertools.combinations(old_selected, 2):
                    sums_3.add(x + a + b)
                    
        return selected

    # Try different strategies
    # Strategy 1: Descending
    res1 = greedy_solve(candidates)
    if len(res1) > best_size:
        best_size = len(res1)
        best_set = set(res1)
        
    # Strategy 2: Ascending
    res2 = greedy_solve(list(reversed(candidates)))
    if len(res2) > best_size:
        best_size = len(res2)
        best_set = set(res2)
        
    # Strategy 3: Random restarts
    for _ in range(20):
        random.shuffle(candidates)
        res3 = greedy_solve(candidates)
        if len(res3) > best_size:
            best_size = len(res3)
            best_set = set(res3)
            
        if time.time() - start_time > timeout:
            break
            
    # If we haven't hit 48, we might try a local search or just return best.
    # Given the constraints, greedy usually finds a good set.
    
    if not best_set:
        return []
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
