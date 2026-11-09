import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    selected = []
    
    # We iterate from largest to smallest to maximize the chance of fitting 48
    # Large numbers are less likely to divide sums of other large numbers
    for x in range(192, 0, -1):
        if len(selected) >= 48:
            break
            
        if time.time() > deadline:
            break
            
        # Check if x can be added to selected
        # Condition 1: x must not divide the sum of any 3 distinct elements in selected
        valid = True
        
        if len(selected) >= 3:
            # Pre-compute sums? No, just iterate.
            for c in combinations(selected, 3):
                if sum(c) % x == 0:
                    valid = False
                    break
        
        # Condition 2: For each a in selected, a must not divide any sum 
        # formed by x and 2 distinct elements from selected \ {a}
        if valid:
            for a in selected:
                # We need to check sums u + v + x where u, v are distinct 
                # and in selected, and u != a, v != a
                # It's easier to just iterate pairs from selected and check if they involve a
                # Or iterate pairs from selected, and if a is in the pair, skip?
                # No, u and v must be from selected \ {a}.
                
                # Optimization: 
                # We check if there exists u, v in selected \ {a} such that (u + v + x) % a == 0
                # This is (u + v) % a == -x % a
                
                # Iterate pairs u, v from selected
                # If u == a or v == a, skip
                # Since selected size is small, just nested loop
                for i in range(len(selected)):
                    u = selected[i]
                    if u == a:
                        continue
                    for j in range(i + 1, len(selected)):
                        v = selected[j]
                        if v == a:
                            continue
                        if (u + v + x) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
        if valid:
            selected.append(x)
            
    return selected
