import math
import time

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # 1. Greedy Initialization
    # We want to maximize sum 1/(512-a).
    # This is equivalent to maximizing sum of weights w[a] = 1/(512-a).
    # w[a] is strictly increasing with a.
    # So we prefer larger a.
    
    chosen = []
    # Iterate from largest to smallest
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    # Calculate current score
    def get_score(s):
        return sum(1.0 / (512.0 - a) for a in s)
    
    current_score = get_score(chosen)
    chosen_set = set(chosen)
    
    # 2. Local Search
    # We look for swaps (x, y) where x in chosen, y not in chosen, y > x
    # such that y is coprime to (chosen - {x})
    # This implies x was blocking y (or some other element was, but if x is removed, 
    # we only care if y is coprime to the rest).
    # Actually, if y was not chosen by greedy, it must have conflicted with some 
    # element z > y in chosen.
    # If we remove x, and x is that blocker, y becomes valid.
    # If x is not the blocker, removing x doesn't help y.
    # So we just need to check: is y coprime to all z in chosen \ {x}?
    
    # To optimize, we can precompute the "blockers" for each y not in chosen.
    # But a simple loop is sufficient given N=511.
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
            
        improved = False
        best_swap = None
        best_gain = 0
        
        # Sort chosen to iterate? No, just iterate.
        # We want to find max gain: 1/(512-y) - 1/(512-x)
        # Since y > x, gain is positive.
        
        # Optimization: Only consider y > x.
        # We can iterate y from 511 down to 1, and x from y-1 down to 1?
        # No, x must be in chosen.
        
        # Let's iterate over all x in chosen.
        for x in chosen:
            # For each x, find best y > x not in chosen
            # such that gcd(y, z) == 1 for all z in chosen \ {x}
            
            # To speed up, we can check candidates y in descending order of y
            # because higher y gives higher score.
            # But we need to check all y to find the *best* swap globally?
            # Or just find *any* improving swap and take the best one found in this pass?
            # Let's find the best swap in one pass.
            
            # Pre-compute the set of chosen without x?
            # chosen_without_x = chosen_set - {x}
            
            # Iterate y from 511 down to x+1
            for y in range(511, x, -1):
                if y in chosen_set:
                    continue
                    
                # Check if y is coprime to all z in chosen \ {x}
                is_coprime = True
                for z in chosen:
                    if z == x:
                        continue
                    if gcd(y, z) != 1:
                        is_coprime = False
                        break
                
                if is_coprime:
                    gain = (1.0 / (512.0 - y)) - (1.0 / (512.0 - x))
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
            
            # Early termination for this x if we found a very good swap?
            # No, just continue to check other x's.
            
        if best_swap:
            x, y = best_swap
            chosen_set.remove(x)
            chosen_set.add(y)
            chosen = list(chosen_set)
            current_score += best_gain
            improved = True
    
    return list(chosen_set)
