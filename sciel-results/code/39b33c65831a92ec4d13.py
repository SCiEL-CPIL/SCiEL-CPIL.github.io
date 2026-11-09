"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 3-AP-free subset of [1, 256] maximizing the sum of reciprocals.
    Strategy:
    1. Start with the greedy construction (smallest elements first) to get a valid base set.
    2. Use a local search (simulated annealing style or hill climbing) to swap elements
       to increase the reciprocal sum while maintaining the 3-AP-free property.
    3. Use a deadline guard to ensure we finish within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    N = 256
    MAX_VAL = N
    
    # Helper to check if adding x to set S creates a 3-AP
    # A 3-AP (a, b, c) with a < b < c has b-a = c-b => 2b = a+c.
    # If we have set S and want to add x, we need to check:
    # 1. Is there a, b in S such that 2b - a = x? (x is the 3rd term)
    # 2. Is there a, c in S such that a + c = 2x? (x is the middle term)
    # 3. Is there b, c in S such that 2b - c = x? Wait, if b,c in S and 2b-c=x, then x, b, c is an AP?
    #    Let's rephrase. An AP is (x1, x2, x3) with x1+x3=2*x2.
    #    If we add x to S, the new APs must involve x.
    #    Case 1: x is the first term: x, b, c in S? No, b,c in S. AP: x, b, c => 2b = x+c => c = 2b-x.
    #            So check if there exists b in S such that c = 2b-x is in S and c > b > x.
    #    Case 2: x is the middle term: a, x, c in S? a,c in S. AP: a, x, c => 2x = a+c.
    #            So check if there exists a in S such that c = 2x-a is in S and c > x > a.
    #    Case 3: x is the last term: a, b, x in S? a,b in S. AP: a, b, x => 2b = a+x => a = 2b-x.
    #            So check if there exists b in S such that a = 2b-x is in S and x > b > a.
    
    # For efficiency, we represent S as a boolean array or a set.
    # Since N=256 is small, a set of integers is fast enough for membership checks.
    # However, checking all pairs in S for each addition is O(|S|^2).
    # A better way: Precompute for each x in [1, N] the set of "forbidden" pairs?
    # Or simply: given S (as a set) and candidate x, check if x conflicts.
    
    def is_valid_addition(S, x):
        """Check if x can be added to S without creating a 3-AP."""
        # We need to ensure no 3-AP contains x.
        # The 3-AP is (y1, y2, y3) with y1+y3=2*y2.
        # If x is y1: y1=x. Need y2, y3 in S such that x + y3 = 2*y2 => y3 = 2*y2 - x.
        #   Also y1 < y2 < y3 => x < y2 < 2*y2-x => x < y2 and y2 < x + (y2-x) ... 
        #   Actually, just check y2 in S and y3 in S.
        for y2 in S:
            if y2 <= x: continue
            y3 = 2 * y2 - x
            if y3 in S:
                return False
        
        # If x is y2: y2=x. Need y1, y3 in S such that y1 + y3 = 2*x.
        #   y1 < x < y3.
        for y1 in S:
            if y1 >= x: continue
            y3 = 2 * x - y1
            if y3 in S:
                return False
        
        # If x is y3: y3=x. Need y1, y2 in S such that y1 + x = 2*y2 => y1 = 2*y2 - x.
        #   y1 < y2 < x.
        for y2 in S:
            if y2 >= x: continue
            y1 = 2 * y2 - x
            if y1 in S:
                return False
        
        return True

    def remove_and_check(S, x):
        """
        Check if removing x from S is valid (it always is, removing elements never creates APs).
        But we need to know if we can add a new element y after removing x.
        """
        pass

    # Initial construction: Greedy
    # The seed program does:
    # for x in 1..256: if no AP with existing, add x.
    # This tends to pick small numbers. Small numbers have large reciprocals.
    # So the greedy set is already quite good.
    
    S = set()
    for x in range(1, MAX_VAL + 1):
        if is_valid_addition(S, x):
            S.add(x)
    
    current_score = sum(1.0 / x for x in S)
    best_S = S.copy()
    best_score = current_score
    
    # Local Search
    # Try to swap an element x in S with an element y not in S (y > x usually, to improve score?
    # Wait, 1/x > 1/y if x < y. So we want to REMOVE large elements and ADD small elements.
    # But small elements are likely already in S.
    # So we want to REMOVE elements x in S and ADD elements y not in S such that 1/y > 1/x?
    # That implies y < x. But if y < x and y is not in S, it's because it conflicted with S.
    # By removing x, we might free up space for y.
    # So the move is: Remove x, Add y.
    # Condition: y < x (to improve score) and is_valid_addition(S - {x}, y) is True.
    # Also, we must ensure S - {x} is valid (it is).
    
    # Let's identify candidates for removal (large x) and addition (small y).
    # To be efficient, we can precompute for each y not in S, which x's in S block it?
    # That's complex. Let's just iterate.
    
    # Optimization:
    # 1. Sort S in descending order (candidates for removal).
    # 2. Sort [1..256] \ S in ascending order (candidates for addition).
    # 3. For each x in S (largest first), try to remove it.
    # 4. For each y in [1..256]\S (smallest first), try to add it to S\{x}.
    # 5. If valid and 1/y > 1/x (i.e., y < x), accept the swap?
    #    Actually, we should check if the new score is better.
    #    New score = current_score - 1/x + 1/y.
    #    We want 1/y - 1/x > 0 => y < x.
    
    # We can do multiple passes.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort candidates
        S_list = sorted(S, reverse=True) # Largest first
        not_in_S = sorted([i for i in range(1, MAX_VAL + 1) if i not in S]) # Smallest first
        
        for x in S_list:
            if time.time() > deadline: break
            if x not in S: continue # might have been swapped in previous iteration
            
            # Try to remove x
            S_minus_x = S - {x}
            
            # Try to add a smaller y
            for y in not_in_S:
                if time.time() > deadline: break
                if y >= x: break # We only want y < x to improve score
                if y in S: continue # y should not be in S
                
                if is_valid_addition(S_minus_x, y):
                    # Check if swap improves score
                    # 1/y > 1/x is guaranteed if y < x
                    # Perform swap
                    S = S_minus_x | {y}
                    current_score += (1.0 / y - 1.0 / x)
                    improved = True
                    # Update best if better
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                    # After a swap, the lists are stale. Break inner loop and restart outer?
                    # Or just continue? Continuing with stale lists might miss things or be inefficient.
                    # Let's break and restart the while loop to refresh lists.
                    break
            if improved:
                break
        
    # Final polish: Try adding any y not in S (even if y > x, i.e. decreasing score? No, we maximize).
    # Wait, can we add an element WITHOUT removing one?
    # If S is not maximal, we can add y.
    # Greedy construction usually produces a maximal set? Not necessarily.
    # Let's try to add any valid y.
    # If we can add y, score increases (since 1/y > 0).
    
    # Check for additions
    for y in range(1, MAX_VAL + 1):
        if time.time() > deadline: break
        if y not in S:
            if is_valid_addition(S, y):
                S.add(y)
                current_score += 1.0 / y
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
    
    # One more pass of swaps after additions?
    # Maybe, but time is tight.
    
    # Ensure we return the best set found
    return list(best_S)

# EVOLVE-BLOCK-END
