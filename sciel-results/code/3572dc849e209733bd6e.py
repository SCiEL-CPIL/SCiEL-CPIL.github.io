"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 3-AP-free subset of [1, 256] maximizing the sum of reciprocals.
    Strategy:
    1. Use base-11 digit representation: select numbers with no zero digits in base 11.
       This guarantees no 3-term AP by the combinatorial property.
    2. Apply local search to swap elements for better reciprocal weight.
    3. Use deadline guard for 60s limit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    N = 256
    MAX_VAL = N
    
    # Helper to check if adding x to set S creates a 3-AP
    def is_valid_addition(S, x):
        """Check if x can be added to S without creating a 3-AP."""
        # If x is the first term: x, y2, y3 with y2,y3 in S
        for y2 in S:
            if y2 <= x: continue
            y3 = 2 * y2 - x
            if y3 in S:
                return False
        
        # If x is the middle term: y1, x, y3 with y1,y3 in S
        for y1 in S:
            if y1 >= x: continue
            y3 = 2 * x - y1
            if y3 in S:
                return False
        
        # If x is the last term: y1, y2, x with y1,y2 in S
        for y2 in S:
            if y2 >= x: continue
            y1 = 2 * y2 - x
            if y1 in S:
                return False
        
        return True

    # Base-11 construction: select numbers with no zero digits in base 11
    # A number n has no zero digits in base 11 if all digits are in {1,...,10}
    def has_no_zero_digit_base11(n):
        """Check if n has no zero digits when written in base 11."""
        if n == 0:
            return False
        while n > 0:
            digit = n % 11
            if digit == 0:
                return False
            n //= 11
        return True

    # Initial construction using base-11 property
    S = set()
    for x in range(1, MAX_VAL + 1):
        if has_no_zero_digit_base11(x):
            S.add(x)
    
    current_score = sum(1.0 / x for x in S)
    best_S = S.copy()
    best_score = current_score
    
    # Local Search: try to swap elements to improve reciprocal sum
    # We want to remove large elements and add small ones (higher reciprocal)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Sort candidates for removal (largest first) and addition (smallest first)
        S_list = sorted(S, reverse=True)
        not_in_S = sorted([i for i in range(1, MAX_VAL + 1) if i not in S])
        
        for x in S_list:
            if time.time() > deadline: break
            if x not in S: continue
            
            # Try to remove x and add a smaller y
            S_minus_x = S - {x}
            
            for y in not_in_S:
                if time.time() > deadline: break
                if y >= x: break  # Only want y < x to improve score
                
                if y in S: continue
                
                if is_valid_addition(S_minus_x, y):
                    # Perform swap
                    S = S_minus_x | {y}
                    current_score += (1.0 / y - 1.0 / x)
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                    
                    break
            if improved:
                break
    
    # Final polish: try adding any valid y
    for y in range(1, MAX_VAL + 1):
        if time.time() > deadline: break
        if y not in S:
            if is_valid_addition(S, y):
                S.add(y)
                current_score += 1.0 / y
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
    
    return list(best_S)

# EVOLVE-BLOCK-END

