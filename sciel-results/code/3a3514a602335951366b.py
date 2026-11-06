import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    
    # Initial best solution
    best_solution = [8, 31, 248]
    best_max_denom = 248
    
    start_time = time.time()
    time_limit = 55.0

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # We want to minimize c.
    # If c < best_max_denom, then b < best_max_denom and a < best_max_denom.
    # So we only need to search a up to best_max_denom - 2.
    
    # We can iterate a from 7 to best_max_denom - 2
    # But best_max_denom changes.
    # Let's just iterate a up to a reasonable limit, e.g., 1000, and refine.
    # Or simply:
    # For a in range(7, 1000):
    #   ...
    
    # Let's start with a high limit and tighten as we find better solutions.
    # Initial best is 248.
    # Let's search for c < 248.
    
    for a in range(7, 248):
        if time.time() - start_time > time_limit:
            break
            
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
            
        # 1/b + 1/c = rem1
        # b > 1/rem1
        # 2/b > rem1 => b < 2/rem1
        b_low = int(1 / rem1) + 1
        b_high = int(2 / rem1)
        
        # Also b > a
        b_low = max(b_low, a + 1)
        
        # Also c > b, so 1/c < 1/b => rem1 - 1/b < 1/b => rem1 < 2/b => b < 2/rem1 (already handled)
        # And c < best_max_denom => 1/(rem1 - 1/b) < best_max_denom
        # rem1 - 1/b > 1/best_max_denom
        # 1/b < rem1 - 1/best_max_denom
        # b > 1 / (rem1 - 1/best_max_denom)
        
        if rem1 > 1 / best_max_denom:
            b_low_c = int(1 / (rem1 - 1 / best_max_denom)) + 1
            b_low = max(b_low, b_low_c)
        else:
            # If rem1 <= 1/best_max_denom, then 1/b + 1/c <= 2/b <= 2/b_low.
            # If 2/b_low <= rem1, no solution with c < best_max_denom?
            # Actually if rem1 is very small, b must be large.
            # If b >= best_max_denom, then c > b >= best_max_denom, so no improvement.
            # So we can skip if b_low >= best_max_denom.
            pass
            
        if b_low >= best_max_denom:
            continue
            
        if b_low > b_high:
            continue
            
        # Iterate b
        for b in range(b_low, b_high + 1):
            if time.time() - start_time > time_limit:
                break
                
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
                
            # c = 1 / rem2
            if rem2.denominator % rem2.numerator == 0:
                c = rem2.denominator // rem2.numerator
                
                if c <= b:
                    continue
                    
                if c < best_max_denom:
                    best_solution = [a, b, c]
                    best_max_denom = c
                    
    if verify(best_solution):
        return best_solution
        
    return [8, 31, 248]
