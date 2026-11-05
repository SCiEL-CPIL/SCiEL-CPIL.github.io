"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50  # 50s safety margin before 60s limit
    
    # Pre-computed minimal Egyptian fraction representations for 1 with denominators > 4
    # 2-term: impossible (1/a + 1/b = 1 => (a-1)(b-1)=1 => a=b=2, both < 5)
    # 3-term: 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5 ≠ 1
    # Let me verify: 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5. Not 1.
    # 
    # Let me find the actual minimal representation:
    # 1/5 + 1/6 + 1/7 + 1/42 = 42/210 + 35/210 + 30/210 + 5/210 = 112/210 = 8/15. Not 1.
    # 
    # Let me think more carefully. We need sum of 1/d_i = 1 with d_i >= 5, distinct.
    # 
    # Try 3 terms: 1/a + 1/b + 1/c = 1, 5 <= a < b < c
    # For a=5: 1/b + 1/c = 4/5. (b+c)/(bc) = 4/5 => 5(b+c) = 4bc => 4bc - 5b - 5c = 0
    # Multiply by 4: 16bc - 20b - 20c = 0 => (4b-5)(4c-5) = 25
    # So 4b-5 and 4c-5 are positive factors of 25: (1,25), (5,5), (25,1)
    # (4b-5, 4c-5) = (1,25): b=6/4=1.5 (not int)
    # (4b-5, 4c-5) = (5,5): b=10/4=2.5 (not int)
    # (4b-5, 4c-5) = (25,1): b=30/4=7.5 (not int)
    # No integer solutions for a=5.
    #
    # For a=6: 1/b + 1/c = 5/6. (b+c)/(bc) = 5/6 => 6(b+c) = 5bc => 5bc - 6b - 6c = 0
    # Multiply by 5: 25bc - 30b - 30c = 0 => (5b-6)(5c-6) = 36
    # Factors of 36: (1,36),(2,18),(3,12),(4,9),(6,6),...
    # (5b-6,5c-6)=(1,36): b=7/5=1.4 (no)
    # (5b-6,5c-6)=(2,18): b=8/5=1.6 (no)
    # (5b-6,5c-6)=(3,12): b=9/5=1.8 (no)
    # (5b-6,5c-6)=(4,9): b=10/5=2, c=15/5=3 (both < 5, invalid)
    # (5b-6,5c-6)=(6,6): b=12/5=2.4 (no)
    # No valid solutions for a=6.
    #
    # For a=7: 1/b + 1/c = 6/7. 7(b+c)=6bc => 6bc-7b-7c=0
    # (6b-7)(6c-7)=49. Factors: (1,49),(7,7),(49,1)
    # (1,49): b=8/6=4/3 (no)
    # (7,7): b=14/6=7/3 (no)
    # (49,1): b=56/6=28/3 (no)
    # No solutions.
    #
    # For a=8: 1/b+1/c=7/8. 8(b+c)=7bc => 7bc-8b-8c=0
    # (7b-8)(7c-8)=64. Factors of 64: (1,64),(2,32),(4,16),(8,8),...
    # (1,64): b=9/7 (no)
    # (2,32): b=10/7 (no)
    # (4,16): b=12/7 (no)
    # (8,8): b=16/7 (no)
    # No solutions.
    #
    # For a=9: 1/b+1/c=8/9. 9(b+c)=8bc => 8bc-9b-9c=0
    # (8b-9)(8c-9)=81. Factors: (1,81),(3,27),(9,9),...
    # (1,81): b=10/8=5/4 (no)
    # (3,27): b=12/8=3/2 (no)
    # (9,9): b=18/8=9/4 (no)
    # No solutions.
    #
    # For a=10: 1/b+1/c=9/10. 10(b+c)=9bc => 9bc-10b-10c=0
    # (9b-10)(9c-10)=100. Factors: (1,100),(2,50),(4,25),(5,20),(10,10),...
    # (1,100): b=11/9 (no)
    # (2,50): b=12/9=4/3 (no)
    # (4,25): b=14/9 (no)
    # (5,20): b=15/9=5/3 (no)
    # (10,10): b=20/9 (no)
    # No solutions.
    #
    # So no 3-term solution exists with all denominators >= 5? Let me try a different approach.
    # Actually, let me just search computationally.
    
    best = None
    
    # For each count k from 3 upward, try to find k distinct denominators >= 5
    # whose reciprocals sum to 1
    
    def search_k(k, path=None, current_sum=None):
        nonlocal best, deadline
        if path is None:
            path = []
        if current_sum is None:
            current_sum = Fraction(0, 1)
        
        if time.time() > deadline:
            return None
        
        if len(path) == k:
            if current_sum == 1:
                return list(path)
            return None
        
        remaining = k - len(path)
        last = path[-1] if path else 4
        
        needed = 1 - current_sum
        
        if needed <= 0:
            return None
        
        # Max sum we can get from 'remaining' terms starting after 'last'
        max_sum = Fraction(0, 1)
        for i in range(remaining):
            max_sum += Fraction(1, last + 1 + i)
        
        if needed > max_sum:
            return None
        
        # Bound the next denominator
        needed_f = float(needed)
        if needed_f < 1e-15:
            return None
        
        d_min = max(last + 1, int(1 / needed_f) + 1)
        d_max_est = int(remaining / needed_f) + 2
        if d_max_est < d_min:
            d_max_est = d_min
        if d_max_est > 100000:
            d_max_est = 100000
        
        for d in range(d_min, min(d_max_est + 1, 10000001)):
            if time.time() > deadline:
                return None
            
            if Fraction(1, d) > needed:
                continue
            
            if remaining > 1:
                max_remaining_sum = Fraction(0, 1)
                for i in range(1, remaining):
                    max_remaining_sum += Fraction(1, d + i)
                if Fraction(1, d) + max_remaining_sum < needed:
                    continue
            
            result = search_k(k, path + [d], current_sum + Fraction(1, d))
            if result is not None:
                return result
        
        return None
    
    # Try k = 3 first, then 4, etc.
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = search_k(k)
        if result is not None:
            best = result
            break
        if time.time() > deadline:
            break
    
    if best is None:
        # Fallback: use the DP approach
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best = reached[target]
                break
    
    if best is None:
        raise RuntimeError("no representation found")
    
    return best

# EVOLVE-BLOCK-END
