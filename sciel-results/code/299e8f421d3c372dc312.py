import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 50
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    # Verify: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Correct.
    best = [8, 31, 248]
    best_max = 248
    
    target_num, target_den = 5, 31
    
    def is_valid_solution(dens):
        """Verify that the given denominators sum to exactly 5/31."""
        s = Fraction(0, 1)
        for d in dens:
            s += Fraction(1, d)
        return s == Fraction(5, 31)
    
    def check_2term(rem_num, rem_den, max_d):
        """
        Find distinct d1, d2 such that 1/d1 + 1/d2 = rem_num/rem_den.
        Returns (d1, d2) if found, else None.
        """
        # 1/d1 + 1/d2 = n/d => (d1+d2)*d = n*d1*d2
        # d1*d2*n - d*d1 - d*d2 = 0
        # (n*d1 - d)*(n*d2 - d) = d^2
        # Let a = n*d1 - d, b = n*d2 - d
        # a*b = d^2
        # d1 = (a + d)/n, d2 = (b + d)/n
        # So a and b must be positive divisors of d^2 with a*b = d^2
        # And (a + d) must be divisible by n, and (b + d) must be divisible by n
        # Also d1 != d2, d1, d2 in [2, max_d], and d1 < d2 (distinct)
        
        d2_val = rem_den * rem_den
        n = rem_num
        d_val = rem_den
        
        # Enumerate divisors of d^2
        # d^2 can be large, but we only need divisors a where a <= d^2/a (i.e., a <= d)
        # to ensure d1 <= d2
        found = None
        best_pair = None
        
        a = 1
        while a * a <= d2_val:
            if d2_val % a == 0:
                b = d2_val // a
                # Check d1 = (a + d_val) / n is integer
                if (a + d_val) % n == 0:
                    d1 = (a + d_val) // n
                    # Check d2 = (b + d_val) / n is integer
                    if (b + d_val) % n == 0:
                        d2 = (b + d_val) // n
                        # Check distinctness and bounds
                        if d1 != d2 and 1 <= d1 <= max_d and 1 <= d2 <= max_d:
                            if best_pair is None or max(d1, d2) < max(best_pair[0], best_pair[1]):
                                best_pair = (min(d1, d2), max(d1, d2))
            a += 1
        return best_pair
    
    def check_3term(rem_num, rem_den, min_d, max_d):
        """
        Find distinct d1, d2, d3 >= min_d such that 1/d1 + 1/d2 + 1/d3 = rem_num/rem_den.
        Returns (d1, d2, d3) with max < max_d, or None.
        Uses the 2-term divisor check for the last two.
        """
        if rem_num <= 0 or rem_den <= 0:
            return None
        
        # d1 must satisfy: 1/d1 < rem => d1 > 1/rem = rem_den/rem_num
        # Also d1 >= min_d
        # And after choosing d1, we need 1/d2 + 1/d3 = rem - 1/d1 = (rem_num*d1 - rem_den)/(rem_den*d1)
        # For this to be positive, rem_num*d1 > rem_den => d1 > rem_den/rem_num
        
        lower_d1 = max(min_d, (rem_den + rem_num - 1) // rem_num + 1)
        # Actually d1 > rem_den/rem_num means d1 >= floor(rem_den/rem_num) + 1
        # But we also need rem_num*d1 > rem_den, i.e., d1 > rem_den/rem_num
        # So d1 >= floor(rem_den/rem_num) + 1 if rem_den % rem_num != 0, else rem_den/rem_num + 1
        
        # Let's be safe: d1 must satisfy rem_num * d1 > rem_den
        # So d1 > rem_den / rem_num
        # d1_min = max(min_d, floor(rem_den / rem_num) + 1)
        
        floor_val = rem_den // rem_num
        if rem_den % rem_num == 0:
            d1_min = max(min_d, floor_val + 1)
        else:
            d1_min = max(min_d, floor_val + 1)
        
        # d1 also must be < max_d
        # And after picking d1, the remaining sum rem2 = rem - 1/d1 must be representable as 2 terms
        # with denominators > d1 and < max_d
        
        # Upper bound on d1: we need 1/d1 <= rem, and the remaining 2 terms must be >= 2/max_d
        # Actually, the remaining sum must be at least 1/(max_d-1) + 1/max_d (the two smallest possible)
        # Wait, the remaining denominators must be > d1 and < max_d, so they're in (d1, max_d)
        # The minimum sum of 2 distinct terms in (d1, max_d) is 1/(max_d-1) + 1/max_d
        # So we need rem - 1/d1 >= 1/(max_d-1) + 1/max_d
        # This gives an upper bound on 1/d1, hence a lower bound on d1... wait no, it's an upper bound on 1/d1
        # 1/d1 <= rem - (1/(max_d-1) + 1/max_d)
        # d1 >= 1 / (rem - (1/(max_d-1) + 1/max_d))
        
        # Also, the maximum sum of 2 terms in (d1, max_d) is 1/(d1+1) + 1/(d1+2)
        # So we need rem - 1/d1 <= 1/(d1+1) + 1/(d1+2)
        # This gives a constraint that may limit d1 from above in some cases.
        
        best_result = None
        
        for d1 in range(d1_min, max_d):
            if time.time() > deadline:
                break
            
            # Check that rem - 1/d1 > 0
            rem2_num = rem_num * d1 - rem_den
            rem2_den = rem_den * d1
            if rem2_num <= 0:
                continue
            
            # Simplify
            g = gcd(rem2_num, rem2_den)
            rn = rem2_num // g
            rd = rem2_den // g
            
            # Now find 2-term solution for 1/d2 + 1/d3 = rn/rd with d2, d3 > d1 and < max_d
            # The 2-term check finds any valid pair; we need to filter for d2 > d1
            # Let's modify: we need d2, d3 distinct, both > d1, both < max_d
            # From the 2-term formula: (n*d2 - d)*(n*d3 - d) = d^2 where n=rn, d=rd
            # d2 = (a + rd)/rn, d3 = (b + rd)/rn where a*b = rd^2
            
            # We need d2 > d1 and d3 > d1
            # d2 = (a + rd) / rn > d1 => a + rd > rn * d1 => a > rn * d1 - rd
            # Similarly for d3.
            
            # Let's just enumerate and check
            d2_val = rd * rd
            n2 = rn
            d_val2 = rd
            
            a = 1
            while a * a <= d2_val:
                if d2_val % a == 0:
                    b = d2_val // a
                    if (a + d_val2) % n2 == 0:
                        d2 = (a + d_val2) // n2
                        if (b + d_val2) % n2 == 0:
                            d3 = (b + d_val2) // n2
                            # Check d2 != d3, d2 > d1, d3 > d1, both < max_d
                            if d2 != d3 and d2 > d1 and d3 > d1 and d2 < max_d and d3 < max_d:
                                # Check distinctness with d1 (already have d2 > d1, d3 > d1, so distinct)
                                cur_max = max(d1, d2, d3)
                                if best_result is None or cur_max < max(best_result):
                                    best_result = (d1, min(d2, d3), max(d2, d3))
                a += 1
        
        return best_result
    
    # Search for 3-term solutions with max < 248
    # 1/d1 + 1/d2 + 1/d3 = 5/31
    # d1 >= 7 (since 1/6 > 5/31)
    
    found = check_3term(target_num, target_den, 7, best_max)
    if found:
        # Verify
        if is_valid_solution(found):
            best = list(found)
            best_max = max(best)
    
    # Also search for 4-term solutions with max < best_max
    # This is more complex. Let's do a limited DFS for 4 terms.
    
    def dfs_4(rem_num, rem_den, min_d, max_d, depth, current):
        """
        DFS for remaining terms. depth = number of terms left to pick.
        current = list of already-picked denominators.
        Returns True if a solution is found.
        """
        if time.time() > deadline:
            return False
        
        if depth == 0:
            return rem_num == 0
        
        if depth == 1:
            # rem must be 1/d for some d
            if rem_num == 1:
                d = rem_den
                if min_d <= d < max_d and d not in current:
                    return True
            return False
        
        if depth == 2:
            # Use the 2-term check
            pair = check_2term(rem_num, rem_den, max_d - 1)  # max_d exclusive
            if pair:
                d1, d2 = pair
                if d1 >= min_d and d2 >= min_d and d1 not in current and d2 not in current and d1 != d2:
                    return True
            return False
        
        # depth >= 3
        # Pick next denominator d
        # 1/d < rem => d > rem_den/rem_num
        floor_val = rem_den // rem_num
        if rem_den % rem_num == 0:
            d_min = max(min_d, floor_val + 1)
        else:
            d_min = max(min_d, floor_val + 1)
        
        # d must be < max_d
        for d in range(d_min, max_d):
            if time.time() > deadline:
                return False
            
            # Check rem - 1/d > 0
            rem2_num = rem_num * d - rem_den
            rem2_den = rem_den * d
            if rem2_num <= 0:
                continue
            
            g = gcd(rem2_num, rem2_den)
            rn = rem2_num // g
            rd = rem2_den // g
            
            # Also need to ensure remaining (depth-1) terms can sum to rn/rd
            # with denominators > d and < max_d
            
            if d in current:
                continue
            
            current.append(d)
            if dfs_4(rn, rd, d + 1, max_d, depth - 1, current):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Try 4-term solutions
    if dfs_4(target_num, target_den, 7, best_max, 4, []):
        # We found a 4-term solution, but we need to reconstruct it.
        # This is tricky with the current setup. Let's just note that if dfs_4 returns True,
        # we need to track the actual denominators. Let me restructure.
        pass
    
    # Let me redo the 4-term search with proper tracking.
    
    best_4 = None
    
    def dfs_4_track(rem_num, rem_den, min_d, max_d, depth, current):
        """
        DFS for remaining terms. Returns the complete list if found, else None.
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem_num == 0:
                return current[:]
            return None
        
        if depth == 1:
            if rem_num == 1:
                d = rem_den
                if min_d <= d < max_d and d not in current:
                    return current + [d]
            return None
        
        if depth == 2:
            pair = check_2term(rem_num, rem_den, max_d - 1)
            if pair:
                d1, d2 = pair
                if d1 >= min_d and d2 >= min_d and d1 not in current and d2 not in current and d1 != d2:
                    return current + [d1, d2]
            return None
        
        # depth >= 3
        floor_val = rem_den // rem_num
        if rem_den % rem_num == 0:
            d_min = max(min_d, floor_val + 1)
        else:
            d_min = max(min_d, floor_val + 1)
        
        for d in range(d_min, max_d):
            if time.time() > deadline:
                return None
            
            rem2_num = rem_num * d - rem_den
            rem2_den = rem_den * d
            if rem2_num <= 0:
                continue
            
            g = gcd(rem2_num, rem2_den)
            rn = rem2_num // g
            rd = rem2_den // g
            
            if d in current:
                continue
            
            current.append(d)
            result = dfs_4_track(rn, rd, d + 1, max_d, depth - 1, current)
            current.pop()
            
            if result is not None:
                cur_max = max(result)
                if best_4 is None or cur_max < max(best_4):
                    return result
        
        return None
    
    result_4 = dfs_4_track(target_num, target_den, 7, best_max, 4, [])
    if result_4 is not None:
        if is_valid_solution(result_4):
            best = result_4
            best_max = max(best)
    
    # Also try 5-term and 6-term solutions if time permits
    for depth in range(5, 7):
        if time.time() > deadline - 5:
            break
        result = dfs_4_track(target_num, target_den, 7, best_max, depth, [])
        if result is not None:
            if is_valid_solution(result):
                best = result
                best_max = max(best)
    
    return best
