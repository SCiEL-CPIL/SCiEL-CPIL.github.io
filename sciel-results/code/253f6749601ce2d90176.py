import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Helper to check if a list of denominators sums to 1
    def is_valid(denoms):
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == 1

    # We established that k=8 is the minimum possible count.
    # We perform a DFS to find a valid set of 8 denominators.
    
    def dfs(current_sum, current_max_denom, remaining_terms, path):
        if time.time() > deadline:
            return None
        
        if remaining_terms == 0:
            if current_sum == 1:
                return list(path)
            return None
        
        # Max possible sum with remaining_terms
        # Smallest available denominators: current_max_denom + 1, ...
        max_possible = Fraction(0, 1)
        for i in range(1, remaining_terms + 1):
            max_possible += Fraction(1, current_max_denom + i)
        
        if current_sum + max_possible < 1:
            return None
            
        # Min possible sum (to ensure we don't overshoot 1 with smallest denoms)
        # Actually, we just iterate d.
        
        # Determine range for next denominator d
        # 1/d < 1 - current_sum  => d > 1 / (1 - current_sum)
        rem = 1 - current_sum
        if rem <= 0:
            return None
            
        d_min = max(current_max_denom + 1, int(1 / float(rem)) + 1)
        
        # 1/d > rem / remaining_terms => d < remaining_terms / rem
        d_max = int(remaining_terms / float(rem))
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
                
            new_sum = current_sum + Fraction(1, d)
            if new_sum > 1:
                # Since d is increasing, 1/d is decreasing, so new_sum will only decrease
                # But we break if new_sum > 1? No, if d_min gives sum > 1, larger d gives smaller sum.
                # So we should continue.
                pass
            
            path.append(d)
            res = dfs(new_sum, d, remaining_terms - 1, path)
            path.pop()
            
            if res is not None:
                return res
                
        return None

    # Start DFS for k=8
    # Initial state: sum=0, max_denom=4, remaining=8, path=[]
    result = dfs(Fraction(0, 1), 4, 8, [])
    
    if result is not None and is_valid(result):
        return result
        
    # Fallback: Pure Greedy (just in case, though k=8 should be found)
    R = Fraction(1, 1)
    d_prev = 4
    denoms = []
    while R > 0:
        if time.time() > deadline:
            break
        d_min = max(d_prev + 1, int(1 / float(R)) + 1)
        d = d_min
        denoms.append(d)
        R -= Fraction(1, d)
        d_prev = d
        
    if is_valid(denoms):
        return denoms
        
    raise RuntimeError("No valid solution found")
