"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    def verify(denoms):
        """Verify that denoms are distinct, in [5, 10000000], and sum of reciprocals is 1."""
        if not denoms:
            return False
        if len(denoms) != len(set(denoms)):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == 1
    
    def dfs_search(num, den, last_d, remaining_depth, path, max_branch=400):
        """
        DFS with aggressive pruning.
        """
        if time.time() > deadline:
            return None
        
        if num == 0:
            return path if remaining_depth >= 0 else None
        
        if remaining_depth == 0:
            return None
        
        # Minimum denominator
        min_d = max(last_d + 1, math.ceil(den / num))
        if min_d < 5:
            min_d = 5
        
        # Maximum denominator for feasibility
        max_d = min(10000000, (den * remaining_depth) // num + 1)
        if max_d < min_d:
            return None
        
        # Pruning: max possible sum with remaining_depth terms starting from min_d
        if remaining_depth * den < num * min_d + 1:
            return None
        
        count = 0
        for d in range(min_d, min_d + max_branch):
            if d > max_d or d > 10000000:
                break
            if d < 5:
                continue
            
            if d * num < den:
                continue
            
            new_num = num * d - den
            if new_num < 0:
                break
            
            if new_num == 0:
                if remaining_depth == 1:
                    return path + [d]
                else:
                    continue
            
            new_den = den * d
            g = math.gcd(new_num, new_den)
            rn = new_num // g
            rd = new_den // g
            
            # Pruning
            if (remaining_depth - 1) * rd < rn * (d + 1) + 1:
                continue
            
            count += 1
            if count > max_branch:
                break
            
            result = dfs_search(rn, rd, d, remaining_depth - 1, path + [d], max_branch)
            if result is not None:
                return result
        
        return None
    
    def try_k_terms(k):
        """Try to find a solution with exactly k terms."""
        result = dfs_search(1, 1, 4, k, [], max_branch=500)
        if result is not None and verify(result):
            return result
        return None
    
    # Try to find minimal solution with DFS
    for k in range(10, 16):
        if time.time() > deadline - 8:
            break
        result = try_k_terms(k)
        if result is not None:
            return result
    
    # Fallback: greedy Egyptian fraction construction
    # This is guaranteed to produce a valid solution
    def greedy_construct():
        denoms = []
        target = Fraction(1, 1)
        used = set()
        min_d = 5
        
        while target > 0 and len(denoms) < 256:
            if target == 0:
                break
            
            # Find smallest d >= min_d such that 1/d <= target
            # 1/d <= target => d >= 1/target
            d = max(min_d, math.ceil(1.0 / float(target)) + 1)
            if d < 5:
                d = 5
            
            # Ensure 1/d <= target and d not used
            while d in used or Fraction(1, d) > target:
                d += 1
            
            if d > 10000000:
                break
            
            denoms.append(d)
            used.add(d)
            min_d = d + 1
            target -= Fraction(1, d)
        
        if target == 0 and len(denoms) <= 256 and verify(denoms):
            return denoms
        return None
    
    result = greedy_construct()
    if result is not None:
        return result
    
    raise RuntimeError("No valid solution found")
# EVOLVE-BLOCK-END
