"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 50  # 50s safety margin before 60s limit
    
    # Try to find minimal number of distinct denominators >= 5 summing to 1
    # Start with small counts and increase
    
    best = None
    
    # For each count k from 3 upward, try to find k distinct denominators >= 5
    # whose reciprocals sum to 1
    
    def search_k(k, min_den=5, start_idx=0, path=None, current_sum=None):
        nonlocal best, deadline
        if path is None:
            path = []
        if current_sum is None:
            current_sum = Fraction(0, 1)
        
        if time.time() > deadline:
            return None
        
        if len(path) == k:
            # Check if sum of reciprocals is 1
            if current_sum == 1:
                return list(path)
            return None
        
        # Pruning: what's the max we can still add?
        remaining = k - len(path)
        # Each new term must be > last term in path
        last = path[-1] if path else min_den - 1
        
        # The remaining sum needed:
        needed = 1 - current_sum
        
        if needed <= 0:
            return None
        
        # Max sum we can get from 'remaining' terms starting after 'last'
        max_sum = Fraction(0, 1)
        for i in range(remaining):
            max_sum += Fraction(1, last + 1 + i)
        
        if needed > max_sum:
            return None
        
        # Try next denominator
        # Compute a practical upper bound for the next denominator
        # The next denominator d must satisfy:
        # 1/d + (sum of remaining-1 terms with denominators > d) = needed
        # The maximum sum of (remaining-1) terms with denominators > d is:
        # 1/(d+1) + 1/(d+2) + ... + 1/(d+remaining-1)
        # So we need: 1/d + max_sum_remaining_minus_1 >= needed
        # And: 1/d <= needed (since all other terms are positive)
        
        # For the lower bound on d: 1/d <= needed → d >= 1/needed
        # For the upper bound on d: 1/d + sum_{i=1}^{remaining-1} 1/(d+i) >= needed
        # This is approximately 1/d + (remaining-1)/d >= needed → d <= remaining/needed
        
        # Convert needed to float for bound estimation
        needed_f = float(needed)
        if needed_f < 1e-15:
            return None
        
        # Lower bound for d: d >= ceil(1/needed)
        d_min = max(last + 1, int(1 / needed_f) + 1)
        
        # Upper bound for d: use the approximation d <= remaining/needed
        # But be more precise: 1/d + 1/(d+1) + ... + 1/(d+remaining-1) >= needed
        # Let's estimate d_max by trying values
        d_max_est = int(remaining / needed_f) + 2
        if d_max_est < d_min:
            d_max_est = d_min
        # Cap to avoid huge searches
        if d_max_est > 100000:
            d_max_est = 100000
        
        for d in range(d_min, min(d_max_est + 1, 10000001)):
            if time.time() > deadline:
                return None
            
            # Check if 1/d <= needed (necessary)
            if Fraction(1, d) > needed:
                continue
            
            # Check if it's possible to complete with remaining-1 terms
            # The max sum of remaining-1 terms with denominators > d is:
            # 1/(d+1) + 1/(d+2) + ... + 1/(d+remaining-1)
            if remaining > 1:
                max_remaining_sum = Fraction(0, 1)
                for i in range(1, remaining):
                    max_remaining_sum += Fraction(1, d + i)
                if Fraction(1, d) + max_remaining_sum < needed:
                    continue
            
            result = search_k(k, min_den, 0, path + [d], current_sum + Fraction(1, d))
            if result is not None:
                return result
        
        return None
    
    # Try k = 3 first
    for k in range(3, 9):
        if time.time() > deadline:
            break
        result = search_k(k)
        if result is not None:
            best = result
            break
        if time.time() > deadline:
            break
    
    if best is None:
        # Fallback: use the DP approach from the seed
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
