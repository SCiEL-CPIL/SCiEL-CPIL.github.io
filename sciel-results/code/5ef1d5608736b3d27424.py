import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    MIN_D = 5
    MAX_D = 10_000_000

    # We need at least 8 distinct denominators >= 5 for sum of reciprocals = 1
    # (max sum of 7 smallest: 1/5+1/6+...+1/11 < 1, but 8 smallest: +1/12 > 1)
    
    # Strategy: DFS with strong pruning using exact fractions
    # Try to find minimal number of terms (start with 8)
    
    def dfs(start, remaining, count, chosen, target_count):
        """
        Find target_count distinct denominators >= start (increasing)
        such that sum of reciprocals = remaining.
        """
        if time.time() > deadline:
            return None
        
        if count == target_count:
            if remaining == 0:
                return list(chosen)
            return None
        
        # Pruning: check if max possible sum with remaining terms is enough
        # Max sum: 1/start + 1/(start+1) + ... + 1/(start + (target_count - count - 1))
        max_possible = Fraction(0)
        for k in range(target_count - count):
            max_possible += Fraction(1, start + k)
        if max_possible < remaining:
            return None
        
        # Pruning: check if min possible sum with remaining terms is small enough
        # Min sum: use largest possible denominators, but we don't have a tight upper bound
        # Skip this for now
        
        # Determine range of candidates for the next denominator
        # We need 1/d <= remaining, so d >= ceil(1/remaining)
        if remaining > 0:
            min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        else:
            min_d = start
        
        min_d = max(min_d, start)
        
        # Upper bound: even with the largest remaining terms, we can't exceed remaining
        # If we pick d, the remaining (target_count - count - 1) terms must sum to remaining - 1/d
        # The max sum of (target_count - count - 1) terms starting from d+1 is:
        # 1/(d+1) + 1/(d+2) + ... + 1/(d + target_count - count - 1)
        # This must be >= remaining - 1/d
        
        # Also, we need 1/d + max_sum_of_rest >= remaining
        # And 1/d + min_sum_of_rest <= remaining (min sum is very small, skip)
        
        # Let's bound d: 
        # 1/d must be <= remaining (already enforced by min_d)
        # Also, remaining - 1/d must be representable as sum of (target_count - count - 1) terms >= d+1
        # Max sum of those terms: sum_{k=1}^{target_count-count-1} 1/(d+k)
        # This must be >= remaining - 1/d
        
        # For efficiency, cap the search range
        # Heuristic: don't search too far
        max_d = min(min_d + 500, 50000)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            term = Fraction(1, d)
            if term > remaining:
                continue
            
            new_remaining = remaining - term
            if new_remaining < 0:
                continue
            
            # If we're at the last term, check if new_remaining == 0
            if count + 1 == target_count:
                if new_remaining == 0:
                    return chosen + [d]
                continue
            
            # Check if new_remaining can be achieved with (target_count - count - 1) terms >= d+1
            # Max sum check
            max_rest = Fraction(0)
            for k in range(1, target_count - count):
                max_rest += Fraction(1, d + k)
            if max_rest < new_remaining:
                # Even the largest possible remaining terms can't reach new_remaining
                # As d increases, max_rest decreases, so we can break
                # Actually, as d increases, 1/d decreases, so new_remaining increases,
                # but max_rest also decreases. Let's just continue.
                continue
            
            chosen.append(d)
            result = dfs(d + 1, new_remaining, count + 1, chosen, target_count)
            if result is not None:
                return result
            chosen.pop()
        
        return None

    # Try increasing number of terms starting from 8
    for n_terms in range(8, 16):
        result = dfs(MIN_D, Fraction(1, 1), 0, [], n_terms)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(set(result)) == n_terms and all(MIN_D <= d <= MAX_D for d in result):
                return result
    
    # Fallback: use a known construction
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... 
    # Let's try a different approach: start with a known identity
    
    # Known: 1 = 1/2 + 1/3 + 1/6, but denominators must be >= 5
    # Split: 1/2 = 1/3 + 1/6, 1/3 = 1/4 + 1/12, etc.
    # This gets complicated. Let's try a greedy-like approach with corrections.
    
    # Alternative: use the fact that 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 
    # = let's compute: 
    # 1/5 = 0.2, 1/6 ≈ 0.1667, 1/7 ≈ 0.1429, 1/8 = 0.125, 1/9 ≈ 0.1111, 
    # 1/10 = 0.1, 1/11 ≈ 0.0909, 1/12 ≈ 0.0833
    # Sum ≈ 1.0199, slightly over 1.
    
    # So 8 terms from 5 to 12 sum to slightly more than 1.
    # We need to adjust. Replace some terms with larger denominators.
    
    # Let's try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/x = 1
    # Sum of first 7: 1/5+1/6+1/7+1/8+1/9+1/10+1/11
    # Let's compute exactly:
    # LCM of 5,6,7,8,9,10,11 = 2^3 * 3^2 * 5 * 7 * 11 = 27720
    # 1/5 = 5544/27720
    # 1/6 = 4620/27720
    # 1/7 = 3960/27720
    # 1/8 = 3465/27720
    # 1/9 = 3080/27720
    # 1/10 = 2772/27720
    # 1/11 = 2520/27720
    # Sum = (5544+4620+3960+3465+3080+2772+2520)/27720 = 25961/27720
    # 1 - 25961/27720 = (27720-25961)/27720 = 1759/27720
    # So 1/x = 1759/27720, x = 27720/1759 ≈ 15.76, not integer.
    
    # Try different combinations. Let's use the DFS but with a wider search.
    
    # Actually, let's just try to find any valid solution quickly.
    # Use a precomputed solution if we can find one.
    
    # Let's try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/x
    # Sum of 1/5+1/6+1/7+1/8+1/9+1/10+1/12:
    # LCM of 5,6,7,8,9,10,12 = 2^3 * 3^2 * 5 * 7 = 2520
    # 1/5 = 504/2520
    # 1/6 = 420/2520
    # 1/7 = 360/2520
    # 1/8 = 315/2520
    # 1/9 = 280/2520
    # 1/10 = 252/2520
    # 1/12 = 210/2520
    # Sum = (504+420+360+315+280+252+210)/2520 = 2341/2520
    # 1 - 2341/2520 = 179/2520
    # x = 2520/179 ≈ 14.08, not integer.
    
    # Let me just run a broader DFS
    for n_terms in range(8, 12):
        result = dfs(MIN_D, Fraction(1, 1), 0, [], n_terms)
        if result is not None:
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(set(result)) == n_terms and all(MIN_D <= d <= MAX_D for d in result):
                return result
    
    # Last resort: return a valid solution found by a different method
    # Use the identity: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - adjustment
    # This is getting complicated. Let's just raise an error if we can't find one.
    
    raise RuntimeError("no representation found")
