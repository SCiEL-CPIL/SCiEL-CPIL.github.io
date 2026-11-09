"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    """Find minimum number of distinct unit fractions (denominators >= 5) summing to 1."""
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    # We know that 3 terms is impossible (minimum sum with 3 distinct denoms >= 5 is 1/5+1/6+1/7 < 1)
    # Actually 1/5 + 1/6 + 1/7 = 41/210 ≈ 0.195, way less than 1.
    # Wait, we need sum = 1. With 3 terms, max is 1/5 + 1/6 + 1/7 which is < 1.
    # So we need at least 4 terms? Let's check: 1/5 + 1/6 + 1/7 + 1/8 = 1.0?
    # 1/5=0.2, 1/6≈0.1667, 1/7≈0.1429, 1/8=0.125. Sum ≈ 0.6345. Still < 1.
    # We need many more terms. Let's think about the minimum number.
    # The greedy algorithm for Egyptian fractions gives a solution, but not necessarily minimal terms.
    # Actually, we want to MINIMIZE the number of terms. So we want the LARGEST possible unit fractions.
    # To minimize terms, we should pick the smallest possible denominators (largest reciprocals).
    # But they must be distinct and >= 5.
    
    # Let's use a BFS/DFS approach trying to find a solution with k terms, starting from small k.
    # But what is the minimum k?
    # If we take the largest possible unit fractions: 1/5, 1/6, 1/7, ...
    # We need sum = 1.
    # Let's estimate: sum of 1/n for n=5 to N.
    # sum_{n=5}^{N} 1/n ≈ ln(N) - ln(4) = ln(N/4)
    # We need ln(N/4) ≈ 1 => N/4 ≈ e => N ≈ 11.
    # So roughly 7 terms (5,6,7,8,9,10,11) might be close.
    # Let's check: 1/5+1/6+1/7+1/8+1/9+1/10+1/11
    # = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 + 0.1 + 0.0909 ≈ 0.9366
    # Still < 1. Need more terms or adjust.
    # Actually, we can choose which denominators to use, not just consecutive.
    # To minimize terms, we want to maximize each reciprocal, i.e., minimize denominators.
    # But we must hit exactly 1.
    
    # Let's try a backtracking search for k terms, starting from k=4, 5, 6, ...
    # But we need to be careful about the search space.
    
    # Alternative: use the known result that the minimum number of distinct unit fractions 
    # with denominators >= 5 summing to 1 is known. Let me think...
    # Actually, let's just do a smart search.
    
    # Key insight: to minimize terms, we should try to use the smallest denominators first.
    # We can do a DFS that tries to fill the sum 1 using distinct denominators >= 5,
    # always trying the smallest available denominator first (greedy choice for minimization).
    
    # But we need exactly sum = 1, and we want to minimize the count.
    # So let's try: can we do it with k terms? Start with small k.
    
    # For a given k, we need to find distinct d1 < d2 < ... < dk, all >= 5,
    # such that sum(1/di) = 1.
    
    # To make this tractable, we can use the fact that d1 must be small.
    # 1/d1 < 1 => d1 > 1, which is always true.
    # But 1/d1 + 1/(d1+1) + ... + 1/(d1+k-1) >= 1 (necessary condition)
    # And 1/d1 + 1/(d1+1) + ... + 1/(d1+k-1) is the max possible sum with k terms starting from d1.
    
    # Actually, for a fixed k, the minimum possible denominator d1 must satisfy:
    # The maximum sum with k terms starting from d1 is sum_{i=0}^{k-1} 1/(d1+i)
    # This must be >= 1.
    # And the minimum sum (with very large denominators) is 0, so no lower bound from that.
    # But we also need the minimum possible sum with k terms starting from d1 to be <= 1,
    # which is always true.
    
    # Let's try k from 4 upward, and for each k, search for a valid set.
    
    def can_complete(remaining, min_denom, k, used):
        """
        Check if we can represent 'remaining' as a sum of k distinct unit fractions
        with denominators >= min_denom, none in 'used'.
        Returns a list of denominators or None.
        """
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            if remaining > 0:
                d = 1 / remaining
                if d >= min_denom and d >= 5 and abs(d - round(d)) < 1e-10:
                    d_int = int(round(d))
                    if d_int not in used and d_int <= 10000000:
                        return [d_int]
            return None
        
        # Pruning: max possible sum with k terms from min_denom
        max_sum = sum(1 / (min_denom + i) for i in range(k))
        if max_sum < remaining - 1e-12:
            return None
        
        # Pruning: min possible sum is 0, so no help there.
        # Lower bound on next denominator: 1/d <= remaining/k => d >= k/remaining
        # But also d >= min_denom
        lb = max(min_denom, int(k / remaining) + 1)
        
        # Upper bound on next denominator: 1/d < remaining (must leave room for rest)
        # Actually, 1/d < remaining is necessary (since other terms are positive)
        # So d > 1/remaining, i.e., d >= floor(1/remaining) + 1
        # But we also need the remaining k-1 terms to be able to sum to remaining - 1/d.
        # The max sum of k-1 terms starting from d+1 is sum_{i=1}^{k-1} 1/(d+i)
        # We need remaining - 1/d <= sum_{i=1}^{k-1} 1/(d+i)
        # This gives an upper bound on d.
        
        # For efficiency, let's just iterate d from lb upward, with a reasonable upper bound.
        # The upper bound: 1/d must be < remaining, so d > 1/remaining.
        # Also, we need remaining - 1/d <= (k-1)/(d+1) (rough upper bound on remaining sum)
        # => remaining - 1/d <= (k-1)/(d+1)
        # => remaining*(d+1) - (d+1)/d <= k-1
        # => remaining*d + remaining - 1 - 1/d <= k-1
        # => remaining*d <= k-1 - remaining + 1 + 1/d
        # => d <= (k - remaining) / remaining + 1/(d*remaining)
        # This is getting complicated. Let's just use a practical upper bound.
        
        ub = int(1 / remaining) + 1  # d must be > 1/remaining
        # But this could be large. Let's also cap it.
        # Actually, if remaining is small, 1/remaining is large, but we need d to be not too large
        # because the remaining terms must sum to a significant amount.
        
        # Let's try a different approach: iterate d from lb, and break when
        # even the max sum of remaining k-1 terms from d+1 can't reach remaining - 1/d.
        
        for d in range(lb, ub + 1):
            if time.time() > deadline:
                return None
            
            if d in used:
                continue
            
            frac = 1.0 / d
            if frac >= remaining:
                break  # d is too small, 1/d >= remaining, can't use it (need strictly less)
            
            new_remaining = remaining - frac
            
            # Check if new_remaining can be achieved with k-1 terms from d+1
            # Max sum of k-1 terms from d+1:
            max_rest = sum(1.0 / (d + 1 + i) for i in range(k - 1))
            if max_rest < new_remaining - 1e-12:
                # d is too large, can't fill the remaining sum
                # But wait, as d increases, 1/d decreases, so new_remaining increases.
                # And max_rest decreases. So if max_rest < new_remaining, 
                # for larger d, new_remaining will be even larger and max_rest even smaller.
                # So we can break.
                break
            
            result = can_complete(new_remaining, d + 1, k - 1, used | {d})
            if result is not None:
                return [d] + result
        
        return None
    
    # Try increasing k
    for k in range(4, 20):
        if time.time() > deadline:
            break
        
        result = can_complete(1.0, 5, k, set())
        if result is not None:
            # Verify with exact fractions
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(result) == k and all(5 <= d <= 10000000 for d in result) and len(set(result)) == k:
                return result
    
    # Fallback: use a known decomposition
    # Let's construct one using the identity 1/n = 1/(n+1) + 1/(n(n+1))
    # Start with 1/5, and keep splitting.
    # Actually, let's just use a greedy approach to get a valid solution.
    
    def greedy_egyptian(target, min_denom, max_terms=256):
        """Greedy Egyptian fraction decomposition."""
        result = []
        remaining = Fraction(target)
        current_min = min_denom
        
        for _ in range(max_terms):
            if remaining == 0:
                break
            if remaining < 0:
                return None
            
            # Find smallest d >= current_min such that 1/d <= remaining
            d = max(current_min, (1 / remaining).numerator // (1 / remaining).denominator + 1)
            # Actually, d = ceil(1/remaining)
            if remaining >= 1:
                d = current_min
            else:
                d = max(current_min, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            
            if d < current_min:
                d = current_min
            
            if 1 / d > remaining:
                return None
            
            result.append(d)
            remaining -= Fraction(1, d)
            current_min = d + 1
        
        if remaining == 0:
            return result
        return None
    
    result = greedy_egyptian(1, 5)
    if result is not None and len(result) <= 256:
        s = sum(Fraction(1, d) for d in result)
        if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
            return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END

