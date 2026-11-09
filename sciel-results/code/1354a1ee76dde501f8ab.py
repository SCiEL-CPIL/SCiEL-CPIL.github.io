"""Program to find few-term Egyptian fractions with denominators > 4."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 55  # Safety margin before 90s limit
    
    # Known solution: 1 = 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30
    # Verify: 1/5 + 1/6 = 11/30; 1/8 + 1/12 = 5/24; 1/15 + 1/20 + 1/30 = 1/3
    # 11/30 + 5/24 + 1/3 = 44/120 + 25/120 + 40/120 = 109/120 ... not quite
    # Let me compute properly:
    # 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30
    # LCM of 5,6,8,12,15,20,30 = 120
    # 24 + 20 + 15 + 10 + 8 + 6 + 4 = 87/120 ... not 1
    
    # Try: 1/5 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12
    # LCM = 27720
    # 5544 + 3960 + 3465 + 3080 + 2772 + 2520 + 2310 = 23651/27720 ... not 1
    
    # Let me use a systematic approach with the split-one identity
    # Start with 1/5, then expand to get sum = 1
    
    # Actually, let me just do a targeted search for k=7
    # We need 7 distinct denominators d1 < d2 < ... < d7 in [5, 10^7]
    # with 1/d1 + ... + 1/d7 = 1
    
    # Strategy: fix d1=5, then need 1/d2 + ... + 1/d7 = 4/5
    # Fix d2=6, then need 1/d3 + ... + 1/d7 = 4/5 - 1/6 = 19/30
    # Fix d3=8, then need 1/d4 + ... + 1/d7 = 19/30 - 1/8 = 67/120
    # Fix d4=12, then need 1/d5 + 1/d6 + 1/d7 = 67/120 - 1/12 = 55/120 = 11/24
    # Now need 3 distinct denominators > 12 summing to 11/24
    
    # 1/d5 + 1/d6 + 1/d7 = 11/24
    # d5 >= 13. Try d5=13: 1/d6 + 1/d7 = 11/24 - 1/13 = (143-24)/312 = 119/312
    # d6 >= 14. 1/d6 + 1/d7 = 119/312
    # d6(d7) = 312/119 * (d6+d7)... this is getting complex
    
    # Let me use a cleaner approach: precompute known small solutions
    # and verify them
    
    # Known: 1 = 1/2 + 1/3 + 1/6, but 2,3 < 5
    # 1 = 1/3 + 1/4 + 1/5 + 1/6 + 1/20
    # Check: LCM(3,4,5,6,20) = 60
    # 20 + 15 + 12 + 10 + 3 = 60/60 = 1 ✓
    # But 3 < 5, so invalid
    
    # 1 = 1/4 + 1/5 + 1/6 + 1/10 + 1/12
    # LCM = 60: 15 + 12 + 10 + 6 + 5 = 48/60 = 4/5 ✗
    
    # Let me search computationally with a bounded DFS
    
    def search(k, min_d, remaining, used):
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            if remaining > 0:
                num, den = remaining.numerator, remaining.denominator
                if num == 1:
                    d = den
                    if d >= min_d and d >= 5 and d not in used and d <= 10000000:
                        return [d]
            return None
        
        # Pruning: max possible sum
        max_possible = sum(Fraction(1, min_d + i) for i in range(k))
        if max_possible < remaining:
            return None
        
        # Lower bound for d
        lower_bound = max(min_d, remaining.denominator // remaining.numerator + 1)
        
        # Upper bound: 1/d must be < remaining, and remaining - 1/d must be
        # achievable by k-1 terms > d
        # Also 1/d > remaining - sum(1/(d+1+i) for i in range(k-1))
        # Simpler: just iterate and break when max_rest < new_remaining
        
        # Limit search range for efficiency
        upper_bound = min(10000000, int(remaining.denominator / remaining.numerator) * 2 + 100)
        if upper_bound < lower_bound:
            upper_bound = lower_bound + 1000
        
        for d in range(lower_bound, upper_bound + 1):
            if time.time() > deadline:
                return None
            
            if d in used:
                continue
            
            frac = Fraction(1, d)
            if frac >= remaining:
                continue
            
            new_remaining = remaining - frac
            
            # Check max rest
            max_rest = sum(Fraction(1, d + 1 + i) for i in range(k - 1))
            if max_rest < new_remaining:
                break
            
            result = search(k - 1, d + 1, new_remaining, used | {d})
            if result is not None:
                return [d] + result
        
        return None
    
    # Try k from 5 upward (theoretical minimum is 5 since 1/5+1/6+1/7+1/8+1/9 < 1)
    # Actually 1/5+1/6+1/7+1/8+1/9+1/10 = let's check
    # LCM(5,6,7,8,9,10) = 2520
    # 504+420+360+315+280+252 = 2131/2520 < 1
    # So k >= 7 likely
    
    for k in range(5, 10):
        if time.time() > deadline:
            break
        result = search(k, 5, Fraction(1, 1), set())
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == k and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: return a known valid solution
    # 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30 + 1/40 + 1/60
    # Let me verify a known one:
    # 1 = 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30 + 1/40 + 1/60
    # LCM = 120: 24+20+15+10+8+6+4+3+2 = 92/120 ✗
    
    # Use: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20
    # This will exceed 1. Let me just return a verified solution.
    
    # Verified: 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30 = ?
    # LCM=120: 24+20+15+10+8+6+4 = 87/120 = 29/40 ✗
    
    # 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30 + 1/40 + 1/60
    # LCM=120: 24+20+15+10+8+6+4+3+2 = 92/120 ✗
    
    # Let me use a different known decomposition:
    # 1 = 1/2 + 1/3 + 1/7 + 1/42
    # But 2,3 < 5
    
    # 1 = 1/3 + 1/4 + 1/5 + 1/6 + 1/20
    # 3 < 5
    
    # 1 = 1/4 + 1/5 + 1/6 + 1/10 + 1/12 + 1/15
    # LCM=60: 15+12+10+6+5+4 = 52/60 ✗
    
    # Just return the search result or a hardcoded verified one
    # 1/5 + 1/6 + 1/8 + 1/12 + 1/15 + 1/20 + 1/30 + 1/40 + 1/60 + 1/120
    # LCM=120: 24+20+15+10+8+6+4+3+2+1 = 93/120 ✗
    
    # I'll return a large valid set as fallback
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # Sum > 1, so I need to be careful
    
    # Fallback: use the greedy algorithm to build a valid set
    remaining = Fraction(1, 1)
    result = []
    d = 5
    while remaining > 0 and len(result) < 256:
        # Find smallest d >= current_min such that 1/d < remaining
        # and remaining - 1/d can be completed
        for dd in range(d, 10000001):
            if Fraction(1, dd) >= remaining:
                continue
            new_rem = remaining - Fraction(1, dd)
            if new_rem == 0:
                result.append(dd)
                remaining = 0
                break
            # Check if new_rem is positive and can be formed
            if new_rem > 0:
                result.append(dd)
                remaining = new_rem
                d = dd + 1
                break
        else:
            break
    
    if remaining == 0 and len(result) >= 1 and len(result) <= 256:
        s = sum(Fraction(1, d) for d in result)
        if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
            return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
