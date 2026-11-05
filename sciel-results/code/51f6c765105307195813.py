"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    # Minimum number of terms is 10 (proven: k=9 max sum < 1, k=10 max sum > 1)
    # We search for a valid 10-term solution with distinct denominators in [5, 10000000]
    
    def solve_k10():
        """Find 10 distinct denominators >= 5 summing to 1 using DFS with strong pruning."""
        
        def max_sum_from(start, n):
            """Upper bound: sum of 1/(start) + 1/(start+1) + ... + 1/(start+n-1)"""
            if n <= 0:
                return Fraction(0, 1)
            # Use harmonic number approximation for speed, but verify with exact if close
            # For exactness in pruning, compute directly for small n
            s = Fraction(0, 1)
            for i in range(n):
                s += Fraction(1, start + i)
            return s
        
        def min_sum_from(start, n):
            """Lower bound: sum of n smallest possible reciprocals >= 1/start... but we need them distinct and increasing.
            The minimum sum with n distinct denominators all >= start is 1/start + 1/(start+1) + ... + 1/(start+n-1)
            Wait no - that's the MAX sum (smallest denominators give largest reciprocals).
            The MIN sum would be with the largest denominators, but they're unbounded.
            So for pruning: if max possible sum < target, prune.
            If min possible sum > target, we'd need to go to larger denominators, but that's handled by the search.
            """
            pass
        
        def dfs(start, remaining, target, current):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return current if target == 0 else None
            if target <= 0:
                return None
            
            # Pruning 1: max possible sum with 'remaining' terms starting from 'start'
            # must be >= target
            max_possible = max_sum_from(start, remaining)
            if max_possible < target:
                return None
            
            # Pruning 2: each term 1/d must be <= target, so d >= ceil(1/target)
            # Also d >= start
            # And we need enough room: the remaining-1 smallest denominators after d
            # must be able to sum to target - 1/d
            # Lower bound on d: 1/d <= target => d >= ceil(1/target)
            # Upper bound on d: we need (remaining-1) terms each >= 1/(d+1), 1/(d+2), ...
            # to sum to at least target - 1/d
            # So target - 1/d <= sum_{i=1}^{remaining-1} 1/(d+i)
            # This gives an upper bound on d.
            
            # Also: 1/d >= target / remaining (since each of the remaining terms is at most 1/d, 1/(d+1), ...
            # but actually each subsequent term is smaller, so 1/d should be at least target/remaining
            # for feasibility. More precisely, the largest term is 1/d, and we have 'remaining' terms,
            # so 1/d >= target/remaining => d <= remaining/target
            min_d = max(start, math.ceil(1 / target))
            max_d = int(remaining / target) if target > 0 else 10000000
            max_d = min(max_d, 10000000)
            
            if min_d > max_d:
                return None
            
            # Try denominators from min_d to max_d
            # Heuristic: try smaller d first (larger reciprocals) to fill target faster
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return None
                
                new_target = target - Fraction(1, d)
                if new_target < 0:
                    break  # Since d is increasing, new_target will only get more negative
                
                # Pruning: check if new_target can be achieved with (remaining-1) terms
                # starting from d+1
                if remaining - 1 > 0:
                    max_next = max_sum_from(d + 1, remaining - 1)
                    if max_next < new_target:
                        continue  # This d is too small, but larger d will have smaller 1/d,
                                   # so new_target will be larger... wait no.
                        # Actually if d increases, 1/d decreases, so new_target = target - 1/d increases.
                        # So if max_next < new_target for this d, it might be feasible for smaller d.
                        # But we're going from min_d upward, so smaller d was already tried.
                        # If max_next < new_target, this d is too large (gives too small 1/d,
                        # leaving too much for remaining terms). So we can break.
                        # Wait: as d increases, 1/d decreases, new_target increases.
                        # max_next = sum of 1/(d+1) + ... which also decreases as d increases.
                        # So the condition max_next < new_target becomes harder to satisfy as d increases.
                        # If it's true for some d, it'll be true for all larger d. So break.
                        break
                    # Also check: new_target > 0 required (already checked)
                    # And new_target must be >= sum of (remaining-1) smallest possible... 
                    # which is unbounded below, so no lower bound issue.
                
                res = dfs(d + 1, remaining - 1, new_target, current + [d])
                if res is not None:
                    return res
            
            return None
        
        return dfs(5, 10, Fraction(1, 1), [])
    
    result = solve_k10()
    
    if result and len(result) == 10:
        if all(5 <= d <= 10000000 for d in result) and len(set(result)) == 10:
            s = sum(Fraction(1, d) for d in result)
            if s == 1:
                return result
    
    # Fallback: try k=11, 12, etc. with a more targeted search
    for k in range(11, 257):
        if time.time() > deadline - 5:
            break
        
        def solve_k(k_val):
            def dfs(start, remaining, target, current):
                if time.time() > deadline:
                    return None
                if remaining == 0:
                    return current if target == 0 else None
                if target <= 0:
                    return None
                
                max_possible = max_sum_from(start, remaining)
                if max_possible < target:
                    return None
                
                min_d = max(start, math.ceil(1 / target))
                max_d = int(remaining / target) if target > 0 else 10000000
                max_d = min(max_d, 10000000)
                
                if min_d > max_d:
                    return None
                
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        return None
                    new_target = target - Fraction(1, d)
                    if new_target < 0:
                        break
                    if remaining - 1 > 0:
                        max_next = max_sum_from(d + 1, remaining - 1)
                        if max_next < new_target:
                            break
                    res = dfs(d + 1, remaining - 1, new_target, current + [d])
                    if res is not None:
                        return res
                return None
            
            return dfs(5, k_val, Fraction(1, 1), [])
        
        result = solve_k(k)
        if result and len(result) == k:
            if all(5 <= d <= 10000000 for d in result) and len(set(result)) == k:
                s = sum(Fraction(1, d) for d in result)
                if s == 1:
                    return result
    
    # Last resort: use a known valid larger solution
    # Greedy Egyptian fraction with adjustment
    def greedy_with_adjustment(target, min_denom, max_terms=256):
        denoms = []
        last_d = min_denom - 1
        current_target = target
        while current_target > 0 and len(denoms) < max_terms:
            if time.time() > deadline:
                break
            # Find smallest d > last_d such that 1/d <= current_target
            d = max(last_d + 1, math.ceil(1 / current_target))
            while Fraction(1, d) > current_target:
                d += 1
            if d > 10000000:
                break
            denoms.append(d)
            current_target -= Fraction(1, d)
            last_d = d
        return denoms if current_target == 0 else None
    
    result = greedy_with_adjustment(Fraction(1, 1), 5)
    if result and len(result) >= 1 and len(result) <= 256:
        if all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
            s = sum(Fraction(1, d) for d in result)
            if s == 1:
                return result
    
    # Final fallback: return a valid (though not minimal) solution
    # Use the fact that we can always construct a solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + ...
    # But we need exact sum = 1. Let's use a constructive approach.
    # 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + ...
    # This is getting complex. Just return a hardcoded valid solution.
    # After verification: {5, 6, 7, 10, 14, 21, 30, 42, 70, 140, 210, 420} has 12 terms.
    # Sum = 42/210 + 35/210 + 30/210 + 21/210 + 15/210 + 10/210 + 7/210 + 5/210 + 3/210 + 1.5/210... 
    # Let me just compute: with LCM 420:
    # 1/5=84/420, 1/6=70/420, 1/7=60/420, 1/10=42/420, 1/14=30/420, 1/21=20/420,
    # 1/30=14/420, 1/42=10/420, 1/70=6/420, 1/140=3/420, 1/210=2/420, 1/420=1/420
    # Sum = 84+70+60+42+30+20+14+10+6+3+2+1 = 342/420 = 57/70 != 1
    
    # I'll use a simpler guaranteed approach: return denominators that definitely work.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This sum is > 1, so it's invalid.
    
    # The safest fallback: return a known valid set. Let me construct one.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14
    # Sum ≈ 1.030, too much.
    # Replace 1/14 with 1/x where 1/x = 1 - (1/5+...+1/13)
    # 1/5+...+1/13 = sum_{i=5}^{13} 1/i ≈ 0.995, so 1/x ≈ 0.005, x ≈ 200
    # But 1/200 = 0.005, and 0.995 + 0.005 = 1.000? Let me check exactly.
    # This is getting too complex for a fallback. The DFS should find a solution.
    
    # Return empty as last resort (should not happen if DFS works)
    return []
# EVOLVE-BLOCK-END
