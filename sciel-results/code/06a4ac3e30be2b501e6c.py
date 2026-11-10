import time
import sys
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors to determine the period of the pattern
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
        
    # We need to search L in [1, period] because the pattern of multiples repeats every 'period'
    # However, the window [L, L+W] might extend beyond period.
    # We can just search L in [1, period] and consider values modulo period?
    # No, values are distinct integers.
    # But the *relative* positions of multiples repeat.
    # If a solution exists for L, it exists for L + k*period.
    # So we only need to check L in [1, period].
    # The values in the solution will be in [L, L+W].
    # We must ensure these values are <= 1000000.
    # Since W is likely small and L <= period (30030), L+W is small.
    
    # Precompute masks for v in 1..period+max(divisors)+W_max
    # Let's assume W_max is 100.
    # Max value we need mask for is period + 100.
    max_search_val = period + 200
    masks = [0] * (max_search_val + 1)
    for v in range(1, max_search_val + 1):
        m = 0
        for i, d in enumerate(divisors):
            if v % d == 0:
                m |= (1 << i)
        masks[v] = m
        
    def find_assignment(L, W, divisors, masks):
        n = len(divisors)
        # Candidates for each divisor are values in [L, L+W] that are multiples
        candidates = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            start_k = (L + d - 1) // d
            end_k = (L + W) // d
            for k in range(start_k, end_k + 1):
                v = k * d
                if L <= v <= L + W:
                    candidates[i].append(v)
                    
        # If any divisor has no candidates, return None
        for i in range(n):
            if not candidates[i]:
                return None
                
        # Backtracking
        assignment = [None] * n
        used = set()
        
        def bt(idx):
            if idx == n:
                return True
            # Optimization: Sort candidates? Or just iterate.
            # To improve pruning, try candidates with fewer options first?
            # Here we fix the divisor order.
            for v in candidates[idx]:
                if v not in used:
                    used.add(v)
                    assignment[idx] = v
                    if bt(idx + 1):
                        return True
                    used.remove(v)
                    assignment[idx] = None
            return False
            
        if bt(0):
            return assignment
        return None

    best_solution = [int(d) for d in divisors]
    best_range = max(best_solution) - min(best_solution)
    
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Search for minimal W
    # The range is likely small. Let's try W from 1 to 100.
    # If not found, we might need to go higher, but 33 is the current best.
    # So we only care about W < 33.
    # Actually, we can just search W from 1 to best_range - 1.
    
    for W in range(1, best_range):
        if time.time() > deadline:
            break
            
        # Slide window L from 1 to period
        # We maintain bit counts for the window [L, L+W]
        bit_counts = [0] * n
        
        # Initialize window for L=1: [1, 1+W]
        for v in range(1, W + 2):
            m = masks[v]
            # Update bit counts
            # Instead of iterating all n bits, iterate set bits
            # Or just loop i in range(n)
            for i in range(n):
                if m & (1 << i):
                    bit_counts[i] += 1
        
        found_for_w = False
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
                
            # Check if all divisors are covered
            # If any bit_count is 0, no solution
            if all(c > 0 for c in bit_counts):
                # Necessary condition met. Check matching.
                sol = find_assignment(L, W, divisors, masks)
                if sol is not None:
                    # Validate
                    if len(set(sol)) == n and all(1 <= x <= max_val and x % divisors[i] == 0 for i, x in enumerate(sol)):
                        return [int(x) for x in sol]
            
            # Slide window: remove L, add L+W+1
            # Remove L
            m_remove = masks[L]
            for i in range(n):
                if m_remove & (1 << i):
                    bit_counts[i] -= 1
            
            # Add L+W+1
            new_v = L + W + 1
            if new_v <= max_search_val:
                m_add = masks[new_v]
                for i in range(n):
                    if m_add & (1 << i):
                        bit_counts[i] += 1
            else:
                # If we go out of precomputed range, we might need to compute mask on fly
                # But period is 30030, W is small. L+W+1 <= 30030 + 33 + 1 < 30100.
                # max_search_val is 30230. So we are safe.
                pass
                
    return best_solution
