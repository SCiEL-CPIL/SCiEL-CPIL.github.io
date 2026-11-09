"""Rewritten program: SAT-based packing of distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # safety margin before 90s kill

    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Initial solution: just the divisors themselves
    best_val = max(divisors) - min(divisors)
    best_assign = divisors[:]

    # We want to minimize span = max(values) - min(values)
    # For a given span W, we need to find a window [L, L+W] containing distinct multiples
    # of each divisor.
    
    # Strategy:
    # 1. For small W (span), the number of candidates per divisor in any window is small.
    # 2. We can iterate over possible L (start of window) and for each L, check if a valid
    #    assignment exists using backtracking or matching.
    # 3. To speed up, we can restrict L to values where at least one divisor has a multiple
    #    at L or near L, or use random sampling.
    
    # A key insight: if a valid assignment exists for span W, it exists for span W+1.
    # So we can binary search or linear search for the minimum W.
    
    # However, checking existence for a specific W and L is the hard part.
    # Let's define a function has_valid_assignment(L, W):
    #   For each divisor d_i, the candidates are multiples of d_i in [L, L+W].
    #   We need to pick one distinct value per divisor.
    #   This is a bipartite matching problem or can be solved with backtracking.
    
    # To make it faster, we can precompute for each divisor d_i and a given L,
    # the list of candidate multiples in [L, L+W].
    
    # Let's try a randomized search over L for increasing W.
    
    # Start with a reasonable lower bound for W.
    # The minimum span must be at least 9 (10 distinct integers).
    # But due to divisibility constraints, it's likely larger.
    
    # Let's try W from 9 up to some limit.
    # For each W, try random L values.
    
    # To improve efficiency, we can use the fact that if L is a multiple of some divisor,
    # it might be a good start. Or we can just sample L uniformly.
    
    # Let's also try a local search approach: start from a known solution and try to
    # reduce the span by moving individual values.
    
    # First, let's do a systematic search for small W.
    
    # Precompute for each divisor, the period is the divisor itself.
    
    # For a fixed W, the condition for a divisor d to have at least one multiple in [L, L+W]
    # is that there exists k such that L <= k*d <= L+W.
    # This is equivalent to: floor((L+W)/d) - ceil(L/d) + 1 >= 1
    # i.e., floor((L+W)/d) >= ceil(L/d)
    
    # Let's define a helper to check if a window [L, L+W] is "feasible" (each divisor has >=1 candidate)
    def is_feasible_window(L, W):
        for d in divisors:
            first_k = (L + d - 1) // d  # ceil(L/d)
            last_k = (L + W) // d        # floor((L+W)/d)
            if first_k > last_k:
                return False
        return True
    
    # Now, for a feasible window, we need to find a distinct assignment.
    # We'll use backtracking with ordering by number of candidates (most constrained first).
    
    def find_assignment(L, W):
        # Build candidate lists
        candidates = []
        for d in divisors:
            first_k = (L + d - 1) // d
            last_k = (L + W) // d
            if first_k > last_k:
                return None
            # List of values: [first_k*d, (first_k+1)*d, ..., last_k*d]
            # We don't want to store huge lists, but W is small in our search.
            # Actually, W is the span, so the number of candidates is at most W/d + 1.
            # For small W, this is manageable.
            cand_list = [k * d for k in range(first_k, last_k + 1)]
            candidates.append(cand_list)
        
        # Order divisors by number of candidates (ascending)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return assignment[:]
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    used.remove(val)
            return None
        
        return backtrack(0)
    
    # Search for the minimum W
    # We'll try W from 9 up to, say, 1000.
    # For each W, we try random L values.
    
    # To make it more efficient, we can restrict L to a range.
    # Since the values must be <= MAX_V, L must be <= MAX_V - W.
    # Also, L must be >= 1.
    
    # Let's try a combination of random sampling and systematic sampling.
    
    found_better = False
    max_W_to_try = 500  # We'll go up to span 500
    
    for W in range(9, max_W_to_try + 1):
        if time.time() > deadline:
            break
        
        # Try random L values
        num_attempts = 20
        for _ in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Sample L from [1, MAX_V - W]
            # But to make it more likely to find a solution, we can bias towards
            # values where L is close to a multiple of some divisor.
            # For now, just uniform random.
            L = random.randint(1, MAX_V - W)
            
            if not is_feasible_window(L, W):
                continue
            
            result = find_assignment(L, W)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found_better = True
                    # If we found a solution with span == W, that's the best possible for this W
                    # But we might find a smaller W later, so continue searching for smaller W?
                    # No, we're iterating W from small to large. So if we find a solution for W,
                    # we can break out of the W loop if we're looking for the minimum.
                    # But wait, we want the minimum span. So if we find a solution for W,
                    # we should continue to check if a smaller W is possible? No, we're going
                    # from small W to large W. So the first W for which we find a solution
                    # is the minimum W.
                    # However, it's possible that for a smaller W, we just didn't find a solution
                    # because we didn't try the right L. So we should be careful.
                    # For now, let's just record the best and continue.
        
        # If we found a solution with span == W, we can stop searching for larger W
        # because we're looking for the minimum.
        # But actually, we want the minimum span, so if we found a solution for W,
        # we should continue to check if a smaller W is possible. But we're iterating
        # from small to large, so if we find a solution for W, it means no solution
        # was found for W-1, W-2, etc. (assuming our search was exhaustive, which it's not).
        # So we can't be sure that W is the minimum.
        # Let's just continue searching for larger W to find a better (smaller) span?
        # No, larger W means larger span. We want smaller span.
        # So we should iterate W from small to large, and for each W, try to find a solution.
        # The first W for which we find a solution is our best candidate for minimum span.
        # But since our search is randomized, we might miss a solution for a smaller W.
        # So we should do multiple passes or increase the number of attempts for smaller W.
        
        # For now, let's just continue and keep the best found.
    
    # Additional local search: try to improve the current best solution
    if best_assign is not None:
        for _ in range(500):
            if time.time() > deadline:
                break
            candidate = best_assign[:]
            # Pick a random index
            idx = random.randint(0, n - 1)
            d = divisors[idx]
            current = candidate[idx]
            # Try to move to a nearby multiple
            # We want to reduce the span, so we should try to move values towards the center.
            # Let's try a few nearby multiples.
            for delta_k in [-3, -2, -1, 1, 2, 3]:
                new_val = current + delta_k * d
                if new_val < 1 or new_val > MAX_V:
                    continue
                if new_val in candidate:
                    continue
                candidate[idx] = new_val
                # Check distinctness
                if len(set(candidate)) == n:
                    span = max(candidate) - min(candidate)
                    if span < best_val:
                        best_val = span
                        best_assign = candidate[:]
                        break
                candidate[idx] = current
    
    # Final validation
    if best_assign is None:
        best_assign = divisors[:]
    
    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)
    
    # Verify
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int), f"Element {i} is not int"
        assert not isinstance(result[i], bool), f"Element {i} is bool"
        assert 1 <= result[i] <= MAX_V, f"Element {i} out of range: {result[i]}"
        assert result[i] % d == 0, f"Element {i} ({result[i]}) not divisible by {d}"
    assert len(set(result)) == n, "Elements are not distinct"
    
    return result

# EVOLVE-BLOCK-END
