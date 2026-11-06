"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                if math.lcm(S_list[i], S_list[j]) <= 256:
                    return False
        return True

    def score(S):
        return sum(1.0 / x for x in S)

    # 1. Greedy Initialization
    # Sort candidates by reciprocal (i.e., ascending value) to prioritize high weights
    candidates = list(range(2, 257))
    S = set()
    for x in candidates:
        if is_valid_set(S | {x}):
            S.add(x)
    
    current_score = score(S)
    
    # 2. Local Search
    # We will try to improve the set by adding, swapping 1-for-1, or 1-for-2.
    # Since we want to maximize sum of reciprocals, any valid addition is good.
    # Swaps are good if the new reciprocal sum is higher.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any single element
        for x in range(2, 257):
            if x not in S:
                new_S = S | {x}
                if is_valid_set(new_S):
                    S = new_S
                    current_score = score(S)
                    improved = True
                    # Don't break immediately, see if we can add more, 
                    # but re-evaluating all x is expensive. 
                    # Actually, if we add x, it might invalidate future adds, 
                    # but since we check validity, it's safe.
                    # However, adding one might prevent adding another. 
                    # Greedy addition is local.
        
        # Try 1-for-1 swaps
        # For each a in S, for each b not in S, if swap improves score and is valid
        # Optimization: Only consider b where 1/b > 1/a (i.e., b < a) is not strictly true for validity,
        # but for score improvement we need 1/b > 1/a.
        # Wait, if b < a, 1/b > 1/a. So we should look for smaller b.
        # But b must be compatible with S \ {a}.
        
        S_list = list(S)
        for a in S_list:
            # We only want to swap if we can find a better b
            # b must be < a to improve score? No, 1/b > 1/a implies b < a.
            # So we look for b in [2, a-1] not in S.
            # But wait, what if b > a but 1/b > 1/a? Impossible.
            # So we only look for b < a.
            # However, it's possible that a large a is blocking many small b's?
            # No, the constraint is pairwise.
            # Let's just iterate all b not in S.
            for b in range(2, 257):
                if b in S:
                    continue
                if 1.0 / b <= 1.0 / a:
                    continue # No score improvement
                new_S = (S - {a}) | {b}
                if is_valid_set(new_S):
                    S = new_S
                    current_score = score(S)
                    improved = True
                    # Restart inner loop? Or just continue.
                    # Changing S invalidates the S_list iteration potentially.
                    # Let's break and restart the while loop for safety.
                    break
            if improved:
                break
        if improved:
            continue

        # Try 1-for-2 swaps (remove a, add b and c)
        # Condition: 1/b + 1/c > 1/a
        S_list = list(S)
        for a in S_list:
            found = False
            # To make this faster, we can pre-filter b, c
            # b, c must be compatible with S \ {a}
            # And b, c compatible with each other
            # And 1/b + 1/c > 1/a
            
            # Heuristic: b and c should be small.
            # Max possible sum of two reciprocals is 1/2 + 1/3 = 0.83.
            # If 1/a > 0.83 (a < 1.2), impossible. a>=2, 1/2=0.5.
            # So we can always potentially improve if we find two small numbers.
            
            base_S = S - {a}
            
            # Iterate over pairs b, c
            # To avoid duplicates, b < c
            for b in range(2, 257):
                if b in S:
                    continue
                # Early check: even if c is the smallest possible (2, if b!=2),
                # does 1/b + 1/c_max > 1/a?
                # Max 1/c is 1/2 (if c=2 and b!=2) or 1/3 (if b=2).
                # Let's just check validity first, it's the bottleneck.
                
                # Check if b is compatible with base_S
                if not all(math.lcm(b, y) > 256 for y in base_S):
                    continue
                
                for c in range(b + 1, 257):
                    if c in S:
                        continue
                    # Score check
                    if 1.0 / b + 1.0 / c <= 1.0 / a:
                        continue
                    
                    # Check if c is compatible with base_S
                    if not all(math.lcm(c, y) > 256 for y in base_S):
                        continue
                    
                    # Check if b and c are compatible
                    if math.lcm(b, c) <= 256:
                        continue
                    
                    # Valid swap
                    new_S = base_S | {b, c}
                    S = new_S
                    current_score = score(S)
                    improved = True
                    found = True
                    break
                if found:
                    break
            if found:
                break
        
    return list(S)
# EVOLVE-BLOCK-END
