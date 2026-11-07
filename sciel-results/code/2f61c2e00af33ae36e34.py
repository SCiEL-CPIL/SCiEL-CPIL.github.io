import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    n = 128
    # Base set: A=[16..32], B=[65..128]
    # Size: 17 + 64 = 81
    base_set = set(range(16, 33)) | set(range(65, 129))
    candidates = [x for x in range(1, n + 1) if x not in base_set]
    
    best_set = base_set.copy()
    best_size = len(best_set)
    
    # Helper to check if adding x to S is valid
    def is_valid_addition(S, x):
        # S is a set, x is an int
        # Check all pairs (a, b) in S such that a < b
        # If a+b, a+x, b+x are all in S U {x}, then invalid
        # Optimization: 
        # 1. a+b must be in S or {x}
        # 2. a+x must be in S or {x}
        # 3. b+x must be in S or {x}
        
        S_new = S | {x}
        
        # We only need to check triples involving x?
        # No, adding x might complete a triple that was previously incomplete?
        # No, if a triple was invalid before, it was because one sum was missing.
        # Adding x can only provide the missing sum.
        # So we only need to check triples where x is one of the elements a,b,c
        # OR x is one of the sums.
        
        # Case 1: x is one of a,b,c.
        # Let the triple be {x, a, b} with a,b in S.
        # Sums: x+a, x+b, a+b.
        # All must be in S_new.
        
        # Case 2: x is one of the sums.
        # Let the triple be {a, b, c} with a,b,c in S.
        # Sums: a+b, a+c, b+c.
        # One of these is x.
        # Say a+b = x.
        # Then we need a+c in S_new and b+c in S_new.
        # Since a,b,c in S, and S subset S_new, a+c and b+c must be in S or {x}.
        
        # It is simpler to just check all triples in S_new that include x?
        # No, if x is a sum, the triple {a,b,c} does not include x.
        # So we must check triples in S that have a sum equal to x.
        
        # Efficient check:
        # 1. Check triples {x, a, b} for a,b in S.
        for a in S:
            if a == x: continue
            for b in S:
                if b == x or b <= a: continue
                # Triple x, a, b
                s1 = x + a
                s2 = x + b
                s3 = a + b
                if s1 in S_new and s2 in S_new and s3 in S_new:
                    return False
        
        # 2. Check triples {a, b, c} in S such that one sum is x.
        # This is harder to iterate.
        # Alternative: Just check all triples in S_new? 
        # |S| ~ 80. |S|^3 is ~ 500k. Too slow for 100 candidates?
        # 100 * 500k = 50M. Might be ok in Python if optimized? No, Python is slow.
        
        # Let's optimize Case 2.
        # If a+b = x, then c must be in S.
        # We need a+c in S_new and b+c in S_new.
        # Iterate pairs (a,b) in S such that a+b = x.
        for a in S:
            b = x - a
            if b in S and b > a:
                # Found pair a,b summing to x
                # Now check if there exists c in S such that a+c in S_new and b+c in S_new
                # Note: c != a, c != b
                for c in S:
                    if c == a or c == b: continue
                    s1 = a + c
                    s2 = b + c
                    # s3 = x is in S_new by definition
                    if s1 in S_new and s2 in S_new:
                        return False
        
        return True

    # Greedy search with random restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        current_set = base_set.copy()
        remaining = candidates.copy()
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        # If we added many, try to refine?
        # Or just restart with different shuffle.
        
    return list(best_set)

# Run construct to test? No, just return the function.
# The evaluator calls construct().
