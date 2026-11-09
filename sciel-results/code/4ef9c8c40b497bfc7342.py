import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    all_elements = set(range(1, n + 1))
    
    # Helper: Check if adding x to current set S creates a violation
    # A violation exists if there are a, b in S such that a+b, a+c, b+c are in S ∪ {x}
    # Actually, the constraint is: No three distinct a,b,c in S such that a+b, a+c, b+c are in S.
    # If we add x, we need to check if x participates in any such triple.
    # x can be one of a,b,c OR one of the sums.
    
    def is_valid_addition(S, x):
        # S is a set of integers. x is not in S.
        # We need to ensure that for any a, b in S, and c in S ∪ {x} (distinct),
        # the condition "a+b in S∪{x} AND a+c in S∪{x} AND b+c in S∪{x}" does not hold
        # for any distinct a,b,c in S∪{x}.
        
        # This check is O(|S|^2) or O(|S|^3) if done naively.
        # Optimization: 
        # If x is added, the only new triples involving x are those where x is a member or a sum.
        # Case 1: x is one of {a,b,c}.
        #   Then we need to check if there exist distinct a,b in S such that:
        #   (a+b in S∪{x}) AND (a+x in S∪{x}) AND (b+x in S∪{x})
        #   Since x is new, a+x and b+x must be in S (unless they equal x, which implies a=0 or b=0, impossible since elements >=1).
        #   So: Check if there exist distinct a,b in S such that (a+b in S OR a+b==x) AND (a+x in S) AND (b+x in S).
        #   
        # Case 2: x is one of the sums {a+b, a+c, b+c}.
        #   Subcase 2a: x = a+b. Then we need a+c in S and b+c in S, with a,b,c distinct in S.
        #   Subcase 2b: x = a+c. Then we need a+b in S and b+c in S, with a,b,c distinct in S.
        #   Subcase 2c: x = b+c. Then we need a+b in S and a+c in S, with a,b,c distinct in S.
        
        # Let's implement a direct check for efficiency.
        # Precompute pairs in S?
        
        # Check Case 1: x is a member (a, b, or c)
        # Assume x = c. Then a,b in S.
        # Conditions: 
        # 1. a+b in S or a+b == x
        # 2. a+x in S
        # 3. b+x in S
        # 4. a != b, a != x, b != x (x not in S, so a!=x, b!=x is auto if a,b in S)
        
        candidates_for_case1 = []
        for a in S:
            if a + x in S:
                candidates_for_case1.append(a)
        
        if len(candidates_for_case1) > 1:
            for i in range(len(candidates_for_case1)):
                a = candidates_for_case1[i]
                for j in range(i+1, len(candidates_for_case1)):
                    b = candidates_for_case1[j]
                    if a == b: continue
                    # Check a+b
                    if a + b == x or a + b in S:
                        return False
        
        # Check Case 2: x is a sum
        # Subcase 2a: x = a+b. Need a+c in S, b+c in S, c in S, a!=b, a!=c, b!=c.
        # This means for any a,b in S with a+b==x, and any c in S distinct from a,b,
        # if a+c in S and b+c in S, then invalid.
        
        # To speed up, we can iterate c in S.
        # For a fixed c, we need a,b in S\{c} such that a+b==x, a+c in S, b+c in S.
        # Let A_c = {a in S\{c} | a+c in S}
        # We need a,b in A_c such that a+b==x and a!=b.
        
        for c in S:
            # Find a in S\{c} such that a+c in S
            # And b in S\{c} such that b+c in S
            # And a+b == x
            # And a != b
            # Note: a and b must be distinct from c.
            
            # Iterate a in S
            # If a == c, skip
            # If a+c not in S, skip
            # Let b = x - a
            # Check if b in S, b != c, b != a, b+c in S
            
            for a in S:
                if a == c: continue
                if a + x > n: continue # Optional optimization? a+c must be in S, so a+c <= n.
                # Actually a+c in S implies a+c <= n.
                if a + c > n: continue
                if (a + c) not in S: continue
                
                b = x - a
                if b == a: continue
                if b == c: continue
                if b < 1 or b > n: continue
                if b not in S: continue
                if b + c > n: continue
                if (b + c) not in S: continue
                
                return False
        
        return True

    # Initialize with a known valid set: Upper half
    current_set = set(range(n//2 + 1, n + 1))
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local Search
    # Strategy: 
    # 1. Try to add random elements.
    # 2. If stuck, remove random elements to allow new additions.
    
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # 1. Try to add a random missing element
        missing = list(all_elements - current_set)
        if missing:
            random.shuffle(missing)
            added = False
            for x in missing:
                if is_valid_addition(current_set, x):
                    current_set.add(x)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    added = True
                    break
            
            if added:
                continue
        
        # 2. If no element could be added, remove a random element to escape
        # To avoid removing too many, we can limit removals or just do one.
        if current_set:
            # Remove a random element
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            
            # After removal, we might be able to add something else.
            # The loop will naturally try to add again.
            
        # Optional: Periodic restart or larger perturbation if stuck for long
        # For now, simple removal should help explore.

    # Ensure best_set is returned as a list of ints
    result = list(best_set)
    return result
