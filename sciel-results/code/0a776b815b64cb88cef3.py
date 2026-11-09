"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    """
    Returns a nonempty subset of [1, 22] with distinct reciprocal subset sums.
    Maximizes cardinality.
    """
    # 1. Precompute integer weights
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i
        
    # 2. Helper to check dissociation using integer sums
    def is_dissociated(subset):
        sums = {0}
        for x in subset:
            w = weights[x]
            new_sums = set(sums)
            for s in sums:
                ns = s + w
                if ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = new_sums
        return True

    # 3. Search for maximum K
    # We know K=5 is possible: {1, 2, 4, 8, 16}
    # Let's try to find larger K.
    
    # To improve speed, we use a randomized greedy approach with integer weights.
    # We try K from 10 down to 1.
    
    random.seed(42)
    
    # Pre-defined good set to ensure baseline
    known_good = [1, 2, 4, 8, 16]
    if is_dissociated(known_good):
        max_found = known_good
    else:
        max_found = [1]

    for k in range(10, 1, -1):
        found = False
        # Try multiple random orders
        for _ in range(200):
            candidates = list(range(1, 23))
            random.shuffle(candidates)
            
            current_set = []
            current_sums = {0}
            
            for x in candidates:
                if len(current_set) >= k:
                    break
                w = weights[x]
                # Check if adding x causes collision
                # New sums will be current_sums U {s + w for s in current_sums}
                # Collision if (s1 + w) == s2 for some s1, s2 in current_sums
                # Or (s1 + w) == (s2 + w) which implies s1 == s2 (already distinct)
                # So we just need to check if {s + w for s in current_sums} intersects current_sums
                
                # Optimization:
                # If we maintain current_sums as a set, we can check intersection.
                # But we need to update current_sums if valid.
                
                # Check collision
                valid = True
                new_sums_set = set()
                for s in current_sums:
                    ns = s + w
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums_set.add(ns)
                
                if valid:
                    # Merge new_sums_set into current_sums
                    current_sums.update(new_sums_set)
                    current_set.append(x)
                    
            if len(current_set) == k:
                max_found = current_set
                found = True
                break
                
        if found:
            break
            
    return max_found

# EVOLVE-BLOCK-END
