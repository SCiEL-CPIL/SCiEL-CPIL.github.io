# EVOLVE-BLOCK-START

def construct():
    """
    Returns a set of distinct positive integers <= 512 such that no element
    is the arithmetic mean of any subset of at least two other elements.
    
    Uses a greedy approach to build a set that avoids 2, 3, and 4-element means.
    This heuristic is expected to produce a valid set of size ~20-25.
    """
    S = []
    # Precompute pairs and triples for speed? Or just loop.
    
    for i in range(1, 513):
        # Check if i can be added to S
        valid = True
        
        # 1. Check if i is a mean of a subset of S
        # Subset size k >= 2
        # We only check k=2, 3, 4 explicitly. 
        # For k=2: 2*i = a+b. 
        # For k=3: 3*i = a+b+c.
        # For k=4: 4*i = a+b+c+d.
        
        # Efficient check for k=2:
        # a+b = 2*i. 
        # We can use a set for O(1) lookup.
        S_set = set(S)
        
        # k=2 for i
        for a in S:
            b = 2 * i - a
            if b in S_set and b != a: # a and b distinct? 
                # Wait, subset elements must be distinct from i. 
                # a, b are from S. i is not in S.
                # Are a and b distinct? 
                # If a == b, then 2a = 2i => a=i. But i not in S. So a!=b always if a,b in S.
                # However, the subset must be of size 2. So we need 2 distinct elements.
                # If a == b, we need two copies of a? No, distinct elements in set.
                # So a and b must be distinct.
                if a != b:
                    valid = False
                    break
        if not valid:
            continue
            
        # k=3 for i
        # 3*i = a+b+c. a,b,c distinct in S.
        # We can iterate pairs a,b and check if c = 3*i - a - b is in S and distinct from a,b.
        # Optimization: Precompute pair sums of S?
        # Since we do this for every i, and S changes, we can just do O(|S|^2) per i.
        # |S| ~ 20. 20^2 = 400. 512 * 400 = 200k. Fast.
        
        found_k3 = False
        for idx_a in range(len(S)):
            a = S[idx_a]
            for idx_b in range(idx_a + 1, len(S)):
                b = S[idx_b]
                c = 3 * i - a - b
                if c in S_set and c != a and c != b:
                    found_k3 = True
                    break
            if found_k3:
                break
        if found_k3:
            continue
            
        # k=4 for i
        # 4*i = a+b+c+d.
        # O(|S|^3) might be 20^3 = 8000. 512 * 8000 = 4M. Still fast.
        found_k4 = False
        for idx_a in range(len(S)):
            a = S[idx_a]
            for idx_b in range(idx_a + 1, len(S)):
                b = S[idx_b]
                for idx_c in range(idx_b + 1, len(S)):
                    c = S[idx_c]
                    d = 4 * i - a - b - c
                    if d in S_set and d != a and d != b and d != c:
                        found_k4 = True
                        break
                if found_k4:
                    break
            if found_k4:
                break
        if found_k4:
            continue
            
        # 2. Check if any y in S is a mean of a subset of S U {i}
        # Subset must contain i (since S is valid).
        # So subset is {i} U T, where T is subset of S \ {y}.
        # Sum(T) + i = k * y.
        # Sum(T) = k * y - i.
        
        for y in S:
            # k=2: 2y = i + a. a in S \ {y}.
            a = 2 * y - i
            if a in S_set and a != y:
                valid = False
                break
            
            if not valid:
                break
            
            # k=3: 3y = i + a + b. a,b in S \ {y}, a!=b.
            found_k3_y = False
            for idx_a in range(len(S)):
                a = S[idx_a]
                if a == y: continue
                for idx_b in range(idx_a + 1, len(S)):
                    b = S[idx_b]
                    if b == y: continue
                    if a == b: continue
                    if a + b + i == 3 * y:
                        found_k3_y = True
                        break
                if found_k3_y:
                    break
            if found_k3_y:
                valid = False
                break
            
            # k=4: 4y = i + a + b + c.
            found_k4_y = False
            for idx_a in range(len(S)):
                a = S[idx_a]
                if a == y: continue
                for idx_b in range(idx_a + 1, len(S)):
                    b = S[idx_b]
                    if b == y: continue
                    if a == b: continue
                    for idx_c in range(idx_b + 1, len(S)):
                        c = S[idx_c]
                        if c == y: continue
                        if a == c or b == c: continue
                        if a + b + c + i == 4 * y:
                            found_k4_y = True
                            break
                    if found_k4_y:
                        break
                if found_k4_y:
                    break
            if found_k4_y:
                valid = False
                break
        
        if valid:
            S.append(i)
            
    return S

# EVOLVE-BLOCK-END
