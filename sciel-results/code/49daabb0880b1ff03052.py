def construct():
    n = 128
    L = set(range(65, n + 1))
    T = set()
    
    # Precompute validity check for T subset
    # A set S = T | L is valid if no forbidden triple exists.
    # We only check triples involving elements in T.
    
    def is_valid(T):
        S = T | L
        # Check triples with 2 small, 1 large
        smalls = list(T)
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                u = smalls[i]
                v = smalls[j]
                s1 = u + v
                if s1 not in S:
                    continue
                # Need a in L such that u+a in S and v+a in S
                # a is in 65..128
                # u+a in S => u+a in L (since u+a > 64 usually)
                # v+a in S => v+a in L
                # So we need a in L such that a <= 128-u and a <= 128-v
                # and a in L.
                # Since L is 65..128, such a exists if 65 <= 128-min(u,v)
                # i.e., min(u,v) <= 63.
                if min(u, v) <= 63:
                    # There exists a (e.g. 65) such that sums are in L
                    # Wait, we need a SPECIFIC a in L.
                    # If min(u,v) <= 63, let a = 65.
                    # u+65 <= 128? Yes if u<=63.
                    # v+65 <= 128? Yes if v<=63.
                    # So if u,v <= 63 and u+v in S, then (u,v,65) is a forbidden triple.
                    return False
        
        # Check triples with 3 smalls
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                for k in range(j+1, len(smalls)):
                    u, v, w = smalls[i], smalls[j], smalls[k]
                    if (u+v in S) and (u+w in S) and (v+w in S):
                        return False
        return True

    candidates = list(range(1, 65))
    # Greedy: add if valid. To improve, we can try to add those that allow more others.
    # But simple greedy might be enough.
    # Let's try sorting by value? Or just iterate.
    
    # A better heuristic: Iterate through candidates.
    # If adding x keeps it valid, add it.
    # To maximize, we should probably add larger smalls first?
    # Or smaller?
    # If u+v >= 65, it's dangerous.
    # Small u,v have small sums.
    # If u+v < 65, s1 not in L. s1 must be in T.
    # So if u+v < 65, we need u+v in T.
    
    # Let's just do a simple greedy: try to add each candidate if possible.
    # Order: 1 to 64.
    for x in candidates:
        if is_valid(T | {x}):
            T.add(x)
            
    return sorted(T | L)
