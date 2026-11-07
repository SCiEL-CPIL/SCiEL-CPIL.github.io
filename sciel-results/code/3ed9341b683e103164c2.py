def construct():
    n = 128
    L = set(range(65, n + 1))
    T = set()
    
    def is_valid(T):
        S = T | L
        smalls = list(T)
        # Check triples with 2 small, 1 large
        # a, b in T, c in L
        # Need a+b in S, a+c in S, b+c in S
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                u = smalls[i]
                v = smalls[j]
                s1 = u + v
                if s1 not in S:
                    continue
                # Find c in L such that u+c in S and v+c in S
                # u+c in S => u+c in L (since u+c > 64)
                # v+c in S => v+c in L
                # c in 65..128
                # u+c <= 128 => c <= 128-u
                # v+c <= 128 => c <= 128-v
                # c >= 65
                max_c = min(128 - u, 128 - v)
                if max_c >= 65:
                    # There exists at least one c (e.g., 65) if 65 <= max_c
                    # We just need to check if ANY c works.
                    # Since L is contiguous 65..128, if max_c >= 65, c=65 works?
                    # Check c=65: u+65 in L? Yes if u<=63.
                    # If u=64, u+65=129 not in L.
                    # So we need to check if there is ANY c in [65, max_c] such that u+c in L and v+c in L.
                    # u+c in L <=> 65 <= u+c <= 128 <=> 65-u <= c <= 128-u.
                    # v+c in L <=> 65-v <= c <= 128-v.
                    # Intersection: c in [max(65, 65-u, 65-v), min(128-u, 128-v)].
                    low = max(65, 65 - u, 65 - v)
                    high = min(128 - u, 128 - v)
                    if low <= high:
                        return False
        
        # Check triples with 3 smalls
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                for k in range(j+1, len(smalls)):
                    u, v, w = smalls[i], smalls[j], smalls[k]
                    if (u+v in S) and (u+w in S) and (v+w in S):
                        return False
        return True

    candidates = list(range(64, 0, -1))
    for x in candidates:
        if is_valid(T | {x}):
            T.add(x)
            
    return sorted(T | L)
