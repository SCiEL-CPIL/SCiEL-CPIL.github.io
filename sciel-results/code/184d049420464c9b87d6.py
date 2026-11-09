import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # We'll work with a set for efficiency
    # Start with a good upper tail, then try to add smaller elements
    # The key constraint: for a < b <= c in S, a must NOT divide (b+c)
    
    def can_add(S, x):
        """Check if adding x to set S keeps it valid."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: x is the smallest (a=x), need b,c in S with x < b <= c
        #         Check: x does NOT divide (b+c) for all such pairs
        # Case 2: x is the middle (a < x <= c), a in S with a < x, c in S with c >= x
        #         Check: a does NOT divide (x+c)
        # Case 3: x is the largest (a < b <= x), a,b in S with a < b <= x
        #         Check: a does NOT divide (b+x)
        
        # Case 1: a = x
        larger = [v for v in S_list if v > x]
        for i in range(len(larger)):
            b = larger[i]
            for j in range(i, len(larger)):
                c = larger[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b = x (x is the middle element)
        # a < x, c >= x, a in S, c in S
        smaller = [v for v in S_list if v < x]
        larger_or_eq = [v for v in S_list if v >= x]
        for a in smaller:
            for c in larger_or_eq:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: c = x (x is the largest element)
        # a < b <= x, a,b in S
        smaller_or_eq = [v for v in S_list if v <= x]
        for i in range(len(smaller_or_eq)):
            a = smaller_or_eq[i]
            for j in range(i+1, len(smaller_or_eq)):
                b = smaller_or_eq[j]
                if (b + x) % a == 0:
                    return False
        
        return True
    
    # Start with upper tail
    S = set(range(128, 257))
    
    # Try to add elements from 127 down to 1
    for x in range(127, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if can_add(S, x):
            S.add(x)
    
    return sorted(S)
