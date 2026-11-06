import time

def construct():
    N = 256
    deadline = time.time() + 80
    
    S = []
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        # Case 1: x is a
        # Need b, c in S with b > x and c >= b such that x does not divide b+c
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b
        # Need a, c in S with a < x and c >= x such that a does not divide x+c
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c
        # Need a, b in S with a < b < x such that a does not divide b+x
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b >= x or b <= a:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    # Greedy from N down to 1
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        
        if can_add(x, S):
            S.append(x)
            
    return S
