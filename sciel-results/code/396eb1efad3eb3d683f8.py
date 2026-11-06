import time

def construct():
    N = 180
    deadline = time.time() + 85  # Safety margin
    
    # 1. Find the largest valid suffix
    # S_K = {K, ..., N}
    def is_suffix_valid(K):
        S = set(range(K, N + 1))
        # Check for triples
        # For each a in S, check if there exist distinct b, c in S such that 1/a = 1/b + 1/c
        # b = a+x, c = a+y, xy = a^2
        for a in range(K, N + 1):
            a2 = a * a
            max_x = N - a
            if max_x < 1:
                continue
            # Find divisors x of a^2 such that x <= max_x
            # and y = a^2/x <= max_x and x != y
            # Iterate x from 1 to sqrt(a^2) = a
            # But we only care about x in [1, max_x]
            limit = min(a, max_x)
            for x in range(1, limit + 1):
                if a2 % x == 0:
                    y = a2 // x
                    if y <= max_x and x != y:
                        # Check if b, c are in S
                        b = a + x
                        c = a + y
                        # Since b, c > a >= K, they are in [K, N]
                        # We just need to ensure they are distinct from each other (x!=y)
                        # and in S (which they are if <= N)
                        return False
        return True

    best_K = N
    for K in range(N, 0, -1):
        if time.time() > deadline:
            break
        if is_suffix_valid(K):
            best_K = K
            # Since we go downwards, we want the smallest K (largest set)
            # But wait, if K is smaller, the set is larger.
            # We want to MINIMIZE K to MAXIMIZE size.
            # So we should stop at the first valid K?
            # No, if K=100 is valid, K=99 might be invalid.
            # We want the smallest K such that S_K is valid.
            # So we iterate K from 1 to N?
            # No, S_K gets larger as K decreases.
            # If S_K is valid, S_{K+1} is a subset, so also valid.
            # So the set of valid K's is an interval [K_min, N].
            # We want K_min.
            # So we can binary search or just iterate from 1 upwards?
            # Actually, if K=1 is valid, size 180.
            # If K=90 is valid, size 91.
            # We want the smallest K.
            pass
    
    # Let's iterate K from 1 to N and find the smallest valid one.
    # But is it true that if S_K is valid, S_{K-1} might be invalid?
    # Yes, adding K-1 might introduce a triple.
    # So we find the smallest K such that S_K is valid.
    
    K_min = N
    for K in range(1, N + 1):
        if time.time() > deadline:
            break
        if is_suffix_valid(K):
            K_min = K
            break # Found the smallest K (largest suffix)
            
    S = set(range(K_min, N + 1))
    
    # 2. Try to add elements from 1 to K_min-1
    # Use a greedy approach: try to add each element if it doesn't break validity.
    # To check if adding x to S is valid, we check if x creates a triple with existing elements.
    # Triple (a,b,c) involving x.
    # x can be a, b, or c.
    # Case 1: x is a (smallest). 1/x = 1/b + 1/c. b,c in S.
    # Case 2: x is b (larger). 1/a = 1/x + 1/c. a,c in S, a < x.
    # Case 3: x is c (larger). 1/a = 1/b + 1/x. a,b in S, a < x.
    
    def can_add(x, S):
        # x not in S
        # Check if adding x creates a triple
        # 1. x is a: 1/x = 1/b + 1/c
        # b, c in S, b != c, b > x, c > x
        # b = x+u, c = x+v, uv = x^2
        x2 = x * x
        max_u = N - x
        limit = min(x, max_u)
        for u in range(1, limit + 1):
            if x2 % u == 0:
                v = x2 // u
                if v <= max_u and u != v:
                    b = x + u
                    c = x + v
                    if b in S and c in S:
                        return False
        
        # 2. x is b: 1/a = 1/x + 1/c
        # a in S, a < x. c in S, c > x (since 1/c < 1/x < 1/a? No, 1/a = 1/x+1/c > 1/x => a<x)
        # c = ax / (x-a)
        for a in S:
            if a >= x:
                continue
            num = a * x
            den = x - a
            if num % den == 0:
                c = num // den
                if c in S and c != x and c != a:
                    return False
        
        # 3. x is c: 1/a = 1/b + 1/x
        # a in S, b in S. a < b < x? Or a < x < b?
        # 1/a = 1/b + 1/x => 1/a > 1/b => a < b.
        # 1/a > 1/x => a < x.
        # So a < b and a < x.
        # b = ax / (x-a)
        # We need b in S.
        # Also b != x, b != a.
        for a in S:
            if a >= x:
                continue
            num = a * x
            den = x - a
            if num % den == 0:
                b = num // den
                if b in S and b != x and b != a:
                    return False
        
        return True

    # Greedily add elements
    for x in range(1, K_min):
        if time.time() > deadline:
            break
        if x not in S and can_add(x, S):
            S.add(x)
            
    return list(S)
