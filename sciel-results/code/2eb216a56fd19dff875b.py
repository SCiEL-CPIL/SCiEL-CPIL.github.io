import time

def _precompute_conflicts(N):
    """
    Precompute for each a in [1, N] the list of pairs (b, c) with b < c
    such that 1/a = 1/b + 1/c and b, c in [1, N].
    We store conflicts[a] = list of (b, c).
    """
    conflicts = {a: [] for a in range(1, N + 1)}
    for a in range(1, N + 1):
        a2 = a * a
        # b > a, c > a
        # (b-a)(c-a) = a^2
        # Let x = b-a, y = c-a. xy = a^2.
        # x < y to ensure b < c (and b != c)
        # x must be a divisor of a^2
        # x < a (since x < y => x^2 < a^2 => x < a)
        # Also b = a+x <= N => x <= N-a
        # c = a+y <= N => y <= N-a => a^2/x <= N-a => x >= a^2/(N-a)
        
        min_x = 1
        if N > a:
            min_x = max(1, (a2 + (N - a) - 1) // (N - a))
        else:
            # If N <= a, no b,c > a exist in [1, N]
            continue
            
        for x in range(min_x, a):
            if a2 % x == 0:
                y = a2 // x
                if x < y: # Ensures b != c
                    b = a + x
                    c = a + y
                    if b <= N and c <= N:
                        conflicts[a].append((b, c))
    return conflicts

def construct():
    N = 180
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # 1. Find the smallest K such that [K, N] is valid
    # A suffix [K, N] is valid if for all a in [K, N], 
    # there are no b, c in [K, N] such that (b-a)(c-a)=a^2.
    # Note: if a in [K, N], then b, c > a >= K, so b, c are in [K, N] automatically 
    # if b, c <= N.
    # So we just need to check if any conflict pair for a has both elements >= K.
    
    # Precompute conflicts for all a
    conflicts = _precompute_conflicts(N)
    
    best_K = N
    for K in range(1, N + 1):
        valid = True
        for a in range(K, N + 1):
            for b, c in conflicts[a]:
                # Since b, c > a >= K, b and c are automatically >= K
                # So if the pair exists in the precomputed list (which ensures b,c <= N),
                # it is a conflict in the suffix [K, N].
                # Wait, the precomputed list contains ALL pairs in [1, N].
                # If we have a pair (b,c) for a, and a >= K, then b,c > a >= K.
                # So the pair is entirely within [K, N].
                # Thus, if conflicts[a] is non-empty, the suffix [K, N] is INVALID.
                if conflicts[a]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_K = K
            break
    
    # 2. Start with the suffix [best_K, N]
    S = set(range(best_K, N + 1))
    
    # 3. Greedily add smaller elements
    # We check from best_K - 1 down to 1
    for a in range(best_K - 1, 0, -1):
        if time.time() > deadline:
            break
        # Check if a can be added
        # a can be added if no conflict pair (b, c) for a is fully contained in S
        # Note: b, c > a. Since S currently contains elements >= best_K > a,
        # we just need to check if (b, c) is a subset of S.
        
        can_add = True
        for b, c in conflicts[a]:
            if b in S and c in S:
                can_add = False
                break
        
        if can_add:
            S.add(a)
            
    return list(S)
