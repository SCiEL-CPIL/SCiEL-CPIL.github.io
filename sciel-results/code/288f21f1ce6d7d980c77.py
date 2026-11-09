import itertools

def _is_valid_paley(p):
    # Construct Paley tournament for prime p
    # i -> j iff (j-i) % p is a quadratic residue
    squares = {x * x % p for x in range(1, p)}
    
    # Build adjacency matrix as list of sets or bitmasks for speed
    # Bitmask approach: adj[i] is an integer where bit j is 1 if i -> j
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
            else:
                adj[j] |= (1 << i)
                
    # Check all 4-subsets
    # Optimization: If any 4-subset fails, return False
    # Number of subsets is C(p, 4)
    # For p=67, C(67,4) ~ 730,000. 
    # Inner loop is constant time (bit checks).
    # Python might be slow for 730k * 4 checks. 
    # 730k * 4 * 3 ops ~ 8M ops. Should be < 1s in PyPy or CPython? 
    # CPython is slower. 8M ops might be 0.5-1s. 
    # We check p=31, 43, 47, 59, 67.
    # Total ops ~ 5 * 1s = 5s. Safe.
    
    for quad in itertools.combinations(range(p), 4):
        a, b, c, d = quad
        # Check if a dominates b,c,d
        if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)):
            continue
        # Check if b dominates a,c,d
        if (adj[b] & (1 << a)) and (adj[b] & (1 << c)) and (adj[b] & (1 << d)):
            continue
        # Check if c dominates a,b,d
        if (adj[c] & (1 << a)) and (adj[c] & (1 << b)) and (adj[c] & (1 << d)):
            continue
        # Check if d dominates a,b,c
        if (adj[d] & (1 << a)) and (adj[d] & (1 << b)) and (adj[d] & (1 << c)):
            continue
        # If none dominate, invalid
        return False
        
    return True

def construct():
    # Try smaller primes p = 3 mod 4
    # Known smallest is 67. Check if smaller ones work.
    candidates = [31, 43, 47, 59]
    
    for p in candidates:
        if _is_valid_paley(p):
            squares = {x * x % p for x in range(1, p)}
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    orientation.append(0 if (j - i) % p in squares else 1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            orientation.append(0 if (j - i) % p in squares else 1)
    return {'n': p, 'orientation': orientation}
