import time, math, random

def construct():
    t0 = time.time()
    LIMIT = 55.0
    def now(): return time.time()
    
    all_m = list(range(2, 129))
    n = len(all_m)
    
    # Precompute gcd matrix
    G = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(all_m[i], all_m[j])
            G[i][j] = g
            G[j][i] = g
    
    # Find all primes in [2,128]
    def is_prime(x):
        if x < 2: return False
        for d in range(2, int(x**0.5)+1):
            if x % d == 0: return False
        return True
    
    prime_indices = [i for i in range(n) if is_prime(all_m[i])]
    composite_indices = [i for i in range(n) if not is_prime(all_m[i])]
    
    def find_valid_residue(i, chosen):
        """Find a residue r for modulus all_m[i] compatible with all in chosen."""
        m = all_m[i]
        for r in range(m):
            ok = True
            for (j, r2) in chosen:
                g = G[i][j]
                if g >= 2 and (r % g) == (r2 % g):
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def verify(chosen):
        for a in range(len(chosen)):
            for b in range(a+1, len(chosen)):
                i, r = chosen[a]
                j, r2 = chosen[b]
                g = G[i][j]
                if g >= 2 and (r % g) == (r2 % g):
                    return False
        return True
    
    best = None
    best_len = 0
    
    # Strategy: start with primes (always compatible), add composites
    # Try multiple residue assignments for primes to maximize composite additions
    
    random.seed(42)
    
    for trial in range(200):
        if now() - t0 > LIMIT * 0.8:
            break
        
        # Assign residues to primes
        chosen = []
        for pi in prime_indices:
            m = all_m[pi]
            r = random.randint(0, m-1)
            chosen.append((pi, r))
        
        # Try to add composites
        added = True
        while added:
            added = False
            for ci in composite_indices:
                if any(ci == j for j, _ in chosen):
                    continue
                r = find_valid_residue(ci, chosen)
                if r is not None:
                    chosen.append((ci, r))
                    added = True
        
        if verify(chosen) and len(chosen) > best_len:
            best_len = len(chosen)
            best = chosen[:]
    
    # Also try: start with primes residue 0, then add composites
    chosen = [(pi, 0) for pi in prime_indices]
    added = True
    while added:
        added = False
        for ci in composite_indices:
            if any(ci == j for j, _ in chosen):
                continue
            r = find_valid_residue(ci, chosen)
            if r is not None:
                chosen.append((ci, r))
                added = True
    if verify(chosen) and len(chosen) > best_len:
        best_len = len(chosen)
        best = chosen[:]
    
    # Try swapping: remove a prime, add its multiples
    for trial in range(100):
        if now() - t0 > LIMIT * 0.9:
            break
        
        base = [(pi, 0) for pi in prime_indices]
        # Remove one prime
        idx = random.choice(prime_indices)
        chosen = [(i, r) for i, r in base if i != idx]
        
        # Add composites
        added = True
        while added:
            added = False
            for ci in composite_indices:
                if any(ci == j for j, _ in chosen):
                    continue
                r = find_valid_residue(ci, chosen)
                if r is not None:
                    chosen.append((ci, r))
                    added = True
        
        if verify(chosen) and len(chosen) > best_len:
            best_len = len(chosen)
            best = chosen[:]
    
    if best is None or not verify(best):
        # Fallback: just primes
        best = [(pi, 0) for pi in prime_indices]
    
    result = [[r, all_m[i]] for (i, r) in best]
    return result
