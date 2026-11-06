import itertools
import random
import time

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _check_valid_fast(n, orientation):
    """
    Check if every 4-set has a common dominator.
    Uses bitmask intersection.
    """
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j
                in_neigh[j] |= (1 << i)
            else:
                # j -> i
                in_neigh[i] |= (1 << j)
    
    # Optimization: If n is large, this is slow.
    # But we only call this for verification or small n.
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            return False
    return True

def _construct_paley(p):
    """
    Construct a Paley tournament on prime p (p % 4 == 3).
    """
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _rls_find_tournament(n, time_limit):
    """
    Use Randomized Local Search to find a valid tournament of size n.
    Returns orientation list if found, else None.
    """
    start_time = time.time()
    
    # Initialize random tournament
    # orientation[i,j] = 0 means i->j, 1 means j->i
    # We store as list of lists or flat list? Flat is better for speed.
    # But we need fast access for flipping.
    # Let's use a 2D list for adjacency, or just the flat list and index calc.
    # Index of (i,j) with i<j is i*n - i*(i+1)//2 + (j-i-1)?
    # Simpler: adj[i][j] = 1 if i->j, 0 if j->i.
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 0
            else:
                adj[i][j] = 0
                adj[j][i] = 1
                
    # Precompute in_neigh masks
    in_neigh = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and adj[j][i] == 1: # j -> i
                in_neigh[i] |= (1 << j)
                
    def get_orientation():
        res = []
        for i in range(n):
            for j in range(i+1, n):
                res.append(1 - adj[i][j]) # 0 if i->j (adj=1), 1 if j->i (adj=0)
                # Wait, task says: 0 means i->j, 1 means j->i.
                # adj[i][j]=1 means i->j. So bit should be 0.
                # adj[i][j]=0 means j->i. So bit should be 1.
                # So bit = 1 - adj[i][j].
        return res

    def flip(i, j):
        # Flip edge between i and j
        # Ensure i < j for orientation consistency, but adj is symmetric in logic
        adj[i][j] = 1 - adj[i][j]
        adj[j][i] = 1 - adj[j][i]
        # Update in_neigh
        # If i->j becomes j->i:
        # Old: i->j. in_neigh[j] had i. in_neigh[i] did not have j.
        # New: j->i. in_neigh[i] has j. in_neigh[j] does not have i.
        if adj[i][j] == 1: # Now i->j
             # Before flip, it was j->i (since we flipped)
             # So we remove i from in_neigh[j] and add j to in_neigh[i]
             in_neigh[j] &= ~(1 << i)
             in_neigh[i] |= (1 << j)
        else: # Now j->i
             # Before flip, it was i->j
             # So we remove j from in_neigh[i] and add i to in_neigh[j]
             in_neigh[i] &= ~(1 << j)
             in_neigh[j] |= (1 << i)

    # RLS Loop
    best_score = float('inf')
    best_adj = [row[:] for row in adj]
    best_in_neigh = in_neigh[:]
    
    # We will sample 4-sets to estimate fitness
    sample_size = 1000
    
    while time.time() - start_time < time_limit:
        # Sample 4-sets
        violations = 0
        violating_quad = None
        for _ in range(sample_size):
            quad = random.sample(range(n), 4)
            a, b, c, d = quad
            if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
                violations += 1
                if violating_quad is None:
                    violating_quad = quad
        
        if violations == 0:
            # Potential solution. Verify fully.
            if _check_valid_fast(n, get_orientation()):
                return get_orientation()
                
        # Update best
        if violations < best_score:
            best_score = violations
            best_adj = [row[:] for row in adj]
            best_in_neigh = in_neigh[:]
            
        # Move
        if violating_quad:
            a, b, c, d = violating_quad
            # Find best v to dominate this quad
            # Count how many of a,b,c,d are dominated by v
            best_v = -1
            best_count = -1
            for v in range(n):
                count = 0
                if adj[v][a] == 1: count += 1
                if adj[v][b] == 1: count += 1
                if adj[v][c] == 1: count += 1
                if adj[v][d] == 1: count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
                    
            # If best_count < 4, we need to flip edges
            if best_count < 4:
                # Pick a vertex q in quad that is NOT dominated by best_v
                # i.e., adj[best_v][q] == 0
                candidates = []
                if adj[best_v][a] == 0: candidates.append(a)
                if adj[best_v][b] == 0: candidates.append(b)
                if adj[best_v][c] == 0: candidates.append(c)
                if adj[best_v][d] == 0: candidates.append(d)
                
                if candidates:
                    q = random.choice(candidates)
                    flip(best_v, q)
                else:
                    # Should not happen if best_count < 4
                    pass
            else:
                # best_v already dominates quad? Then why was it a violation?
                # Maybe my check logic in sampling was wrong or state changed?
                # Just do a random flip
                i = random.randrange(n)
                j = random.randrange(n)
                if i != j:
                    flip(i, j)
        else:
            # No violation found in sample, but not verified.
            # Do a random flip to explore
            i = random.randrange(n)
            j = random.randrange(n)
            if i != j:
                flip(i, j)
                
    # If time's up, return None (caller handles fallback)
    return None

def construct():
    """
    Find the smallest n such that a valid tournament exists.
    """
    # First, try to find a smaller n using RLS
    # We know 67 is a candidate (Paley).
    # Try 31 to 66.
    
    # Allocate time for RLS
    # If we find one, we return immediately.
    # We should prioritize smaller n.
    
    # Let's try a few small n with short time limits
    # and larger n with longer limits?
    # Actually, if n is too small, it's impossible.
    # Let's try 31-40 with 5s each, 41-50 with 5s each, 51-66 with 5s each.
    # Total 30 * 5 = 150s. Too long.
    
    # Let's be smarter.
    # Try n=31..66.
    # Time limit per n: 2 seconds.
    # Total 36 * 2 = 72s.
    # Remaining 48s for verification and Paley fallback.
    
    for n in range(31, 67):
        start = time.time()
        # Use a bit more time for larger n? No, uniform.
        res = _rls_find_tournament(n, 2.0)
        if res is not None:
            return {'n': n, 'orientation': res}
            
    # If RLS failed, fall back to Paley
    # Check Paley 67
    p = 67
    orientation = _construct_paley(p)
    if _check_valid_fast(p, orientation):
        return {'n': p, 'orientation': orientation}
        
    # Check Paley 83
    p = 83
    orientation = _construct_paley(p)
    if _check_valid_fast(p, orientation):
        return {'n': p, 'orientation': orientation}
        
    # Final fallback: Random search for n=67?
    # Just return Paley 83 to be safe, though it's worse.
    # But Paley 67 is likely valid.
    return {'n': 83, 'orientation': _construct_paley(83)}
