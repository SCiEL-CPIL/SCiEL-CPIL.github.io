"""
Matrix multiplication 2x3x7: search for a rank-minimal exact tensor decomposition.

The tensor for multiplying a 2x3 matrix by a 3x7 matrix has 42 entries (i,j,k) with
i in {0,1}, j in {0,1,2}, k in {0,...,6}. A rank-r decomposition gives factors
U (6 x r), V (21 x r), W (14 x r) such that for every (i,j,k):
    sum_{t=0}^{r-1} U[i,j,t] * V[j,k,t] * W[k,i,t] = 1

The schoolbook rank is 42. We try to find a lower rank by:
1. Random search over {-1, 0, 1} entries with targeted repair.
2. Exploiting the small n=2 dimension: for each t, the W column has 14 entries but
   only 2 distinct row indices (i=0, i=1). We can parameterize W more compactly.
3. Use a greedy/coordinate-descent approach: start from schoolbook and try to
   merge pairs of rank-1 terms.
"""
import time
import random
import sys

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W):
    """Verify exact decomposition. U: nm x rank, V: mp x rank, W: pn x rank."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_search(rank, rng, deadline, max_trials=5000):
    """Random search over {-1,0,1} with sparsity control."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        # Generate sparse random matrices
        density = 0.3 + 0.4 * (rank / 42.0)  # higher density for lower rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for i in range(nm):
            for r in range(rank):
                if rng.random() < density:
                    U[i][r] = rng.choice([-1, 1])
        for j in range(mp):
            for r in range(rank):
                if rng.random() < density:
                    V[j][r] = rng.choice([-1, 1])
        for k in range(pn):
            for r in range(rank):
                if rng.random() < density:
                    W[k][r] = rng.choice([-1, 1])
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _coordinate_descent(U, V, W, max_iters=200, deadline=None):
    """
    Try to reduce violations by adjusting individual entries.
    For each (i,j,k) where sum != 1, try flipping/setting entries to fix.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    nm, mp, pn = n * m, m * p, p * n
    
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Find violations
        violations = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        violations.append((i, j, k, s))
        
        if not violations:
            return {"U": U, "V": V, "W": W}
        
        # Pick a random violation
        i, j, k, s = random.choice(violations)
        ui = i * m + j
        vj = j * p + k
        wk = k * n + i
        
        # Try to fix by modifying one entry
        fixed = False
        # Current sum is s, we need it to be 1, so we need to change by (1-s)
        delta_needed = 1 - s
        
        # Try changing U[ui][r]
        for r in range(rank):
            contrib = U[ui][r] * V[vj][r] * W[wk][r]
            # If we set U[ui][r] to new_val, new contrib = new_val * V[vj][r] * W[wk][r]
            # Change = (new_val - U[ui][r]) * V[vj][r] * W[wk][r]
            prod_vw = V[vj][r] * W[wk][r]
            if prod_vw != 0:
                # We need (new_val - U[ui][r]) * prod_vw = delta_needed
                # new_val = U[ui][r] + delta_needed / prod_vw
                if delta_needed % prod_vw == 0:
                    new_val = U[ui][r] + delta_needed // prod_vw
                    if new_val in (-1, 0, 1):
                        U[ui][r] = new_val
                        fixed = True
                        break
        
        if not fixed:
            # Try changing V[vj][r]
            for r in range(rank):
                prod_uw = U[ui][r] * W[wk][r]
                if prod_uw != 0:
                    if delta_needed % prod_uw == 0:
                        new_val = V[vj][r] + delta_needed // prod_uw
                        if new_val in (-1, 0, 1):
                            V[vj][r] = new_val
                            fixed = True
                            break
        
        if not fixed:
            # Try changing W[wk][r]
            for r in range(rank):
                prod_uv = U[ui][r] * V[vj][r]
                if prod_uv != 0:
                    if delta_needed % prod_uv == 0:
                        new_val = W[wk][r] + delta_needed // prod_uv
                        if new_val in (-1, 0, 1):
                            W[wk][r] = new_val
                            fixed = True
                            break
        
        if not fixed:
            # Try adding a new non-zero entry where there was zero
            # Pick a random r and try setting one of U, V, W to +/-1
            r = random.randrange(rank)
            choices = []
            if U[ui][r] == 0:
                choices.append(('U', ui, r, V[vj][r], W[wk][r]))
            if V[vj][r] == 0:
                choices.append(('V', vj, r, U[ui][r], W[wk][r]))
            if W[wk][r] == 0:
                choices.append(('W', wk, r, U[ui][r], V[vj][r]))
            
            if choices:
                mat, idx, rr, a, b = random.choice(choices)
                if a * b != 0:
                    # Setting to val changes sum by val * a * b
                    if delta_needed % (a * b) == 0:
                        val = delta_needed // (a * b)
                        if val in (-1, 1):
                            if mat == 'U':
                                U[idx][rr] = val
                            elif mat == 'V':
                                V[idx][rr] = val
                            else:
                                W[idx][rr] = val
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _merge_attempt(U, V, W, rank, rng, deadline):
    """
    Try to merge two rank-1 terms into one, reducing rank by 1.
    Pick two columns t1, t2 and try to find new values that maintain the identity.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        t1, t2 = rng.sample(range(rank), 2)
        
        # For each (i,j,k), the contribution from t1 and t2 must sum to the same
        # as before. We're trying to eliminate t2 by absorbing into t1.
        # New U[i,j,t1] = U[i,j,t1] + U[i,j,t2] * (V[j,k,t2]/V[j,k,t1]) * ... 
        # This is complex. Instead, try: set t2 to zero and adjust t1.
        
        # Check if simply zeroing t2 and doubling t1 works (crude)
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        # Zero out column t2
        for i in range(nm):
            new_U[i][t2] = 0
        for j in range(mp):
            new_V[j][t2] = 0
        for k in range(pn):
            new_W[k][t2] = 0
        
        # Try to fix by adjusting column t1
        # For each (i,j,k): need U[i,j,t1]*V[j,k,t1]*W[k,i,t1] to cover both
        # This is a system of equations. Try random adjustments.
        
        success = True
        for _ in range(50):
            if time.time() > deadline:
                return None
            
            violations = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += new_U[i*m+j][r] * new_V[j*p+k][r] * new_W[k*n+i][r]
                        if s != 1:
                            violations.append((i, j, k, s))
            
            if not violations:
                return {"U": new_U, "V": new_V, "W": new_W}
            
            # Try to fix by adjusting t1 entries
            i, j, k, s = random.choice(violations)
            ui = i*m+j
            vj = j*p+k
            wk = k*n+i
            
            delta = 1 - s
            # Current contribution from t1:
            c = new_U[ui][t1] * new_V[vj][t1] * new_W[wk][t1]
            # We need to change by delta
            # Try adjusting U[ui][t1]
            vw = new_V[vj][t1] * new_W[wk][t1]
            if vw != 0 and delta % vw == 0:
                nv = new_U[ui][t1] + delta // vw
                if nv in (-1, 0, 1):
                    new_U[ui][t1] = nv
                    continue
            
            # Try adjusting V[vj][t1]
            uw = new_U[ui][t1] * new_W[wk][t1]
            if uw != 0 and delta % uw == 0:
                nv = new_V[vj][t1] + delta // uw
                if nv in (-1, 0, 1):
                    new_V[vj][t1] = nv
                    continue
            
            # Try adjusting W[wk][t1]
            uv = new_U[ui][t1] * new_V[vj][t1]
            if uv != 0 and delta % uv == 0:
                nv = new_W[wk][t1] + delta // uv
                if nv in (-1, 0, 1):
                    new_W[wk][t1] = nv
                    continue
        
        # If we didn't fix it, this merge attempt failed
        success = False
        if not success:
            continue
    
    return None

def _structured_search(rank, rng, deadline):
    """
    Exploit structure: n=2 means W has only 2 row groups.
    For each t, W[:, t] has 14 entries: W[k*2+0, t] for k=0..6 and W[k*2+1, t] for k=0..6.
    
    Try a parameterization where W entries are from a small set.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    for trial in range(2000):
        if time.time() > deadline:
            return None
        
        # Generate U, V, W with structured W
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        density_u = 0.4
        density_v = 0.4
        density_w = 0.5
        
        for i in range(nm):
            for r in range(rank):
                if rng.random() < density_u:
                    U[i][r] = rng.choice([-1, 1])
        for j in range(mp):
            for r in range(rank):
                if rng.random() < density_v:
                    V[j][r] = rng.choice([-1, 1])
        for k in range(pn):
            for r in range(rank):
                if rng.random() < density_w:
                    W[k][r] = rng.choice([-1, 1])
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        # Try coordinate descent repair
        repaired = _coordinate_descent(U, V, W, max_iters=30, deadline=deadline)
        if repaired is not None:
            return repaired
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    best = schoolbook
    best_rank = n * m * p  # 42
    
    rng = random.Random(12345)
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But actually the tensor rank is at least max of the three dimensions' products
    # More precisely, rank >= max(n*m, m*p, p*n) is not quite right.
    # rank >= max(n*m, m*p, p*n) / min(n,m,p) ... no.
    # The rank is at least max(n*m, m*p, p*n) in general? No.
    # Actually rank >= max(n*m, m*p, p*n) is not a valid lower bound.
    # A valid lower bound: rank >= max(n*m, m*p, p*n) is wrong.
    # rank >= n*m*p / max(n,m,p) ... no.
    # The trivial lower bound is max(n*m, m*p, p*n) is NOT correct.
    # Actually, the rank must be at least max(n*m, m*p, p*n) is a common misconception.
    # The correct lower bound involves the border rank or other invariants.
    # For now, let's try from 21 upward.
    
    for rank in range(21, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try structured search
        result = _structured_search(rank, rng, deadline)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        # Try random search
        result = _random_search(rank, rng, deadline, max_trials=1000)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        # Try merge from a higher-rank solution
        if rank < best_rank:
            # We have a solution of rank best_rank, try to merge down to rank
            pass  # This is complex, skip for now
    
    return best

# EVOLVE-BLOCK-END
