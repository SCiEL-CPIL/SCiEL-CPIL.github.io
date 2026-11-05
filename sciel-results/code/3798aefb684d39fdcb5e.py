"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def _sq_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _is_iso(p, q, r):
    d1 = _sq_dist(p, q)
    d2 = _sq_dist(p, r)
    d3 = _sq_dist(q, r)
    return d1 == d2 or d1 == d3 or d2 == d3

def _greedy_blocking(all_points, rng):
    """
    Greedy construction using a 'max blocking' heuristic.
    At each step, pick the point that, when added, blocks the most other available points.
    """
    S = []
    S_set = set()
    avail = list(all_points)
    avail_set = set(all_points)
    
    # Precompute distances between all pairs for speed?
    # 256 points -> 256*256 = 65536 distances.
    # Let's map point to index.
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    n_points = len(all_points)
    
    # dist[i][j] = squared distance
    dist = [[0]*n_points for _ in range(n_points)]
    for i in range(n_points):
        pi = all_points[i]
        for j in range(i+1, n_points):
            pj = all_points[j]
            d = _sq_dist(pi, pj)
            dist[i][j] = d
            dist[j][i] = d
            
    # is_iso_idx[i][j][k] is too big? 256^3 = 16M booleans. 16MB. Acceptable.
    # But calculating it takes time. Let's calculate on the fly or use a faster check.
    # Check: d(i,j) == d(i,k) or d(i,j) == d(j,k) or d(j,k) == d(i,k)
    
    while avail:
        best_p = None
        best_blocking = -1
        best_q_to_remove = None # To avoid recalculating for the winner
        
        # We need to evaluate blocking for each candidate in avail
        # Blocking of p: number of q in avail (q!=p) such that exists s in S_set 
        # where is_iso(p, q, s) is True.
        
        # Optimization: 
        # If S is empty, no q is blocked by p (since no s exists). 
        # So initially, all p have blocking 0. We pick randomly.
        # Once S has points, we can compute blocking.
        
        if not S_set:
            # Random pick
            idx = rng.randrange(len(avail))
            p = avail[idx]
            S.append(p)
            S_set.add(p)
            avail.pop(idx)
            avail_set.remove(p)
            continue
            
        # For each candidate p in avail, calculate how many q in avail it blocks
        # q is blocked by p if exists s in S such that is_iso(p, q, s)
        
        # We can iterate q first? 
        # Let's stick to p-first for clarity, but optimize the inner loop.
        
        # To speed up, we can precompute for each p, the set of q that form an iso triangle with ANY s in S?
        # No, S changes.
        
        # Let's just do it. 256 candidates * 256 avail * |S| checks.
        # |S| grows.
        
        scores = {}
        for p in avail:
            pi = point_to_idx[p]
            block_count = 0
            # Check which q in avail are blocked by p
            # q is blocked if exists s in S such that is_iso(p, q, s)
            # We can break early if we just need the count, but we need the count.
            
            # Optimization: 
            # For a fixed p, q is blocked if q is in the union of "iso-neighbors" of p for each s in S.
            # Let Iso(p, s) = { q | is_iso(p, q, s) }
            # Then Blocked(p) = Avail ∩ (Union_{s in S} Iso(p, s))
            
            # Precomputing Iso(p, s) for all p, s is 256*256 sets of size ~256.
            # That's 65k sets.
            
            # Let's just check on the fly.
            for q in avail:
                if q == p:
                    continue
                qi = point_to_idx[q]
                # Check if exists s in S such that is_iso(p, q, s)
                blocked = False
                for s in S:
                    si = point_to_idx[s]
                    # is_iso(p, q, s)
                    if dist[pi][qi] == dist[pi][si] or dist[pi][qi] == dist[qi][si] or dist[qi][si] == dist[pi][si]:
                        blocked = True
                        break
                if blocked:
                    block_count += 1
            
            scores[p] = block_count
            
        # Find max score
        max_score = -1
        candidates = []
        for p, score in scores.items():
            if score > max_score:
                max_score = score
                candidates = [p]
            elif score == max_score:
                candidates.append(p)
                
        p = rng.choice(candidates)
        
        # Add p to S
        S.append(p)
        S_set.add(p)
        
        # Remove p and all blocked q from avail
        pi = point_to_idx[p]
        new_avail = []
        for q in avail:
            if q == p:
                continue
            qi = point_to_idx[q]
            # Check if q is blocked by p given S
            blocked = False
            for s in S:
                si = point_to_idx[s]
                if dist[pi][qi] == dist[pi][si] or dist[pi][qi] == dist[qi][si] or dist[qi][si] == dist[pi][si]:
                    blocked = True
                    break
            if not blocked:
                new_avail.append(q)
                
        avail = new_avail
        avail_set = set(avail)
        
    return S

def _is_maximal(S, all_points, dist, point_to_idx):
    S_set = set(S)
    for p in all_points:
        if p in S_set:
            continue
        pi = point_to_idx[p]
        # Check if p can be added
        can_add = True
        for i in range(len(S)):
            si = point_to_idx[S[i]]
            for j in range(i+1, len(S)):
                sj = point_to_idx[S[j]]
                if dist[pi][si] == dist[pi][sj] or dist[pi][si] == dist[si][sj] or dist[si][sj] == dist[pi][sj]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _minimize_by_deletion(S, all_points, dist, point_to_idx):
    S = list(S)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            reduced = S[:i] + S[i+1:]
            if _is_maximal(reduced, all_points, dist, point_to_idx):
                S = reduced
                changed = True
                break
    return S

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    n_points = len(all_points)
    
    dist = [[0]*n_points for _ in range(n_points)]
    for i in range(n_points):
        pi = all_points[i]
        for j in range(i+1, n_points):
            pj = all_points[j]
            d = _sq_dist(pi, pj)
            dist[i][j] = d
            dist[j][i] = d
            
    best = None
    best_size = len(all_points) + 1
    start = time.time()
    deadline = start + 280 # Safety margin
    
    rng = random.Random(42)
    
    # Run multiple instances of the greedy blocking heuristic
    for attempt in range(500):
        if time.time() > deadline - 20:
            break
            
        S = _greedy_blocking(all_points, rng)
        S = _minimize_by_deletion(S, all_points, dist, point_to_idx)
        
        if len(S) < best_size:
            best_size = len(S)
            best = S
            
    return best

# EVOLVE-BLOCK-END

