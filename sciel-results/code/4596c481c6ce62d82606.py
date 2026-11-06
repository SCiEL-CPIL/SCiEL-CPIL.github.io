import time
import random

_START = time.time()
_DEADLINE = 135  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _k4_count(adj, n):
    """Count K4s (or return early with count). Use bitmask for speed."""
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            # common neighbors > j
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            # count edges among common
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                # edges from k to higher in common
                higher = common & ~((1 << (k + 1)) - 1)
                cnt += (adj[k] & higher).bit_count()
                c ^= lsb
    return cnt

def _has_k4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                higher = common & ~((1 << (k + 1)) - 1)
                if adj[k] & higher:
                    return True
                c ^= lsb
    return False

def _has_ind_6(adj, n):
    """Backtracking with pruning; returns True if independent set of 6 exists."""
    # Precompute non-neighbor masks
    non_adj = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        non_adj[i] = full & ~adj[i] & ~(1 << i)
    
    def bt(cands, chosen):
        if len(chosen) == 6:
            return True
        if cands.bit_count() < 6 - len(chosen):
            return False
        # pick vertex with fewest non-neighbors in cands (most constrained)
        best_v = -1
        best_cnt = n + 1
        c = cands
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (non_adj[v] & cands).bit_count()
            if cnt < best_cnt:
                best_cnt = cnt
                best_v = v
            if cnt == 0:
                break
            c ^= lsb
        if best_v < 0:
            return False
        # branch: include best_v
        new_cands = cands & non_adj[best_v]
        if bt(new_cands, chosen + [best_v]):
            return True
        # branch: exclude best_v
        return bt(cands & ~(1 << best_v), chosen)
    
    return bt(full, [])

def _paley17():
    p = 17
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj

def _try_extend(adj, n, time_limit, rng):
    """Try to extend graph on n vertices to n+1 by adding a vertex with chosen adjacency.
    Returns (new_adj, new_n) if successful, else (None, n)."""
    start = time.time()
    # We want to add vertex n with some adjacency mask to 0..n-1.
    # Constraint: no K4 in new graph, no Ind6 in new graph.
    # K4 involving new vertex: new vertex + 3 existing that form a triangle and all adjacent to new.
    #   i.e., if new is adjacent to {a,b,c} and {a,b,c} form a triangle -> K4.
    # Ind6 involving new vertex: new + 5 existing that are pairwise non-adjacent and all non-adjacent to new.
    #   i.e., if new is non-adjacent to an independent set of 5 -> Ind6.
    
    # Precompute triangles among existing vertices (for K4 check)
    # And independent sets of size 5 (for Ind6 check) - expensive, so do on the fly.
    
    # Strategy: try random adjacency masks, but biased.
    # A good heuristic: new vertex should be adjacent to ~half the vertices.
    # Also, to avoid K4, new vertex's neighbors should be an independent set (no triangles among them).
    # To avoid Ind6, new vertex's non-neighbors should have no independent set of 5.
    
    # Let's try a "greedy + random" approach:
    # 1. Start with a random mask.
    # 2. Check constraints. If violated, flip bits to fix.
    
    full_old = (1 << n) - 1
    
    for attempt in range(2000):
        if time.time() - start > time_limit:
            break
        
        # Generate candidate adjacency mask for new vertex
        # Bias: prefer masks where neighbors form a triangle-free set
        mask = 0
        for i in range(n):
            mask |= (rng.random() < 0.5) << i
        
        # Check K4: neighbors of new vertex must contain no triangle
        # i.e., for any triangle (a,b,c) in old graph, not all three can be in mask
        ok = True
        # Iterate triangles in old graph
        # Precompute triangle list? Too slow for each attempt. Do on the fly with bitmasks.
        # For each pair (i,j) in mask that is an edge, check if there's a k in mask adjacent to both.
        m = mask
        while m:
            lsb_i = m & -m
            i = lsb_i.bit_length() - 1
            m ^= lsb_i
            # neighbors of i in mask
            ni = adj[i] & mask
            while ni:
                lsb_j = ni & -ni
                j = lsb_j.bit_length() - 1
                ni ^= lsb_j
                if j < i:
                    continue
                # common neighbors of i and j in mask, with index > j
                common = adj[i] & adj[j] & mask & ~((1 << (j + 1)) - 1)
                if common:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        
        # Check Ind6: non-neighbors of new vertex must have no independent set of 5
        non_mask = full_old & ~mask
        # Check if non_mask contains an independent set of size 5
        # Use backtracking
        def has_ind5(cands):
            if cands.bit_count() < 5:
                return False
            # pick vertex with fewest non-neighbors in cands
            best_v = -1
            best_cnt = n + 1
            c = cands
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                # non-neighbors of v in cands
                nn = cands & ~adj[v] & ~(1 << v)
                cnt = nn.bit_count()
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_v = v
                    if cnt == 0:
                        break
                c ^= lsb
            if best_v < 0:
                return False
            # include best_v
            if has_ind5(cands & ~adj[best_v] & ~(1 << best_v)):
                return True
            # exclude best_v
            return has_ind5(cands & ~(1 << best_v))
        
        if has_ind5(non_mask):
            continue
        
        # Valid extension!
        new_adj = adj + [0]
        new_adj[n] = mask
        for i in range(n):
            if (mask >> i) & 1:
                new_adj[i] |= 1 << n
        return new_adj, n + 1
    
    return None, n

def construct():
    rng = random.Random(42)
    adj = _paley17()
    n = 17
    best_adj = adj[:]
    best_n = n
    
    # Try to extend up to n=30 or until time runs out
    # Use a time budget for each extension attempt
    # If an extension fails, try a few times with different random seeds before giving up on that n
    
    current_time = _elapsed()
    for target_n in range(n + 1, 31):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time per extension: more time for smaller n, less for larger
        # Total remaining time / remaining steps
        remaining_steps = 31 - target_n
        time_per_ext = min(3.0, max(0.5, (_DEADLINE - 5 - _elapsed()) / max(1, remaining_steps)))
        
        success = False
        # Try multiple times with different random seeds
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 5:
                break
            new_adj, new_n = _try_extend(adj, n, time_per_ext, rng)
            if new_adj is not None:
                adj = new_adj
                n = new_n
                best_adj = adj[:]
                best_n = n
                success = True
                break
        
        if not success:
            # Try a completely random restart from Paley17 for this target?
            # Or just break, since we can't grow further
            # Let's try one more time with a fresh random seed and slightly more time
            if _elapsed() < _DEADLINE - 10:
                new_adj, new_n = _try_extend(_paley17(), 17, _DEADLINE - 10 - _elapsed(), rng)
                if new_adj is not None:
                    # This only gets us to 18, not helpful if we already have more
                    # But if we were stuck at 17, this helps
                    if new_n > best_n:
                        adj = new_adj
                        n = new_n
                        best_adj = adj[:]
                        best_n = n
    
    # Build edge list from best_adj
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
