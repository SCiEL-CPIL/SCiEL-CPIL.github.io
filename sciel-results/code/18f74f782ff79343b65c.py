"""Local search for Schütte's problem k=4: minimize n such that every 4-set has a common dominator."""
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s

    # We'll search over n from 31 upward. For each n, run local search to find a
    # tournament where every 4-set has a common dominator.
    # Key insight: a 4-set {a,b,c,d} has a common dominator v iff there exists v
    # such that v->a, v->b, v->c, v->d all hold.
    # We use bitsets for fast computation.

    for n in range(31, 46):
        if time.time() > deadline:
            break
        # Try multiple random restarts
        best_bad = None
        best_orient = None
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            orient = _run_local_search(n, deadline, restart)
            if orient is None:
                continue
            bad = _count_bad_sets(orient, n)
            if bad == 0:
                return {'n': n, 'orientation': _orient_to_list(orient, n)}
            if best_bad is None or bad < best_bad:
                best_bad = bad
                best_orient = orient

    # If no solution found, fall back to Paley on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}


def _orient_to_list(orient, n):
    """Convert orientation bitset (numpy bool array) to list of bits in lexicographic order."""
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            # orient[i,j] = True means i -> j, so bit is 0
            # orient[i,j] = False means j -> i, so bit is 1
            result.append(0 if orient[i, j] else 1)
    return result


def _count_bad_sets(orient, n):
    """Count 4-sets that have NO common dominator.
    A 4-set {a,b,c,d} is "good" if some vertex v dominates all four.
    We check all vertices v: v must have edges to all of a,b,c,d.
    """
    # For each vertex v, compute the set of vertices it dominates (out-neighbors)
    # out[v] = bitmask of vertices v points to
    # A 4-set {a,b,c,d} is good if exists v such that a,b,c,d are all in out[v]
    # i.e., the 4-set is a subset of out[v] for some v.
    
    # For efficiency: for each 4-set, check if any v dominates all 4.
    # out_mask[v] = set of out-neighbors of v as a Python int bitmask
    out_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and orient[i, j]:
                mask |= (1 << j)
        out_mask[i] = mask

    # A 4-set {a,b,c,d} is good if for some v, all of a,b,c,d are in out_mask[v]
    # Equivalently, the 4-set is NOT good if for ALL v, at least one of a,b,c,d is NOT in out_mask[v]
    
    # Better: count good 4-sets. A 4-set is good if it's contained in out[v] for some v.
    # For each v, the number of 4-sets contained in out[v] is C(out_degree[v], 4).
    # But we need to count DISTINCT 4-sets that are covered by at least one v.
    # Use inclusion-exclusion or just check each 4-set.
    
    # For n up to ~40, C(n,4) is at most ~91390, manageable.
    bad = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Check if any v dominates all of a,b,c,d
                    found = False
                    for v in range(n):
                        if v == a or v == b or v == c or v == d:
                            continue
                        if (out_mask[v] & (1 << a)) and (out_mask[v] & (1 << b)) and \
                           (out_mask[v] & (1 << c)) and (out_mask[v] & (1 << d)):
                            found = True
                            break
                    if not found:
                        bad += 1
    return bad


def _count_bad_sets_fast(orient, n):
    """Faster count using numpy: for each 4-set, check if union of in-neighbors covers all 4."""
    # in_mask[v] = set of vertices that point TO v (in-neighbors)
    # A 4-set {a,b,c,d} has a common dominator iff there exists v such that
    # v is in in_mask[a] AND in_mask[b] AND in_mask[c] AND in_mask[d]
    # i.e., the intersection of in_mask[a], in_mask[b], in_mask[c], in_mask[d] is non-empty
    # (excluding a,b,c,d themselves, but v can't be in the set since v->v is false)
    
    in_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and orient[j, i]:  # j -> i
                in_mask[i] |= (1 << j)
    
    bad = 0
    for a in range(n):
        ma = in_mask[a]
        for b in range(a + 1, n):
            mb = in_mask[b]
            for c in range(b + 1, n):
                mc = in_mask[c]
                for d in range(c + 1, n):
                    md = in_mask[d]
                    # Intersection of all four in-masks
                    inter = ma & mb & mc & md
                    if inter == 0:
                        bad += 1
    return bad


def _run_local_search(n, deadline, seed):
    """Run local search on a tournament of n vertices. Returns orientation or None."""
    rng = np.random.RandomState(seed * 1000 + 42)
    
    # Random initial tournament
    orient = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                orient[i, j] = True  # i -> j
            else:
                orient[j, i] = True  # j -> i
    
    # Compute initial bad count
    bad = _count_bad_sets_fast(orient, n)
    
    if bad == 0:
        return orient
    
    # Local search: flip edges to reduce bad count
    # For each edge (i,j), flipping it changes the in/out masks.
    # We'll do coordinate descent: for each edge, try flipping and keep if it improves.
    
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        improved = False
        # Random order of edges for diversity
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        
        for (i, j) in edges:
            if time.time() > deadline - 5:
                break
            # Try flipping edge (i,j)
            # Current: one of orient[i,j] or orient[j,i] is True
            # Flipping: swap the direction
            old_i_j = orient[i, j]
            old_j_i = orient[j, i]
            
            # Flip
            orient[i, j] = old_j_i
            orient[j, i] = old_i_j
            
            new_bad = _count_bad_sets_fast(orient, n)
            
            if new_bad <= bad:
                bad = new_bad
                improved = True
                if bad == 0:
                    return orient
            else:
                # Revert
                orient[i, j] = old_i_j
                orient[j, i] = old_j_i
        
        if not improved:
            # Stuck; try a random perturbation
            if bad > 0 and n >= 5:
                # Flip a random edge
                i = rng.randint(n)
                j = rng.randint(n)
                if i != j:
                    orient[i, j] = orient[j, i]
                    orient[j, i] = orient[i, j]
                    bad = _count_bad_sets_fast(orient, n)
            else:
                break
    
    return orient if bad == 0 else None
