"""Stronger search for K4-free graphs with no independent 6-set (R(4,6) lower bound)."""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 145.0
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _has_k4(adj, n):
    """Return True if graph has a K4. adj is list of bitmasks."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not (adj[i] >> k) & 1 or not (adj[j] >> k) & 1:
                    continue
                # common neighbors of i,j,k
                common = adj[i] & adj[j] & adj[k]
                # remove i,j,k themselves
                common &= ~((1 << i) | (1 << j) | (1 << k))
                if common:
                    return True
    return False

def _has_independent_set_6(adj, n):
    """Return True if graph has an independent set of size 6.
    Equivalent to clique of size 6 in complement."""
    # Build complement adj
    comp_adj = []
    full = (1 << n) - 1
    for i in range(n):
        # complement: all except self and neighbors
        c = full & ~(adj[i] | (1 << i))
        comp_adj.append(c)
    # Look for K6 in complement
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp_adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not (comp_adj[i] >> k) & 1 or not (comp_adj[j] >> k) & 1:
                    continue
                for l in range(k + 1, n):
                    if not (comp_adj[i] >> l) & 1 or not (comp_adj[j] >> l) & 1 or not (comp_adj[k] >> l) & 1:
                        continue
                    common = comp_adj[i] & comp_adj[j] & comp_adj[k] & comp_adj[l]
                    common &= ~((1 << i) | (1 << j) | (1 << k) | (1 << l))
                    if common:
                        return True
    return False

def _is_valid(adj, n):
    if _has_k4(adj, n):
        return False
    if _has_independent_set_6(adj, n):
        return False
    return True

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _random_adj(n, p):
    """Generate a random graph with edge probability p."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _repair(adj, n, max_iters=20000):
    """Targeted repair: fix K4s and independent 6-sets by flipping edges."""
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        if _has_k4(adj, n):
            # Find a K4 and remove one edge from it
            k4 = _find_k4(adj, n)
            if k4 is None:
                break
            # Remove a random edge from the K4
            a, b, c, d = k4
            pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        elif _has_independent_set_6(adj, n):
            # Find an independent 6-set and add an edge within it
            indep = _find_independent_set_6(adj, n)
            if indep is None:
                break
            # Add a random edge within the independent set
            u, v = random.choice(list(combinations(indep, 2)))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            break
    return adj

def _find_k4(adj, n):
    """Find and return a K4 (tuple of 4 vertices) or None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not (adj[i] >> k) & 1 or not (adj[j] >> k) & 1:
                    continue
                common = adj[i] & adj[j] & adj[k]
                common &= ~((1 << i) | (1 << j) | (1 << k))
                if common:
                    # find one bit
                    l = common.bit_length() - 1
                    return (i, j, k, l)
    return None

def _find_independent_set_6(adj, n):
    """Find and return an independent set of 6 vertices or None."""
    full = (1 << n) - 1
    comp_adj = []
    for i in range(n):
        c = full & ~(adj[i] | (1 << i))
        comp_adj.append(c)
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp_adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not (comp_adj[i] >> k) & 1 or not (comp_adj[j] >> k) & 1:
                    continue
                for l in range(k + 1, n):
                    if not (comp_adj[i] >> l) & 1 or not (comp_adj[j] >> l) & 1 or not (comp_adj[k] >> l) & 1:
                        continue
                    common = comp_adj[i] & comp_adj[j] & comp_adj[k] & comp_adj[l]
                    common &= ~((1 << i) | (1 << j) | (1 << k) | (1 << l))
                    if common:
                        m = common.bit_length() - 1
                        return (i, j, k, l, m)
    return None

def _local_search(adj, n, max_iters=50000):
    """Local search: flip random edges, keep if valid or closer to valid."""
    best_adj = adj[:]
    best_score = _score(adj, n)
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        # Flip a random edge
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            continue
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        s = _score(adj, n)
        if s > best_score:
            best_score = s
            best_adj = adj[:]
            if best_score == 0:
                break
        else:
            # revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
    return best_adj, best_score

def _score(adj, n):
    """Score: number of violations (K4s + independent 6-sets). Lower is better."""
    # Count K4s (approximate by finding one)
    k4_count = 0
    if _has_k4(adj, n):
        k4_count = 1  # Just flag
    indep_count = 0
    if _has_independent_set_6(adj, n):
        indep_count = 1
    return k4_count + indep_count

def _construct_for_n(n):
    """Try to find a valid graph on n vertices."""
    # Start with random graphs at various densities
    best_result = None
    best_score = float('inf')
    
    # Try a range of densities
    for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if _time_left() < 10.0:
            break
        for attempt in range(50):
            if _time_left() < 10.0:
                break
            adj = _random_adj(n, p)
            # Repair
            adj = _repair(adj, n, max_iters=5000)
            s = _score(adj, n)
            if s == 0:
                return adj
            if s < best_score:
                best_score = s
                best_result = adj[:]
    
    # If we have a near-valid graph, do local search on it
    if best_result is not None and _time_left() > 20.0:
        adj, score = _local_search(best_result, n, max_iters=20000)
        if score == 0:
            return adj
        if score < best_score:
            best_result = adj
            best_score = score
    
    return best_result

def construct():
    global _START
    _START = time.time()
    
    # Try from high n down to low n
    # The Paley graph on 17 is known valid. We try to go higher.
    # Known record is 35. We'll try 35 down to 17.
    
    # First, ensure we have at least the 17-vertex Paley graph
    squares = {x * x % 17 for x in range(1, 17)}
    paley_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    paley_adj = [0] * 17
    for i, j in paley_edges:
        paley_adj[i] |= (1 << j)
        paley_adj[j] |= (1 << i)
    
    # Verify Paley is valid
    if not _is_valid(paley_adj, 17):
        # Shouldn't happen, but fallback
        pass
    
    best_n = 17
    best_adj = paley_adj[:]
    
    # Try higher n values
    for n in range(35, 17, -1):
        if _time_left() < 30.0:
            break
        result = _construct_for_n(n)
        if result is not None:
            if _is_valid(result, n):
                best_n = n
                best_adj = result
                # If we found a valid graph, we could try even higher, but we're going down
                # Actually we're going from 35 down, so if 35 works, we return immediately
                break
    
    # Try to improve: if we found a valid graph at n, try n+1
    while best_n < 40 and _time_left() > 30.0:
        n_try = best_n + 1
        result = _construct_for_n(n_try)
        if result is not None and _is_valid(result, n_try):
            best_n = n_try
            best_adj = result
        else:
            break
    
    edges = _adj_to_edges(best_adj, best_n)
    return {'n': best_n, 'edges': edges}
