"""Rewritten for the R(4,6)-style task: maximize n of a K4-free graph with alpha <= 5.

Approach (CIRCULANT GRAPH SEARCH):
  * Exploit vertex-transitive symmetry: a circulant graph on n vertices is
    defined by a connection set S ⊆ {1, ..., floor(n/2)}.
  * For each n (from 35 down), search over circulant graphs via local search
    on the connection set (add/remove elements of S) to eliminate K4s and
    independent 6-sets.
  * Use bitmask adjacency for fast exact checks.
  * Fall back to random local search on general graphs if circulant search
    doesn't find a valid graph.
  * An explicit wall-clock deadline guard returns the best-so-far before the limit.
"""

import random
import time

_START = time.time()
_DEADLINE = _START + 115.0

_RNG = random.Random(20240607)


def _time_left():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Exact checks (bitmask adjacency: adj[i] is an int bitmask of neighbours of i)
# ---------------------------------------------------------------------------

def _find_k4(adj, n):
    """Return a 4-tuple of vertices forming a K4, or None if K4-free."""
    for u in range(n):
        au = adj[u]
        vv = au & (~((1 << (u + 1)) - 1))
        while vv:
            v = (vv & -vv).bit_length() - 1
            vv &= vv - 1
            c = au & adj[v] & (~((1 << (v + 1)) - 1))
            if not c:
                continue
            cc = c
            while cc:
                x = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                ymask = adj[x] & c
                if ymask:
                    y = (ymask & -ymask).bit_length() - 1
                    return [u, v, x, y]
    return None


def _find_clique(adj, n, k):
    """Return a k-clique (list of k vertices) in the graph (adj,n), or None."""
    full = (1 << n) - 1
    found = [None]

    def rec(cand, start, chosen):
        if found[0] is not None:
            return
        if len(chosen) == k:
            found[0] = list(chosen)
            return
        need = k - len(chosen)
        usable = cand & (~((1 << start) - 1))
        if usable.bit_count() < need:
            return
        c = usable
        while c:
            v = (c & -c).bit_length() - 1
            c &= c - 1
            chosen.append(v)
            rec(cand & adj[v], v + 1, chosen)
            chosen.pop()
            if found[0] is not None:
                return

    rec(full, 0, [])
    return found[0]


def _find_indep6(adj, n):
    """Return an independent set of 6 vertices in (adj,n), or None if alpha < 6."""
    full = (1 << n) - 1
    comp = [full & ~adj[i] & ~(1 << i) for i in range(n)]
    return _find_clique(comp, n, 6)


def _is_valid(adj, n):
    return _find_k4(adj, n) is None and _find_indep6(adj, n) is None


# ---------------------------------------------------------------------------
# Circulant graph utilities
# ---------------------------------------------------------------------------

def _build_circulant_adj(n, S):
    """Build adjacency bitmask list for circulant graph with connection set S."""
    adj = [0] * n
    for i in range(n):
        for s in S:
            j1 = (i + s) % n
            j2 = (i - s) % n
            if j1 != i:
                adj[i] |= (1 << j1)
            if j2 != i:
                adj[i] |= (1 << j2)
    return adj


def _circulant_to_dict(n, S):
    """Convert circulant graph to the required output format."""
    adj = _build_circulant_adj(n, S)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


# ---------------------------------------------------------------------------
# Local search on circulant connection sets
# ---------------------------------------------------------------------------

def _repair_circulant(n, S, max_iters=50000):
    """Local search on the connection set S for a circulant graph on n vertices.
    
    Returns (S, ok) where ok is True iff the resulting graph is valid.
    """
    S = set(S)
    half = n // 2
    possible = set(range(1, half + 1))
    
    for it in range(max_iters):
        if _time_left() <= 5.0:
            break
        
        adj = _build_circulant_adj(n, S)
        k4 = _find_k4(adj, n)
        if k4 is not None:
            # Remove a random element from S to break the K4
            if S:
                s_remove = _RNG.choice(list(S))
                S.discard(s_remove)
            else:
                # S is empty, graph has no edges; add a random one
                if possible:
                    s_add = _RNG.choice(list(possible))
                    S.add(s_add)
            continue
        
        i6 = _find_indep6(adj, n)
        if i6 is not None:
            # Add a random element from S to break the independent set
            candidates = possible - S
            if candidates:
                s_add = _RNG.choice(list(candidates))
                S.add(s_add)
            else:
                # S is full, remove a random one
                if S:
                    s_remove = _RNG.choice(list(S))
                    S.discard(s_remove)
            continue
        
        # No violation found -> valid
        return S, True
    
    adj = _build_circulant_adj(n, S)
    return S, _is_valid(adj, n)


def _search_circulant(n, max_iters_per_start=30000, num_starts=20):
    """Try to find a valid circulant graph on n vertices via local search.
    
    Returns (S, ok) or (None, False).
    """
    half = n // 2
    possible = set(range(1, half + 1))
    
    # Try multiple random initial connection sets
    best_S = None
    best_ok = False
    
    for start in range(num_starts):
        if _time_left() <= 10.0:
            break
        
        # Random initial S with density around 0.4-0.6
        density = _RNG.uniform(0.3, 0.7)
        S_init = set()
        for s in possible:
            if _RNG.random() < density:
                S_init.add(s)
        
        S, ok = _repair_circulant(n, S_init, max_iters=max_iters_per_start)
        if ok:
            return S, True
        
        # Track the "best" S (fewest violations) - simplified: just keep last
        best_S = S
        best_ok = ok
    
    return best_S, best_ok


# ---------------------------------------------------------------------------
# General graph local search (fallback)
# ---------------------------------------------------------------------------

def _repair(adj, n, max_iters=200000):
    """Flip random edges to eliminate K4s and independent 6-sets."""
    for _ in range(max_iters):
        if _time_left() <= 3.0:
            break
        k4 = _find_k4(adj, n)
        if k4 is not None:
            a, b = _RNG.sample(k4, 2)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            continue
        i6 = _find_indep6(adj, n)
        if i6 is not None:
            a, b = _RNG.sample(i6, 2)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            continue
        return adj, n, True
    return adj, n, _is_valid(adj, n)


def _init_random(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _RNG.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _grow(adj, n, q):
    """Try to add one vertex with random neighbourhood (prob q) and re-repair."""
    new_n = n + 1
    new_adj = adj + [0]
    for i in range(n):
        if _RNG.random() < q:
            new_adj[i] |= (1 << new_n)
            new_adj[new_n] |= (1 << i)
    new_adj, new_n, ok = _repair(new_adj, new_n)
    return new_adj, new_n, ok


def _to_dict(n, adj):
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


# ---------------------------------------------------------------------------
# Fallback: the seed Paley graph on 17 vertices (known valid).
# ---------------------------------------------------------------------------

def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17,
            'edges': [[i, j] for i in range(17) for j in range(i + 1, 17)
                      if (j - i) % 17 in squares]}


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def _search_portfolio():
    best = None  # (n, result_dict)
    
    # Phase 1: Circulant graph search from n=35 down to n=20
    for n in range(35, 19, -1):
        if _time_left() <= 15.0:
            break
        if best is not None and n <= best[0]:
            continue
        
        S, ok = _search_circulant(n, max_iters_per_start=20000, num_starts=10)
        if ok:
            result = _circulant_to_dict(n, S)
            # Verify
            adj = _build_circulant_adj(n, S)
            if _is_valid(adj, n):
                best = (n, result)
                # Found a valid graph; try to go higher
                continue
    
    # Phase 2: If circulant search found something, try to grow via general search
    if best is not None and _time_left() > 12.0:
        n_found = best[0]
        # Try to grow from the circulant graph
        S_best = None
        # Reconstruct S from the found result - actually, let's just do general growth
        # from a random start at n_found+1
        for target_n in range(n_found + 1, min(n_found + 4, 36)):
            if _time_left() <= 12.0:
                break
            for p in (0.45, 0.5, 0.55):
                if _time_left() <= 12.0:
                    break
                adj = _init_random(target_n, p)
                adj, n2, ok = _repair(adj, target_n, max_iters=30000)
                if ok and n2 > best[0]:
                    result = _to_dict(n2, adj)
                    best = (n2, result)
    
    # Phase 3: General random local search if circulant didn't find anything good
    if (best is None or best[0] < 30) and _time_left() > 20.0:
        plan = []
        for n in range(32, 19, -1):
            if best is not None and n <= best[0]:
                continue
            for p in (0.45, 0.5, 0.55, 0.6):
                plan.append((n, p))
        _RNG.shuffle(plan)
        
        for n, p in plan:
            if _time_left() <= 15.0:
                break
            if best is not None and n <= best[0]:
                continue
            adj = _init_random(n, p)
            adj, n2, ok = _repair(adj, n, max_iters=25000)
            if ok and (best is None or n2 > best[0]):
                best = (n2, _to_dict(n2, adj))
    
    return best


def construct():
    result = None
    try:
        best = _search_portfolio()
        if best is not None:
            n, result = best
    except Exception:
        result = None

    if result is None:
        result = _paley17()

    return result
