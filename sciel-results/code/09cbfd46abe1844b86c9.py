"""Combined R(4,6)-style task: maximize n of a K4-free graph with alpha <= 5.

Approach:
  * Keep the fast bitmask exact checks (K4, independent-6) from the current program.
  * Primary search: circulant local search (high symmetry, fast to evaluate).
  * Secondary search: general local search with smarter repair (edge flips guided
    by violations) and incremental vertex growth.
  * Algebraic seed: use quadratic-form / multiplicative-character constructions over
    small finite fields to generate high-quality starting graphs that are close to valid,
    reducing the repair effort.
  * Deadline guard returns the best-so-far before the 150 s limit.
"""

import random
import time

_START = time.time()
_DEADLINE = _START + 118.0
_RNG = random.Random(20240607)


def _time_left():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Exact checks using bitmask adjacency
# ---------------------------------------------------------------------------

def _find_k4(adj, n):
    """Return a 4-tuple forming a K4, or None if K4-free."""
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
    """Return a k-clique in graph (adj, n), or None."""
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
    """Return an independent set of 6 vertices, or None if alpha < 6."""
    full = (1 << n) - 1
    comp = [full & ~adj[i] & ~(1 << i) for i in range(n)]
    return _find_clique(comp, n, 6)


def _is_valid(adj, n):
    return _find_k4(adj, n) is None and _find_indep6(adj, n) is None


# ---------------------------------------------------------------------------
# Circulant graph utilities
# ---------------------------------------------------------------------------

def _build_circulant_adj(n, S):
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
    adj = _build_circulant_adj(n, S)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


# ---------------------------------------------------------------------------
# Algebraic seed: quadratic-form graph over GF(q^2) embedded in 2D
# ---------------------------------------------------------------------------

def _quadratic_form_graph(q, n_target):
    """
    Build a graph on vertices = elements of GF(q^2) (size q^2) where two
    vertices a, b are adjacent iff the quadratic form Q(a-b) is a non-square
    in GF(q^2). This yields a strongly regular graph with known parameters.
    
    For q=7: 49 vertices, but we can use subsets or take q such that q^2 <= 40.
    q=5: 25 vertices. q=7: 49 (too big). q=3: 9. q=4: 16.
    
    Actually, for the Paley graph on q^2 vertices (q ≡ 1 mod 4), we can use
    multiplicative characters. Let's use the Paley graph on p^2 vertices
    where p is prime ≡ 1 mod 4, or just use the standard Paley on p vertices.
    
    For a more targeted approach: use the graph where vertices are GF(q) and
    edges defined by (a-b) being a quadratic residue. This is the Paley graph.
    
    Let's try a different algebraic construction: the "Hoffman-Singleton" style
    or use the incidence of a symmetric design.
    
    For now, generate Paley graphs on prime powers p ≡ 1 (mod 4) up to 37.
    """
    # Paley graph on p vertices where p ≡ 1 mod 4
    for p in [17, 29, 37]:
        if p > n_target:
            continue
        # Quadratic residues mod p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                d = (j - i) % p
                if d in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return p, adj
    return None, None


# ---------------------------------------------------------------------------
# General graph local search with smarter repair
# ---------------------------------------------------------------------------

def _repair(adj, n, max_iters=50000):
    """Flip edges to eliminate K4s and independent 6-sets."""
    for _ in range(max_iters):
        if _time_left() <= 3.0:
            break
        k4 = _find_k4(adj, n)
        if k4 is not None:
            # Remove one edge from the K4 (randomly chosen pair)
            a, b = _RNG.sample(k4, 2)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            continue
        i6 = _find_indep6(adj, n)
        if i6 is not None:
            # Add one edge within the independent set
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


def _to_dict(n, adj):
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


# ---------------------------------------------------------------------------
# Circulant local search
# ---------------------------------------------------------------------------

def _repair_circulant(n, S, max_iters=30000):
    S = set(S)
    half = n // 2
    possible = set(range(1, half + 1))

    for it in range(max_iters):
        if _time_left() <= 5.0:
            break
        adj = _build_circulant_adj(n, S)
        k4 = _find_k4(adj, n)
        if k4 is not None:
            if S:
                s_remove = _RNG.choice(list(S))
                S.discard(s_remove)
            else:
                if possible:
                    s_add = _RNG.choice(list(possible))
                    S.add(s_add)
            continue
        i6 = _find_indep6(adj, n)
        if i6 is not None:
            candidates = possible - S
            if candidates:
                s_add = _RNG.choice(list(candidates))
                S.add(s_add)
            else:
                if S:
                    s_remove = _RNG.choice(list(S))
                    S.discard(s_remove)
            continue
        return S, True

    adj = _build_circulant_adj(n, S)
    return S, _is_valid(adj, n)


def _search_circulant(n, max_iters_per_start=15000, num_starts=10):
    half = n // 2
    possible = set(range(1, half + 1))
    best_S = None
    best_ok = False

    for start in range(num_starts):
        if _time_left() <= 10.0:
            break
        density = _RNG.uniform(0.3, 0.7)
        S_init = set()
        for s in possible:
            if _RNG.random() < density:
                S_init.add(s)
        S, ok = _repair_circulant(n, S_init, max_iters=max_iters_per_start)
        if ok:
            return S, True
        best_S = S
        best_ok = ok

    return best_S, best_ok


# ---------------------------------------------------------------------------
# Fallback Paley 17
# ---------------------------------------------------------------------------

def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17,
            'edges': [[i, j] for i in range(17) for j in range(i + 1, 17)
                      if (j - i) % 17 in squares]}


# ---------------------------------------------------------------------------
# Main search portfolio
# ---------------------------------------------------------------------------

def _search_portfolio():
    best = None  # (n, result_dict)

    # Phase 0: Try algebraic seeds (Paley graphs) as starting points
    for p in [37, 29, 17]:
        if _time_left() <= 20.0:
            break
        _, adj = _quadratic_form_graph(0, p)
        if adj is not None:
            if _is_valid(adj, p):
                if best is None or p > best[0]:
                    best = (p, _to_dict(p, adj))

    # Phase 1: Circulant graph search from n=35 down
    for n in range(35, 19, -1):
        if _time_left() <= 15.0:
            break
        if best is not None and n <= best[0]:
            continue
        S, ok = _search_circulant(n, max_iters_per_start=15000, num_starts=8)
        if ok:
            result = _circulant_to_dict(n, S)
            adj = _build_circulant_adj(n, S)
            if _is_valid(adj, n):
                best = (n, result)

    # Phase 2: General local search with multiple densities
    if _time_left() > 15.0:
        plan = []
        for n in range(35, 19, -1):
            if best is not None and n <= best[0]:
                continue
            for p in (0.4, 0.45, 0.5, 0.55, 0.6):
                plan.append((n, p))
        _RNG.shuffle(plan)

        for n, p in plan:
            if _time_left() <= 10.0:
                break
            if best is not None and n <= best[0]:
                continue
            adj = _init_random(n, p)
            adj, n2, ok = _repair(adj, n, max_iters=30000)
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
