<<<<<<< SEARCH
"""Clique partition of two fixed dense random graphs via repeated maximum-clique extraction."""
import time

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly (Random(seed), u<v, rng.random()<p)."""
    from random import Random
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_clique(adj, n):
    """Maximum clique via Bron–Kerbosch with Tomita coloring + pivot (dense graphs)."""
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -deg[i])
    remap = [0] * n
    for new, old in enumerate(order):
        remap[old] = new
    m = n
    cand = [set() for _ in range(m)]
    for old in range(n):
        ni = remap[old]
        for v in adj[old]:
            cand[ni].add(remap[v])

    best = []
    best_size = 0

    def color_sort(R, P):
        nonlocal best, best_size
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best = list(R)
            return
        colored = []
        colors = {}
        avail = set(P)
        while avail:
            color = len(colored) + 1
            layer = set()
            for v in avail:
                if not (cand[v] & avail):
                    layer.add(v)
            for v in layer:
                avail.remove(v)
            if not layer:
                break
            colors.update((v, color) for v in layer)
            colored.append((layer, color))
        for layer, color in reversed(colored):
            if len(R) + color <= best_size:
                break
            for v in layer:
                R.append(v)
                P2 = set(P & cand[v])
                color_sort(R, P2)
                R.pop()
                P.discard(v)

    color_sort([], set(range(m)))

    if not best:
        return []
    inv = [0] * m
    for new, old in enumerate(order):
        inv[new] = old
    return [inv[x] for x in best]


def _max_clique_fallback(adj, n, deadline):
    """Greedy degeneracy-order clique: O(n^2) guaranteed, always a clique."""
    order = sorted(range(n), key=lambda i: len(adj[i]))
    pos = {v: i for i, v in enumerate(order)}
    R = []
    for v in order:
        if all(pos[u] > pos[v] for u in adj[v]):
            R.append(v)
    return R


def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = _build_adj(n, edges)
    rem = set(edges)
    cliques = []
    while rem:
        if time.time() >= deadline:
            break
        cl = _max_clique(adj, n)
        if len(cl) < 2:
            cl = _max_clique_fallback(adj, n, deadline)
        if len(cl) < 2:
            break
        used = set()
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                if a > b:
                    a, b = b, a
                used.add((a, b))
        covered = used & rem
        if not covered:
            break
        rem -= covered
        cliques.append(cl)
    # Any uncovered edges (only if we hit the deadline) become 2-cliques (valid cliques).
    for e in rem:
        cliques.append(list(e))
    return cliques


def construct():
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        dl = min(_DEADLINE, time.time() + 60.0)  # per-instance margin
        result.append(_solve_instance(n, p, seed, dl))
    return result
=======
"""Clique partition of two fixed dense random graphs via optimized max-clique extraction."""
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 135.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly (Random(seed), u<v, rng.random()<p)."""
    from random import Random
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _build_adj_bitmask(n, edges):
    """Build adjacency as list of integers (bitmasks)."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _max_clique_bitmask(adj, n, deadline):
    """Maximum clique via Bron-Kerbosch with Tomita coloring using bitmasks."""
    deg = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda i: -deg[i])
    remap = [0] * n
    for new, old in enumerate(order):
        remap[old] = new
    
    # Rebuild adjacency in remapped order
    cand = [0] * n
    for old in range(n):
        ni = remap[old]
        for v in range(n):
            if adj[old] & (1 << v):
                cand[ni] |= (1 << remap[v])
    
    best = []
    best_size = 0
    
    def bk(R_mask, P_mask):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if P_mask == 0:
            if R_mask.bit_count() > best_size:
                best_size = R_mask.bit_count()
                best = [i for i in range(n) if R_mask & (1 << i)]
            return
        
        # Tomita coloring
        avail = P_mask
        colored = []
        while avail:
            color = len(colored) + 1
            # Find independent set in avail (no edges between them)
            layer = 0
            temp = avail
            while temp:
                v_bit = temp & (-temp)
                v = v_bit.bit_length() - 1
                if not (cand[v] & temp):
                    layer |= v_bit
                temp &= temp - 1
            avail &= ~layer
            if layer:
                colored.append((layer, color))
        
        # Process layers in reverse
        for layer, color in reversed(colored):
            if R_mask.bit_count() + color <= best_size:
                break
            temp = layer
            while temp:
                v_bit = temp & (-temp)
                v = v_bit.bit_length() - 1
                temp &= temp - 1
                new_R = R_mask | v_bit
                new_P = P_mask & cand[v]
                bk(new_R, new_P)
                P_mask &= ~v_bit
        
    bk(0, (1 << n) - 1)
    
    if not best:
        return []
    inv = [0] * n
    for new, old in enumerate(order):
        inv[new] = old
    return [inv[x] for x in best]


def _greedy_clique_from_order(adj, n, order):
    """Greedy: build clique by adding vertices that connect to all existing clique members."""
    clique = []
    for v in order:
        if all(adj[v] & (1 << u) for u in clique):
            clique.append(v)
    return clique


def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = _build_adj_bitmask(n, edges)
    rem = set(edges)
    cliques = []
    
    # Try multiple strategies: max clique, then greedy with different orderings
    strategies = []
    
    # Strategy 1: Maximum clique
    cl = _max_clique_bitmask(adj, n, deadline)
    if len(cl) >= 2:
        strategies.append(cl)
    
    # Strategy 2: Greedy with degree ordering (descending)
    deg_order = sorted(range(n), key=lambda i: -adj[i].bit_count())
    strategies.append(_greedy_clique_from_order(adj, n, deg_order))
    
    # Strategy 3: Greedy with random orderings
    rng = _random.Random(seed + 1000)
    for _ in range(5):
        rand_order = list(range(n))
        rng.shuffle(rand_order)
        strategies.append(_greedy_clique_from_order(adj, n, rand_order))
    
    # Try each strategy, pick the one that covers the most remaining edges
    best_strategy = None
    best_covered = 0
    for cl in strategies:
        if len(cl) < 2:
            continue
        used = set()
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                if a > b:
                    a, b = b, a
                if (a, b) in rem:
                    used.add((a, b))
        if len(used) > best_covered:
            best_covered = len(used)
            best_strategy = cl
    
    while rem:
        if time.time() >= deadline:
            break
        
        if best_strategy and best_covered > 0:
            cl = best_strategy
            used = set()
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    used.add((a, b))
            covered = used & rem
            if covered:
                rem -= covered
                cliques.append(cl)
                best_strategy = None
                best_covered = 0
                continue
        
        # Regenerate strategies if needed
        strategies = []
        cl = _max_clique_bitmask(adj, n, deadline)
        if len(cl) >= 2:
            strategies.append(cl)
        
        deg_order = sorted(range(n), key=lambda i: -adj[i].bit_count())
        strategies.append(_greedy_clique_from_order(adj, n, deg_order))
        
        for _ in range(3):
            rand_order = list(range(n))
            rng.shuffle(rand_order)
            strategies.append(_greedy_clique_from_order(adj, n, rand_order))
        
        best_strategy = None
        best_covered = 0
        for cl in strategies:
            if len(cl) < 2:
                continue
            used = set()
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in rem:
                        used.add((a, b))
            if len(used) > best_covered:
                best_covered = len(used)
                best_strategy = cl
        
        if not best_strategy or best_covered == 0:
            # Fallback: just take any edge
            e = next(iter(rem))
            cliques.append(list(e))
            rem.discard(e)
            continue
    
    # Any uncovered edges become 2-cliques
    for e in rem:
        cliques.append(list(e))
    
    return cliques


def construct():
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        dl = min(_DEADLINE, time.time() + 65.0)  # per-instance margin
        result.append(_solve_instance(n, p, seed, dl))
    return result
>>>>>>> REPLACE
