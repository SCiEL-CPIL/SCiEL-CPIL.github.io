import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 115.0  # 150s limit, ~35s safety margin

def _time_left():
    return DEADLINE - (time.time() - _START)

# Precompute all 4-cycles once (independent of graph)
def _precompute_cycles(n):
    cycles = []
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # 3 ways to make a 4-cycle on 4 labeled vertices
        cycles.append(((a, b), (b, c), (c, d), (d, a)))
        cycles.append(((a, b), (b, d), (d, c), (c, a)))
        cycles.append(((a, c), (c, b), (b, d), (d, a)))
    return cycles

_CYCLES = _precompute_cycles(20)

def _validate(edges):
    """Return True iff every 4-cycle that is fully present is rainbow."""
    edge_set = {}
    for u, v, c in edges:
        edge_set[(u, v)] = c
        edge_set[(v, u)] = c
    for cyc in _CYCLES:
        cols = []
        ok = True
        for (u, v) in cyc:
            c = edge_set.get((u, v))
            if c is None:
                ok = False
                break
            cols.append(c)
        if ok and len(set(cols)) != 4:
            return False
    return True

def _greedy_build(seed, initial_edges, order_type='random'):
    """Build a valid coloured graph, optionally starting from initial_edges."""
    n = 20
    num_colors = 20
    rng = random.Random(seed)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    adj = [[-1] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    edges = []

    if initial_edges:
        for u, v, c in initial_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
        existing = set((min(u, v), max(u, v)) for u, v, _ in edges)
        remaining = [e for e in all_edges if e not in existing]
    else:
        remaining = all_edges[:]

    if order_type == 'random':
        rng.shuffle(remaining)
    elif order_type == 'reverse':
        remaining.reverse()

    for (u, v) in remaining:
        impossible = False
        forbidden = set()
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
            if impossible:
                break
        if not impossible:
            # Choose color that minimizes future conflicts: prefer colors
            # that appear least among neighbors of u and v
            color_freq = {}
            for a in neighbors[u]:
                if a != v:
                    ca = adj[u][a]
                    color_freq[ca] = color_freq.get(ca, 0) + 1
            for b in neighbors[v]:
                if b != u:
                    cb = adj[v][b]
                    color_freq[cb] = color_freq.get(cb, 0) + 1
            available = [c for c in range(num_colors) if c not in forbidden]
            if available:
                # Sort by frequency (ascending), then randomize ties
                rng.shuffle(available)
                available.sort(key=lambda c: color_freq.get(c, 0))
                c = available[0]
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
    return edges

def _bipartite_seed(rng):
    """Build a K_{10,10}-style bipartite core with a linear colouring,
    removing edges to fix any non-rainbow 4-cycles."""
    n = 20
    A = list(range(10))
    B = list(range(10, 20))
    a = rng.randint(0, 19)
    b = rng.randint(1, 19)
    c0 = rng.randint(0, 19)

    def col(u, v):
        return (a * u + b * v + c0) % 20

    ec = {}
    for u in A:
        for v in B:
            ec[(u, v)] = col(u, v)
            ec[(v, u)] = col(u, v)

    bad = []
    for i in range(10):
        for j in range(i + 1, 10):
            u1, u2 = A[i], A[j]
            for k in range(10):
                for l in range(k + 1, 10):
                    v1, v2 = B[k], B[l]
                    cols = {ec[(u1, v1)], ec[(u2, v1)], ec[(u2, v2)], ec[(u1, v2)]}
                    if len(cols) < 4:
                        bad.append([(u1, v1), (u2, v1), (u2, v2), (u1, v2)])

    remove = set()
    rem = bad[:]
    while rem and _time_left() > 15.0:
        # Score each candidate edge by how many bad cycles it hits,
        # with a small random perturbation to break ties and explore
        best_e, best_score = None, -1
        cands = set()
        for cyc in rem:
            for e in cyc:
                if e not in remove:
                    cands.add(e)
        if not cands:
            break
        for e in cands:
            hits = sum(1 for cyc in rem if e in cyc)
            score = hits + rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_e = e
        if best_e is None:
            break
        remove.add(best_e)
        rem = [cyc for cyc in rem if best_e not in cyc]

    out = []
    for u in A:
        for v in B:
            if (u, v) not in remove:
                out.append([u, v, ec[(u, v)]])
    return out

def construct():
    rng = random.Random()
    best_edges = []
    best_count = 0

    # Phase 1: random greedy restarts (no seed)
    for ot in ('random', 'natural', 'reverse'):
        for i in range(60):
            if _time_left() < 20.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 2: bipartite-seeded greedy
    for i in range(20):
        if _time_left() < 20.0:
            break
        seed_edges = _bipartite_seed(rng)
        edges = _greedy_build(1000 + i, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Phase 3: local search with multiple move types
    current = best_edges[:]
    it = 0
    while _time_left() > 10.0:
        it += 1
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove 1-3 edges and rebuild
            k = random.randint(1, 3)
            if len(current) <= k:
                continue
            to_remove = random.sample(current, k)
            rset = set((min(u, v), max(u, v)) for u, v, _ in to_remove)
            base = [e for e in current if (min(e[0], e[1]), max(e[0], e[1])) not in rset]
            new_edges = _greedy_build(it * 131 + random.randint(0, 100), base, 'random')
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges
                current = new_edges[:]
        elif move_type < 0.7:
            # Try to add a missing edge with a valid color
            n = 20
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            existing = set((min(u, v), max(u, v)) for u, v, _ in current)
            missing = [e for e in all_edges if e not in existing]
            if missing:
                rng.shuffle(missing)
                # Build adjacency from current
                adj = [[-1] * n for _ in range(n)]
                neighbors = [set() for _ in range(n)]
                for u, v, c in current:
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                for (u, v) in missing[:20]:  # Try up to 20 candidates
                    forbidden = set()
                    possible = True
                    for a in neighbors[u]:
                        if a == v:
                            continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a:
                                continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                if ca == cb or cb == cv or ca == cv:
                                    possible = False
                                    break
                                forbidden.add(ca)
                                forbidden.add(cb)
                                forbidden.add(cv)
                        if not possible:
                            break
                    if possible:
                        available = [c for c in range(20) if c not in forbidden]
                        if available:
                            c = random.choice(available)
                            new_edges = current + [[u, v, c]]
                            if _validate(new_edges):
                                best_count = len(new_edges)
                                best_edges = new_edges
                                current = new_edges[:]
                                break
        else:
            # Recolor an existing edge
            if current:
                idx = random.randint(0, len(current) - 1)
                u, v, old_c = current[idx]
                n = 20
                adj = [[-1] * n for _ in range(n)]
                neighbors = [set() for _ in range(n)]
                for uu, vv, cc in current:
                    adj[uu][vv] = cc
                    adj[vv][uu] = cc
                    neighbors[uu].add(vv)
                    neighbors[vv].add(uu)
                adj[u][v] = -1
                adj[v][u] = -1
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                
                forbidden = set()
                possible = True
                for a in neighbors[u]:
                    if a == v:
                        continue
                    ca = adj[u][a]
                    for b in neighbors[v]:
                        if b == u or b == a:
                            continue
                        if adj[a][b] != -1:
                            cb = adj[a][b]
                            cv = adj[b][v]
                            if ca == cb or cb == cv or ca == cv:
                                possible = False
                                break
                            forbidden.add(ca)
                            forbidden.add(cb)
                            forbidden.add(cv)
                    if not possible:
                        break
                if possible:
                    available = [c for c in range(20) if c not in forbidden and c != old_c]
                    if available:
                        new_c = random.choice(available)
                        new_edges = [e[:] for e in current]
                        new_edges[idx] = [u, v, new_c]
                        if _validate(new_edges):
                            # Only accept if it enables more edges later
                            # For now, accept any valid recoloring
                            current = new_edges

    # Final safety: guarantee validity of what we return
    if not _validate(best_edges):
        # Fallback: rebuild a guaranteed-valid solution
        for ot in ('random', 'natural', 'reverse'):
            for i in range(20):
                if _time_left() < 2.0:
                    break
                e = _greedy_build(5000 + i, None, ot)
                if _validate(e) and len(e) > best_count:
                    best_count = len(e)
                    best_edges = e
        # If still invalid (shouldn't happen), return a trivial valid graph
        if not _validate(best_edges):
            best_edges = [[0, v, 0] for v in range(1, 20)]

    # Final polish: try to add any remaining missing edges
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    existing = set((min(u, v), max(u, v)) for u, v, _ in best_edges)
    missing = [e for e in all_edges if e not in existing]
    if missing and _time_left() > 3.0:
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in best_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        rng.shuffle(missing)
        for (u, v) in missing:
            if _time_left() < 3.0:
                break
            forbidden = set()
            possible = True
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        if ca == cb or cb == cv or ca == cv:
                            possible = False
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if not possible:
                    break
            if possible:
                available = [c for c in range(20) if c not in forbidden]
                if available:
                    c = available[0]
                    best_edges.append([u, v, c])
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)

    return best_edges
