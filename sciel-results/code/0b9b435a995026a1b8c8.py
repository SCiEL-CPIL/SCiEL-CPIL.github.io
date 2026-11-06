"""Construct dense 4-cycle-rainbow edge-coloured graph on 20 vertices.

Combines:
  * fast greedy builder with bitmask forbidden-colour computation
  * bipartite seeding to avoid C4-free local optima
  * local search with re-color-to-unlock, remove+re-greedy, and color-class swap moves
  * fast incremental validity check
  * NumPy-based fast validation
"""
import time
import random
from itertools import combinations
import numpy as np

N = 20
N_COLORS = 20
DEADLINE = 145.0

ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]
ALL_PAIRS_SET = set(ALL_PAIRS)


def _validate(edges):
    """Check that every 4-cycle has four distinct colours."""
    if not edges:
        return True
    adj = np.full((N, N), -1, dtype=np.int8)
    for u, v, c in edges:
        adj[u, v] = c
        adj[v, u] = c

    for a in range(N):
        for b in range(a + 1, N):
            if adj[a, b] == -1:
                continue
            for c in range(b + 1, N):
                if adj[a, c] == -1 or adj[b, c] == -1:
                    continue
                for d in range(c + 1, N):
                    # Check cycle a-b-c-d-a
                    if adj[b, c] != -1 and adj[c, d] != -1 and adj[d, a] != -1:
                        cols = (adj[a, b], adj[b, c], adj[c, d], adj[d, a])
                        if len(set(cols)) != 4:
                            return False
                    # Check cycle a-b-d-c-a
                    if adj[a, d] != -1 and adj[b, d] != -1 and adj[c, d] != -1:
                        cols = (adj[a, b], adj[b, d], adj[d, c], adj[c, a])
                        if len(set(cols)) != 4:
                            return False
                    # Check cycle a-c-b-d-a
                    if adj[a, c] != -1 and adj[c, b] != -1 and adj[b, d] != -1 and adj[d, a] != -1:
                        cols = (adj[a, c], adj[c, b], adj[b, d], adj[d, a])
                        if len(set(cols)) != 4:
                            return False
    return True


def _fast_check_after_change(adj, neighbors, changed_edge):
    """Check only the 4-cycles that involve changed_edge=(u,v)."""
    u, v = changed_edge
    for x in list(neighbors[u]):
        if x == v:
            continue
        cx = adj[u][x]
        for y in list(neighbors[v]):
            if y == u or y == x:
                continue
            if adj[x][y] == -1:
                continue
            cols = (cx, adj[x][y], adj[y][v], adj[v][u])
            if len(set(cols)) != 4:
                return False
    return True


def _try_add_edge(adj, neighbors, u, v):
    """Try to add edge (u,v) with a valid color. Returns color or -1."""
    forbidden_mask = 0
    for a in list(neighbors[u]):
        if a == v:
            continue
        ca = adj[u][a]
        for b in list(neighbors[v]):
            if b == u or b == a:
                continue
            cb = adj[a][b]
            if cb == -1:
                continue
            cv = adj[b][v]
            if ca == cb or cb == cv or ca == cv:
                return -1  # impossible
            forbidden_mask |= (1 << ca) | (1 << cb) | (1 << cv)
    
    if forbidden_mask == 0:
        return random.randrange(N_COLORS)
    
    # Find available color
    for c in range(N_COLORS):
        if not (forbidden_mask & (1 << c)):
            return c
    return -1


def _try_add_edge_deterministic(adj, neighbors, u, v):
    """Try to add edge (u,v) with a valid color (deterministic). Returns color or -1."""
    forbidden_mask = 0
    for a in list(neighbors[u]):
        if a == v:
            continue
        ca = adj[u][a]
        for b in list(neighbors[v]):
            if b == u or b == a:
                continue
            cb = adj[a][b]
            if cb == -1:
                continue
            cv = adj[b][v]
            if ca == cb or cb == cv or ca == cv:
                return -1  # impossible
            forbidden_mask |= (1 << ca) | (1 << cb) | (1 << cv)
    
    if forbidden_mask == 0:
        return 0  # any color works, use 0 for determinism
    
    for c in range(N_COLORS):
        if not (forbidden_mask & (1 << c)):
            return c
    return -1


def construct():
    start = time.time()

    def elapsed():
        return time.time() - start

    def deadline_ok():
        return elapsed() < DEADLINE

    def greedy_build(seed, initial_edges=None, order_type='random'):
        rng = random.Random(seed)

        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        edges = []

        if initial_edges:
            for u, v, c in initial_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing_set = set((min(u, v), max(u, v)) for u, v, _ in edges)
            remaining = [e for e in ALL_PAIRS if e not in existing_set]
        else:
            remaining = ALL_PAIRS[:]

        if order_type == 'random':
            rng.shuffle(remaining)
        elif order_type == 'reverse':
            remaining.reverse()

        for (u, v) in remaining:
            if v in neighbors[u]:
                continue

            # Compute forbidden colors using bitmask
            forbidden_mask = 0
            impossible = False
            for a in list(neighbors[u]):
                if a == v:
                    continue
                ca = adj[u][a]
                for b in list(neighbors[v]):
                    if b == u or b == a:
                        continue
                    cb = adj[a][b]
                    if cb == -1:
                        continue
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden_mask |= (1 << ca) | (1 << cb) | (1 << cv)
                if impossible:
                    break

            if not impossible:
                if forbidden_mask == 0:
                    c = rng.randrange(N_COLORS)
                else:
                    # Find available color
                    c = -1
                    for cc in range(N_COLORS):
                        if not (forbidden_mask & (1 << cc)):
                            c = cc
                            break
                    if c == -1:
                        continue
                    c = rng.choice([cc for cc in range(N_COLORS) if not (forbidden_mask & (1 << cc))])
                
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])

        return edges

    def bipartite_seed(seed, density=0.3):
        """Create a bipartite graph seed to avoid C4-free local optima."""
        rng = random.Random(seed)
        group_a = list(range(10))
        group_b = list(range(10, 20))
        edges = []
        for a in group_a:
            for b in group_b:
                if rng.random() < density:
                    c = rng.randrange(N_COLORS)
                    edges.append([a, b, c])
        return edges

    def recolor_to_unlock(edges, seed, time_limit):
        """Try re-coloring edges to unlock new edge additions."""
        rng = random.Random(seed)
        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        current_edges = [e[:] for e in edges]
        current_count = len(current_edges)
        
        # Build list of missing edges
        existing_set = set((min(u, v), max(u, v)) for u, v, _ in current_edges)
        missing_edges = [e for e in ALL_PAIRS if e not in existing_set]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try re-coloring each edge
            for idx in range(len(current_edges)):
                if time.time() > time_limit:
                    break
                
                u, v, old_c = current_edges[idx]
                
                # Try each new color
                for new_c in range(N_COLORS):
                    if new_c == old_c:
                        continue
                    
                    # Check if re-coloring is valid
                    valid = True
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
                                if ca == new_c or new_c == cv or cv == cb or ca == cb:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Apply re-color
                    adj[u][v] = new_c
                    adj[v][u] = new_c
                    current_edges[idx][2] = new_c
                    
                    # Try to add new edges
                    added = False
                    for (wu, wv) in missing_edges:
                        if adj[wu][wv] != -1:
                            continue
                        nc = _try_add_edge(adj, neighbors, wu, wv)
                        if nc != -1:
                            adj[wu][wv] = nc
                            adj[wv][wu] = nc
                            neighbors[wu].add(wv)
                            neighbors[wv].add(wu)
                            current_edges.append([wu, wv, nc])
                            missing_edges.remove((wu, wv))
                            added = True
                    
                    if added:
                        improved = True
                        break
                    else:
                        # Revert re-color
                        adj[u][v] = old_c
                        adj[v][u] = old_c
                        current_edges[idx][2] = old_c
        
        return current_edges

    best_edges = None
    best_count = 0

    # ---- Phase 1: multiple greedy restarts ----
    # First try bipartite seeds
    for i in range(30):
        if not deadline_ok():
            break
        seed_edges = bipartite_seed(i * 31 + 7, density=0.25 + 0.01 * (i % 5))
        edges = greedy_build(i * 100 + 1, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Then try standard greedy restarts
    order_types = ['random', 'natural', 'reverse']
    for ot in order_types:
        for i in range(40):
            if not deadline_ok():
                break
            edges = greedy_build(i * 7 + 13, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    if best_edges is None:
        best_edges = [[0, v, v % N_COLORS] for v in range(1, N)]
        best_count = len(best_edges)

    # ---- Phase 2: local search with recolor-to-unlock ----
    current_best = best_edges[:]
    refine_iter = 0
    max_refine = 12000

    while deadline_ok() and refine_iter < max_refine:
        refine_iter += 1

        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        for u, v, c in current_best:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)

        move = random.random()
        candidate = None

        if move < 0.45 and len(current_best) > 0:
            # ---- Re-color to unlock move ----
            time_limit = time.time() + min(2.0, (DEADLINE - elapsed()) * 0.05)
            new_edges = recolor_to_unlock(current_best, refine_iter * 1000 + random.randint(0, 1000), time_limit)
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges
                current_best = new_edges[:]
                candidate = new_edges
        elif move < 0.60:
            # ---- Color-class swap move ----
            c1, c2 = random.sample(range(N_COLORS), 2)
            cand = []
            for e in current_best:
                if e[2] == c1:
                    cand.append([e[0], e[1], c2])
                elif e[2] == c2:
                    cand.append([e[0], e[1], c1])
                else:
                    cand.append([e[0], e[1], e[2]])
            seed = refine_iter * 1000 + random.randint(0, 1000)
            new_edges = greedy_build(seed, cand, 'random')
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges
                current_best = new_edges[:]
                candidate = new_edges
        elif move < 0.75:
            # ---- Bipartite seed + re-greedy ----
            seed_edges = bipartite_seed(refine_iter * 17 + 3, density=0.2 + 0.05 * random.random())
            if len(current_best) > 10:
                keep = random.sample(current_best, min(15, len(current_best)))
                keep_set = set((min(u, v), max(u, v)) for u, v, _ in keep)
                combined = keep + [e for e in seed_edges if (min(e[0], e[1]), max(e[0], e[1])) not in keep_set]
            else:
                combined = seed_edges
            seed = refine_iter * 1000 + random.randint(0, 1000)
            new_edges = greedy_build(seed, combined, 'random')
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges
                current_best = new_edges[:]
                candidate = new_edges
        else:
            # ---- Remove + re-greedy move ----
            k = random.randint(1, 6)
            if len(current_best) > k:
                edges_to_remove = random.sample(current_best, k)
                remove_set = set((min(u, v), max(u, v)) for u, v, _ in edges_to_remove)
                base_edges = [e for e in current_best
                              if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
                seed = refine_iter * 1000 + random.randint(0, 1000)
                new_edges = greedy_build(seed, base_edges, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current_best = new_edges[:]
                    candidate = new_edges

        # Periodic validation
        if refine_iter % 500 == 0 and best_edges is not None:
            if not _validate(best_edges):
                current_best = [[0, v, v % N_COLORS] for v in range(1, N)]
                best_count = len(current_best)
                best_edges = current_best[:]

    # Final validation
    if best_edges is not None and not _validate(best_edges):
        return [[0, v, v % N_COLORS] for v in range(1, N)]

    return best_edges if best_edges is not None else [[0, v, v % N_COLORS] for v in range(1, N)]


if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
    print(f"Valid: {_validate(result)}")
