"""Rewritten program: Min-SAT style local search on K4-free graphs to minimize max triangle-free induced subgraph."""
import time
import random

def _edges_to_adj(edge_set, n=18):
    adj = [0] * n
    for (u, v) in edge_set:
        if u == v:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_adj(adj, n=18):
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v]
            cm = common
            while cm:
                lsb = cm & (-cm)
                w = lsb.bit_length() - 1
                if adj[w] & (common & ~(1 << w)):
                    return False
                cm ^= lsb
    return True

def _is_k4_free(edge_set, n=18):
    adj = _edges_to_adj(edge_set, n)
    return _is_k4_free_adj(adj, n)

def _max_tf_size(edge_set, n=18, deadline=None):
    """Return the size of the largest triangle-free induced subgraph.
    Uses exact branch-and-bound with time guard."""
    adj = _edges_to_adj(edge_set, n)
    full = (1 << n) - 1

    # Precompute triangles
    triangles = []
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v]
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    triangles.append((1 << u) | (1 << v) | (1 << w))

    if not triangles:
        return n

    # Greedy lower bound on max TF size (equivalently upper bound on min vertex cover)
    remaining = full
    removed = 0
    # Greedy: repeatedly remove vertex in most remaining triangles
    while True:
        if deadline is not None and time.time() > deadline:
            break
        best_v = -1
        best_c = -1
        for v in range(n):
            if not (remaining >> v) & 1:
                continue
            c = 0
            for tri in triangles:
                if (tri >> v) & 1 and (tri & remaining) == tri:
                    c += 1
            if c > best_c:
                best_c = c
                best_v = v
        if best_c <= 0:
            break
        remaining &= ~(1 << best_v)
        removed += 1

    lb_tf = bin(remaining).count("1")  # lower bound on max TF size
    ub_cover = removed  # upper bound on min cover

    if ub_cover == 0:
        return n

    # Branch and bound for min vertex cover of triangles
    best_cover = ub_cover
    if deadline is None:
        deadline = time.time() + 2.0

    def _bnb(tri_list, cover_mask, cover_size):
        nonlocal best_cover
        if time.time() > deadline:
            return
        if not tri_list:
            if cover_size < best_cover:
                best_cover = cover_size
            return
        if cover_size >= best_cover:
            return
        # Pick triangle with fewest uncovered vertices
        tri = tri_list[0]
        for v in range(n):
            if (tri >> v) & 1 and not (cover_mask >> v) & 1:
                new_cover = cover_mask | (1 << v)
                new_tris = [t for t in tri_list if not (t & (1 << v))]
                _bnb(new_tris, new_cover, cover_size + 1)
                if time.time() > deadline:
                    return

    _bnb(triangles, 0, 0)
    return max(lb_tf, n - best_cover)

def _random_greedy_k4_free(n=18, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    adj = [0] * n
    edge_set = set()
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    for (u, v) in all_edges:
        common = adj[u] & adj[v]
        ok = True
        cm = common
        while cm:
            lsb = cm & (-cm)
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~(1 << w)):
                ok = False
                break
            cm ^= lsb
        if ok:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
    return list(edge_set)

def _local_search(edges, n=18, max_iters=300, seed=None, deadline=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_score = _max_tf_size(edge_set, n, deadline)
    best_edges = list(edge_set)
    best_score = current_score
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(edge_set) == 0:
            break
        rem_edge = rng.choice(list(edge_set))
        cand = rng.choice(all_edges)
        if cand == rem_edge:
            continue
        new_set = edge_set - {rem_edge} | {cand}
        new_adj = _edges_to_adj(new_set, n)
        if not _is_k4_free_adj(new_adj, n):
            continue
        new_score = _max_tf_size(new_set, n, deadline)
        if new_score < current_score:
            current_score = new_score
            edge_set = new_set
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edge_set)
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 18

    # Base: complete tripartite K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    base_edges = [[a, b] for i in range(3) for j in range(i + 1, 3)
                  for a in parts[i] for b in parts[j]]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    best_edges = base_edges
    best_score = _max_tf_size(base_set, n, deadline)

    # Local search on base
    ls_deadline = min(deadline, start_time + 142)
    ls_edges, ls_score = _local_search(base_edges, n, max_iters=200, seed=999, deadline=ls_deadline)
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges

    # Random restarts with local search
    for trial in range(100):
        if time.time() > deadline:
            break
        edges = _random_greedy_k4_free(n, seed=trial + 100)
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not _is_k4_free(edge_set, n):
            continue
        score = _max_tf_size(edge_set, n, deadline)
        if score < best_score:
            best_score = score
            best_edges = edges
        ls_edges, ls_score = _local_search(edges, n, max_iters=100, seed=trial + 200, deadline=ls_deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges

    return [[u, v] for (u, v) in best_edges]
