import time
import random

def _is_3_colorable(adj, n=16):
    """
    Fast 3-colorability check using bitmask DP with memoization.
    adj: list of n sets of neighbors.
    Returns True if graph is 3-colorable, False otherwise.
    """
    # Precompute neighbor masks
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask

    # Order vertices by descending degree for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    # For each vertex in order, compute mask of neighbors that appear earlier in order
    earlier_mask = [0] * n
    for v in range(n):
        mask = 0
        for w in adj[v]:
            if pos[w] < pos[v]:
                mask |= (1 << w)
        earlier_mask[v] = mask

    # color[v] = -1 means uncolored, else 0,1,2
    color = [-1] * n

    # Use iterative deepening DFS with forward checking
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        # Determine forbidden colors based on earlier neighbors
        forbidden = 0
        em = earlier_mask[v]
        m = em
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            c = color[w]
            if c != -1:
                forbidden |= (1 << c)
            m ^= lsb

        # Try colors in order of least constraining value
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(i + 1):
                    return True
        color[v] = -1
        return False

    return backtrack(0)


def _build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_triangle_free(adj, n=16):
    for u in range(n):
        for v in adj[u]:
            if v > u:
                # check common neighbor
                au = adj[u]
                av = adj[v]
                for w in au:
                    if w != v and w in av:
                        return False
    return True


def _can_add_edge(u, v, adj):
    """Check if adding edge (u,v) keeps graph triangle-free."""
    if u == v or v in adj[u]:
        return False
    au = adj[u]
    av = adj[v]
    # triangle iff common neighbor exists
    # iterate over smaller set
    if len(au) < len(av):
        for w in au:
            if w in av:
                return False
    else:
        for w in av:
            if w in au:
                return False
    return True


# ---------- Grötzsch graph (Mycielski of C5) on vertices 0..10 ----------
def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges


def _initial_4chromatic_graph():
    """Return a 4-chromatic triangle-free graph on 11 vertices (Grötzsch)."""
    return _groetzsch_edges()


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin (limit 150s)
    n = 16

    # Start from Grötzsch graph (4-chromatic, triangle-free) on 11 vertices, 5 isolated
    base_edges = _groetzsch_edges()
    best_edges = [list(e) for e in base_edges]
    best_adj = _build_adj(best_edges, n)
    assert _is_triangle_free(best_adj, n)
    assert not _is_3_colorable(best_adj, n)

    best_count = len(best_edges)

    def elapsed_ok():
        return time.time() < deadline

    def to_set(edges):
        return set(frozenset((u, v)) for u, v in edges)

    def try_add(u, v, adj, edges):
        """Return (new_adj, new_edges) if adding (u,v) keeps triangle-free and 4-chromatic."""
        if not _can_add_edge(u, v, adj):
            return None
        new_adj = [s.copy() for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if _is_3_colorable(new_adj, n):
            return None
        new_edges = edges + [[u, v]]
        return new_adj, new_edges

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Phase 1: Greedy fill
    es = to_set(best_edges)
    changed = True
    while changed and elapsed_ok():
        changed = False
        for u, v in all_pairs:
            if not elapsed_ok():
                break
            if v in best_adj[u]:
                continue
            if frozenset((u, v)) in es:
                continue
            res = try_add(u, v, best_adj, best_edges)
            if res is not None:
                best_adj, best_edges = res
                es = to_set(best_edges)
                best_count = len(best_edges)
                changed = True

    # Phase 2: Intensive local search with multiple move types
    global_best_edges = [list(e) for e in best_edges]
    global_best_count = best_count

    def local_search(cur_adj, cur_edges, cur_es):
        """Run local search until no improvement or time runs out."""
        improved = True
        while improved and elapsed_ok():
            improved = False
            # Move 1: add single edge
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if frozenset((u, v)) in cur_es:
                    continue
                if v in cur_adj[u]:
                    continue
                res = try_add(u, v, cur_adj, cur_edges)
                if res is not None:
                    cur_adj, cur_edges = res
                    cur_es = to_set(cur_edges)
                    improved = True
                    break
            if improved or not elapsed_ok():
                continue

            # Move 2: remove 1, add 2 (net +1)
            for i, e0 in enumerate(list(cur_edges)):
                if not elapsed_ok():
                    break
                u0, v0 = e0
                cand_adj = [s.copy() for s in cur_adj]
                cand_adj[u0].discard(v0)
                cand_adj[v0].discard(u0)
                cand_edges = cur_edges[:i] + cur_edges[i + 1:]
                cand_es = to_set(cand_edges)
                added = 0
                for u, v in all_pairs:
                    if not elapsed_ok():
                        break
                    if v in cand_adj[u]:
                        continue
                    if frozenset((u, v)) in cand_es:
                        continue
                    if not _can_add_edge(u, v, cand_adj):
                        continue
                    tmp_adj = [s.copy() for s in cand_adj]
                    tmp_adj[u].add(v)
                    tmp_adj[v].add(u)
                    if _is_3_colorable(tmp_adj, n):
                        continue
                    cand_adj = tmp_adj
                    cand_edges = cand_edges + [[u, v]]
                    cand_es = to_set(cand_edges)
                    added += 1
                    if added == 2:
                        break
                if added == 2:
                    cur_adj = cand_adj
                    cur_edges = cand_edges
                    cur_es = cand_es
                    improved = True
                    break

            # Move 3: remove 2, add 3 (net +1) - more aggressive
            if not improved:
                for i, e0 in enumerate(list(cur_edges)):
                    if not elapsed_ok():
                        break
                    for j in range(i + 1, len(cur_edges)):
                        if not elapsed_ok():
                            break
                        e1 = cur_edges[j]
                        cand_adj = [s.copy() for s in cur_adj]
                        u0, v0 = e0
                        cand_adj[u0].discard(v0)
                        cand_adj[v0].discard(u0)
                        u1, v1 = e1
                        cand_adj[u1].discard(v1)
                        cand_adj[v1].discard(u1)
                        # Remove edges i and j (j may shift if i < j)
                        cand_edges = cur_edges[:i] + cur_edges[i+1:j] + cur_edges[j+1:]
                        cand_es = to_set(cand_edges)
                        added = 0
                        for u, v in all_pairs:
                            if not elapsed_ok():
                                break
                            if v in cand_adj[u]:
                                continue
                            if frozenset((u, v)) in cand_es:
                                continue
                            if not _can_add_edge(u, v, cand_adj):
                                continue
                            tmp_adj = [s.copy() for s in cand_adj]
                            tmp_adj[u].add(v)
                            tmp_adj[v].add(u)
                            if _is_3_colorable(tmp_adj, n):
                                continue
                            cand_adj = tmp_adj
                            cand_edges = cand_edges + [[u, v]]
                            cand_es = to_set(cand_edges)
                            added += 1
                            if added == 3:
                                break
                        if added == 3:
                            cur_adj = cand_adj
                            cur_edges = cand_edges
                            cur_es = cand_es
                            improved = True
                            break
                    if improved:
                        break

        return cur_adj, cur_edges, cur_es

    best_adj, best_edges, es = local_search(best_adj, best_edges, es)
    best_count = len(best_edges)
    if best_count > global_best_count:
        global_best_count = best_count
        global_best_edges = [list(e) for e in best_edges]

    # Phase 3: Multiple restarts with random perturbation
    restarts = 0
    while elapsed_ok():
        restarts += 1

        # Different perturbation strategies based on restart number
        if restarts % 3 == 1:
            # Mild: remove 2-3 edges
            perturb_edges = [list(e) for e in global_best_edges]
            num_remove = random.randint(2, 3)
        elif restarts % 3 == 2:
            # Medium: remove 4-6 edges
            perturb_edges = [list(e) for e in global_best_edges]
            num_remove = random.randint(4, 6)
        else:
            # Aggressive: start fresh from Grötzsch + random edges
            perturb_edges = [list(e) for e in _groetzsch_edges()]
            # Add some random edges that maintain triangle-free
            perturb_adj = _build_adj(perturb_edges, n)
            random.shuffle(all_pairs)
            for u, v in all_pairs[:random.randint(10, 20)]:
                if v not in perturb_adj[u] and _can_add_edge(u, v, perturb_adj):
                    perturb_adj[u].add(v)
                    perturb_adj[v].add(u)
                    perturb_edges.append([u, v])

        perturb_adj = _build_adj(perturb_edges, n)
        if not _is_triangle_free(perturb_adj, n):
            continue
        if _is_3_colorable(perturb_adj, n):
            continue

        cur_adj = perturb_adj
        cur_edges = perturb_edges
        cur_es = to_set(cur_edges)

        # Greedy fill from perturbed state
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in cur_adj[u]:
                    continue
                if frozenset((u, v)) in cur_es:
                    continue
                res = try_add(u, v, cur_adj, cur_edges)
                if res is not None:
                    cur_adj, cur_edges = res
                    cur_es = to_set(cur_edges)
                    changed = True

        # Local search
        cur_adj, cur_edges, cur_es = local_search(cur_adj, cur_edges, cur_es)

        if len(cur_edges) > global_best_count:
            global_best_count = len(cur_edges)
            global_best_edges = [list(e) for e in cur_edges]

        # Early termination if we've hit a high quality threshold
        if global_best_count >= 32:
            break

    # Final validation
    final_adj = _build_adj(global_best_edges, n)
    assert _is_triangle_free(final_adj, n)
    assert not _is_3_colorable(final_adj, n)

    return global_best_edges
