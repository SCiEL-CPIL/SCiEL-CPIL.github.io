import random
import time

_N = 18
# adjacency bitmask per vertex
_adj = [0] * _N
_edges = []          # list of (u, v) with u < v
_EDGE_SET = set()


def _add_edge(u, v):
    if u == v:
        return
    if u > v:
        u, v = v, u
    if (u, v) in _EDGE_SET:
        return
    _adj[u] |= 1 << v
    _adj[v] |= 1 << u
    _EDGE_SET.add((u, v))
    _edges.append((u, v))


def _remove_edge(u, v):
    if u > v:
        u, v = v, u
    if (u, v) not in _EDGE_SET:
        return
    _adj[u] &= ~(1 << v)
    _adj[v] &= ~(1 << u)
    _EDGE_SET.remove((u, v))
    _edges.remove((u, v))


def _is_k4_free():
    # A K4 exists iff there is an edge (u,v) whose common-neighborhood contains an edge.
    for (u, v) in _edges:
        common = _adj[u] & _adj[v]
        # check if any two vertices in common are adjacent
        m = common
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            rest = m & ~(1 << a)
            if rest & _adj[a]:
                return False
            m = rest
    return True


def _max_triangle_free_induced_size():
    # Exact branch & bound: max size of subset S with no triangle in the induced subgraph.
    # We want to MINIMIZE |R| such that every triangle has at least one vertex in R.
    # Then answer = _N - min_vertex_cover_size.

    full = (1 << _N) - 1

    # Build list of triangles in current graph.
    triangles = []
    for (u, v) in _edges:
        common = _adj[u] & _adj[v]
        m = common
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            triangles.append((u, v, w))
            m = m & ~(1 << w)

    if not triangles:
        return _N

    # Convert to bitmasks for faster checking
    tri_masks = []
    for (u, v, w) in triangles:
        tri_masks.append((1 << u) | (1 << v) | (1 << w))

    # Precompute vertex -> list of triangle indices
    v_tri_idx = [[] for _ in range(_N)]
    for i, tm in enumerate(tri_masks):
        m = tm
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            v_tri_idx[v].append(i)
            m = m & ~(1 << v)

    best = [_N]  # best cover size found (we minimize); start with all vertices

    # Greedy upper bound
    def _greedy_cover():
        cover = 0
        remaining_tris = set(range(len(tri_masks)))
        while remaining_tris:
            # Pick triangle with fewest remaining vertices? Or pick vertex hitting most tris.
            # Let's pick the vertex that appears in the most remaining triangles.
            counts = [0] * _N
            for ti in remaining_tris:
                m = tri_masks[ti]
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    counts[v] += 1
                    m = m & ~(1 << v)
            if max(counts) == 0:
                break
            best_v = max(range(_N), key=lambda x: counts[x])
            cover += 1
            # Remove all triangles containing best_v
            new_remaining = set()
            for ti in remaining_tris:
                if not (tri_masks[ti] & (1 << best_v)):
                    new_remaining.add(ti)
            remaining_tris = new_remaining
        return cover

    best[0] = min(best[0], _greedy_cover())

    # Lower bound: disjoint triangles in free_mask
    def _disjoint_tri_count(free_mask):
        cnt = 0
        used = 0
        for tm in tri_masks:
            if (free_mask & tm) == tm:
                if not (used & tm):
                    cnt += 1
                    used |= tm
                    if cnt >= best[0]:
                        return cnt
        return cnt

    def _search(cover_mask, cover_size, free_mask):
        if cover_size >= best[0]:
            return
        # Check if free_mask has any triangle
        has_tri = False
        for tm in tri_masks:
            if (free_mask & tm) == tm:
                has_tri = True
                break
        if not has_tri:
            if cover_size < best[0]:
                best[0] = cover_size
            return
        # Lower bound
        lb = _disjoint_tri_count(free_mask)
        if cover_size + lb >= best[0]:
            return
        # Pick a triangle in free_mask to branch on
        # Heuristic: pick triangle whose vertices have highest combined degree in remaining triangles
        branch_tm = None
        best_score_tri = -1
        for tm in tri_masks:
            if (free_mask & tm) == tm:
                # Score: sum of degrees of its vertices in triangles that are fully in free
                score = 0
                m = tm
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    for ti in v_tri_idx[v]:
                        if (free_mask & tri_masks[ti]) == tri_masks[ti]:
                            score += 1
                    m = m & ~(1 << v)
                if score > best_score_tri:
                    best_score_tri = score
                    branch_tm = tm
        if branch_tm is None:
            return
        # Order vertices by degree in free triangles (more constrained first)
        verts = []
        m = branch_tm
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m = m & ~(1 << v)
        verts.sort(key=lambda x: -len([ti for ti in v_tri_idx[x] if (free_mask & tri_masks[ti]) == tri_masks[ti]]))
        for v in verts:
            new_cover = cover_mask | (1 << v)
            new_free = free_mask & ~(1 << v)
            _search(new_cover, cover_size + 1, new_free)

    _search(0, 0, (1 << _N) - 1)
    return _N - best[0]


def _score():
    if not _is_k4_free():
        return 10**9
    return _max_triangle_free_induced_size()


def _random_greedy_k4_free(rng):
    # Build a random K4-free graph by adding edges in random order, skipping if it creates a K4.
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    rng.shuffle(all_pairs)
    for (u, v) in all_pairs:
        _add_edge(u, v)
        if not _is_k4_free():
            _remove_edge(u, v)
    return


def _try_add_edge(u, v):
    """Try to add edge (u,v). Returns True if added successfully (K4-free maintained)."""
    _add_edge(u, v)
    if _is_k4_free():
        return True
    _remove_edge(u, v)
    return False


def _try_remove_edge(u, v):
    """Remove edge (u,v) if it exists. Returns True if removed."""
    if (u, v) in _EDGE_SET or (v, u) in _EDGE_SET:
        _remove_edge(u, v)
        return True
    return False


def _simulated_annealing(start_edges, deadline, rng):
    """
    Perform simulated annealing on the edge set, accepting worse moves with probability
    exp(-delta / T). We minimize the score (largest triangle-free induced subgraph size).
    """
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    for (u, v) in start_edges:
        _add_edge(u, v)

    current_score = _score()
    best_score = current_score
    best_edges = list(_edges)

    T = 2.0  # initial temperature
    cooling = 0.995
    min_T = 0.05

    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]

    while time.time() < deadline and T > min_T:
        # Choose a random move: add an edge, remove an edge, or swap
        move_type = rng.random()
        if move_type < 0.4:
            # Add a random missing edge
            missing = [(u, v) for (u, v) in all_pairs if (u, v) not in _EDGE_SET]
            if not missing:
                break
            u, v = rng.choice(missing)
            _add_edge(u, v)
            if not _is_k4_free():
                _remove_edge(u, v)
                T *= cooling
                continue
            new_score = _score()
            delta = new_score - current_score
            if delta <= 0 or rng.random() < min(1.0, 1.0 / (1.0 + 0.1 * T)):  # accept with prob
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = list(_edges)
            else:
                _remove_edge(u, v)
        elif move_type < 0.8:
            # Remove a random edge
            if not _edges:
                break
            u, v = rng.choice(_edges)
            _remove_edge(u, v)
            new_score = _score()
            delta = new_score - current_score
            if delta <= 0 or rng.random() < min(1.0, 1.0 / (1.0 + 0.1 * T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = list(_edges)
            else:
                _add_edge(u, v)
        else:
            # Swap: remove one edge, add another
            if not _edges:
                break
            u1, v1 = rng.choice(_edges)
            missing = [(u, v) for (u, v) in all_pairs if (u, v) not in _EDGE_SET]
            if not missing:
                break
            u2, v2 = rng.choice(missing)
            _remove_edge(u1, v1)
            _add_edge(u2, v2)
            if not _is_k4_free():
                _remove_edge(u2, v2)
                _add_edge(u1, v1)
                T *= cooling
                continue
            new_score = _score()
            delta = new_score - current_score
            if delta <= 0 or rng.random() < min(1.0, 1.0 / (1.0 + 0.1 * T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = list(_edges)
            else:
                _remove_edge(u2, v2)
                _add_edge(u1, v1)
        T *= cooling

    return best_edges, best_score


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (and 150s kill)
    rng = random.Random(12345)

    best_edges = None
    best_score = 10**9

    # Initial: complete tripartite K6,6,6
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in range(i * 6, i * 6 + 6):
                for b in range(j * 6, j * 6 + 6):
                    _add_edge(a, b)
    s = _score()
    if s < best_score:
        best_score = s
        best_edges = list(_edges)

    # Multiple random restarts with local search and simulated annealing
    iters = 0
    while time.time() < deadline:
        iters += 1
        # random restart
        _random_greedy_k4_free(rng)
        s = _score()
        if s < best_score:
            best_score = s
            best_edges = list(_edges)

        # Also try starting from the best known graph and perturbing
        if iters % 3 == 0 and best_edges is not None and time.time() < deadline:
            # Load best edges, then randomly remove some and re-optimize
            _adj[:] = [0] * _N
            _edges.clear()
            _EDGE_SET.clear()
            for (u, v) in best_edges:
                _add_edge(u, v)
            # Randomly remove 3-8 edges to create a perturbation
            if _edges:
                n_remove = min(len(_edges), rng.randint(3, 8))
                to_remove = rng.sample(_edges, n_remove)
                for (u, v) in to_remove:
                    _remove_edge(u, v)
            s = _score()
            if s < best_score:
                best_score = s
                best_edges = list(_edges)

        # Local search: iteratively try to add missing edges that improve the score
        if time.time() < deadline:
            s = _score()
            all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
            # Multiple passes over all pairs
            for pass_num in range(50):
                if time.time() > deadline:
                    break
                rng.shuffle(all_pairs)
                any_improved = False
                for (u, v) in all_pairs:
                    if time.time() > deadline:
                        break
                    if (u, v) in _EDGE_SET:
                        continue
                    if _try_add_edge(u, v):
                        ns = _score()
                        if ns < s:
                            s = ns
                            any_improved = True
                            if s < best_score:
                                best_score = s
                                best_edges = list(_edges)
                        else:
                            # Edge was added but didn't improve; remove it to keep graph sparse
                            _remove_edge(u, v)
                if not any_improved:
                    break

        # Simulated annealing polish
        if time.time() < deadline - 10.0 and best_edges is not None:
            sa_deadline = min(deadline - 5.0, time.time() + 20.0)
            sa_edges, sa_score = _simulated_annealing(best_edges, sa_deadline, rng)
            if sa_score < best_score:
                best_score = sa_score
                best_edges = sa_edges

    if best_edges is None:
        # fallback to K6,6,6
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in range(i * 6, i * 6 + 6) for b in range(j * 6, j * 6 + 6)]

    return [[u, v] for (u, v) in best_edges]
