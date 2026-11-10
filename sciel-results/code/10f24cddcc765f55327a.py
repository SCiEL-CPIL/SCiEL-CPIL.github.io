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
    """Exact branch & bound: find min vertex cover of triangle hypergraph.
    Returns _N - min_cover_size."""
    
    # Build list of triangles in current graph
    triangles = []
    for (u, v) in _edges:
        common = _adj[u] & _adj[v]
        m = common
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            triangles.append((1 << u) | (1 << v) | (1 << w))
            m = m & ~(1 << w)

    if not triangles:
        return _N

    tri_masks = triangles
    n_tris = len(tri_masks)
    
    # Precompute vertex -> bitmask of triangle indices containing that vertex
    v_tri_mask = [0] * _N
    for i, tm in enumerate(tri_masks):
        m = tm
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            v_tri_mask[v] |= (1 << i)
            m = m & ~(1 << v)

    best = [_N]
    
    # Greedy upper bound: repeatedly pick vertex in most uncovered triangles
    def _greedy_cover():
        cover = 0
        uncovered = (1 << n_tris) - 1
        while uncovered:
            # Find vertex covering most uncovered triangles
            best_v = -1
            best_cnt = 0
            for v in range(_N):
                cnt = bin(uncovered & v_tri_mask[v]).count('1')
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v < 0 or best_cnt == 0:
                break
            cover += 1
            uncovered &= ~v_tri_mask[best_v]
        return cover

    best[0] = min(best[0], _greedy_cover())
    
    # Lower bound: maximum number of disjoint triangles in free_mask
    def _disjoint_tri_lb(free_mask):
        cnt = 0
        used = 0
        for tm in tri_masks:
            if (free_mask & tm) == tm and not (used & tm):
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
        lb = _disjoint_tri_lb(free_mask)
        if cover_size + lb >= best[0]:
            return
        # Pick a triangle in free_mask to branch on - choose one with highest total degree
        branch_tm = None
        best_score = -1
        for tm in tri_masks:
            if (free_mask & tm) == tm:
                score = 0
                m = tm
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    score += bin(v_tri_mask[v] & ((1 << n_tris) - 1)).count('1')
                    m = m & ~(1 << v)
                if score > best_score:
                    best_score = score
                    branch_tm = tm
        if branch_tm is None:
            return
        # Order vertices by degree in triangles (more constrained first)
        verts = []
        m = branch_tm
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m = m & ~(1 << v)
        verts.sort(key=lambda x: -bin(v_tri_mask[x]).count('1'))
        for v in verts:
            new_cover = cover_mask | (1 << v)
            new_free = free_mask & ~(1 << v)
            _search(new_cover, cover_size + 1, new_free)

    _search(0, 0, (1 << _N) - 1)
    return _N - best[0]


_score_cache = {}

def _score():
    if not _is_k4_free():
        return 10**9
    # Use edge set as cache key
    key = tuple(sorted(_EDGE_SET))
    if key in _score_cache:
        return _score_cache[key]
    result = _max_triangle_free_induced_size()
    _score_cache[key] = result
    # Limit cache size
    if len(_score_cache) > 500:
        _score_cache.clear()
    return result


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

def _build_turan_like(rng, parts=3, part_sizes=None):
    """Build a Turan-like graph (complete multipartite) with some random perturbations."""
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    
    if part_sizes is None:
        base = _N // parts
        rem = _N % parts
        part_sizes = [base + (1 if i < rem else 0) for i in range(parts)]
    
    # Create parts
    parts_list = []
    idx = 0
    for ps in part_sizes:
        parts_list.append(list(range(idx, idx + ps)))
        idx += ps
    
    # Add all edges between different parts
    for i in range(parts):
        for j in range(i + 1, parts):
            for a in parts_list[i]:
                for b in parts_list[j]:
                    _add_edge(a, b)
    
    # Randomly remove some edges to create variation
    n_remove = rng.randint(0, 5)
    if _edges:
        to_remove = rng.sample(_edges, min(n_remove, len(_edges)))
        for (u, v) in to_remove:
            _remove_edge(u, v)
    
    # Try to add some intra-part edges if they don't create K4
    for _ in range(10):
        # Pick a random part
        part_idx = rng.randint(0, parts - 1)
        part = parts_list[part_idx]
        if len(part) < 2:
            continue
        u, v = rng.sample(part, 2)
        if (u, v) not in _EDGE_SET and (v, u) not in _EDGE_SET:
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


def _local_search_add_edges(current_edges, deadline, rng):
    """Try to add edges that improve the score."""
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    for (u, v) in current_edges:
        _add_edge(u, v)
    
    s = _score()
    best_s = s
    best_e = list(_edges)
    
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    for pass_num in range(100):
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
                    if s < best_s:
                        best_s = s
                        best_e = list(_edges)
                else:
                    _remove_edge(u, v)
        if not any_improved:
            break
    
    return best_e, best_s

def _local_search_remove_edges(current_edges, deadline, rng):
    """Try to remove edges that improve the score (make graph denser in some sense)."""
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    for (u, v) in current_edges:
        _add_edge(u, v)
    
    s = _score()
    best_s = s
    best_e = list(_edges)
    
    for pass_num in range(50):
        if time.time() > deadline:
            break
        if not _edges:
            break
        # Try removing each edge
        edges_copy = list(_edges)
        rng.shuffle(edges_copy)
        any_improved = False
        for (u, v) in edges_copy:
            if time.time() > deadline:
                break
            if (u, v) not in _EDGE_SET:
                continue
            _remove_edge(u, v)
            ns = _score()
            if ns < s:
                s = ns
                any_improved = True
                if s < best_s:
                    best_s = s
                    best_e = list(_edges)
            else:
                _add_edge(u, v)
        if not any_improved:
            break
    
    return best_e, best_s

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (and 150s kill)
    rng = random.Random(12345)

    best_edges = None
    best_score = 10**9

    # Try multiple initial constructions
    initial_strategies = [
        lambda: _build_turan_like(rng, parts=3),  # K6,6,6
        lambda: _build_turan_like(rng, parts=4, part_sizes=[5,5,4,4]),  # K5,5,4,4
        lambda: _build_turan_like(rng, parts=2, part_sizes=[9,9]),  # K9,9
        lambda: _random_greedy_k4_free(rng),
    ]
    
    for strategy in initial_strategies:
        if time.time() > deadline:
            break
        strategy()
        s = _score()
        if s < best_score:
            best_score = s
            best_edges = list(_edges)
    
    # Main search loop
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # Alternate between different search strategies
        if iters % 4 == 1:
            # Random restart + local search add
            _random_greedy_k4_free(rng)
            s = _score()
            if s < best_score:
                best_score = s
                best_edges = list(_edges)
            # Local search: add edges
            if time.time() < deadline - 5:
                ls_edges, ls_score = _local_search_add_edges(list(_edges), min(deadline - 5, time.time() + 15), rng)
                if ls_score < best_score:
                    best_score = ls_score
                    best_edges = ls_edges
        elif iters % 4 == 2:
            # Perturb best + local search remove
            if best_edges is not None:
                _adj[:] = [0] * _N
                _edges.clear()
                _EDGE_SET.clear()
                for (u, v) in best_edges:
                    _add_edge(u, v)
                # Randomly remove 2-5 edges
                if _edges:
                    n_remove = min(len(_edges), rng.randint(2, 5))
                    to_remove = rng.sample(_edges, n_remove)
                    for (u, v) in to_remove:
                        _remove_edge(u, v)
                s = _score()
                if s < best_score:
                    best_score = s
                    best_edges = list(_edges)
                # Local search: remove edges
                if time.time() < deadline - 5:
                    ls_edges, ls_score = _local_search_remove_edges(list(_edges), min(deadline - 5, time.time() + 15), rng)
                    if ls_score < best_score:
                        best_score = ls_score
                        best_edges = ls_edges
        elif iters % 4 == 3:
            # Turan-like with different parameters
            parts = rng.choice([3, 4, 5])
            base = _N // parts
            rem = _N % parts
            ps = [base + (1 if i < rem else 0) for i in range(parts)]
            rng.shuffle(ps)
            _build_turan_like(rng, parts=parts, part_sizes=ps)
            s = _score()
            if s < best_score:
                best_score = s
                best_edges = list(_edges)
            # Local search add
            if time.time() < deadline - 5:
                ls_edges, ls_score = _local_search_add_edges(list(_edges), min(deadline - 5, time.time() + 15), rng)
                if ls_score < best_score:
                    best_score = ls_score
                    best_edges = ls_edges
        else:
            # Simulated annealing polish on best
            if best_edges is not None and time.time() < deadline - 10:
                sa_deadline = min(deadline - 5, time.time() + 20)
                sa_edges, sa_score = _simulated_annealing(best_edges, sa_deadline, rng)
                if sa_score < best_score:
                    best_score = sa_score
                    best_edges = sa_edges
                # Follow up with local search
                if time.time() < deadline - 5:
                    ls_edges, ls_score = _local_search_add_edges(best_edges, min(deadline - 5, time.time() + 10), rng)
                    if ls_score < best_score:
                        best_score = ls_score
                        best_edges = ls_edges

    if best_edges is None:
        # fallback to K6,6,6
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in range(i * 6, i * 6 + 6) for b in range(j * 6, j * 6 + 6)]

    return [[u, v] for (u, v) in best_edges]
