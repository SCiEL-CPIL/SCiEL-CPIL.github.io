"""Triangle-free 4-chromatic graphs on 16 vertices: Grötzsch base + aggressive local search.

Strategy: Start with the Grötzsch graph (Mycielski(C5), 11 vertices, 20 edges, 4-chromatic,
triangle-free) and aggressively add edges involving the 5 extra vertices (11-15) while
maintaining triangle-freeness and 4-chromaticity. Use edge swaps to escape local optima.
"""
# EVOLVE-BLOCK-START
import time
import random

N = 16


def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices (0..10), 20 edges.
    Vertices 0..4 form C5.
    Vertices 5..9 are copies of 0..4.
    Vertex 10 is the center connected to 5..9.
    """
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])

    # Copies 5-9 correspond to 0-4.
    # Edges between original and copies:
    orig_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in orig_edges:
        edges.append([v, 5 + u])
        edges.append([u, 5 + v])

    # Center 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5 + i])

    return edges


def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj


def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2


def _is_triangle_free(adj):
    """Check if graph is triangle-free using bitmasks."""
    for u in range(N):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True


def _is_3_colorable(adj):
    """
    Fast backtracking to check if graph is 3-colorable.
    Uses bitmasks for speed.
    """
    colors = [-1] * N
    # Order vertices by degree descending for efficiency
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]

        # Find which colors are forbidden by colored neighbors
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:  # Already colored
                forbidden |= (1 << colors[u])

        # Try colors 0, 1, 2 that are not forbidden
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)


def _is_not_3_colorable(adj):
    return not _is_3_colorable(adj)


def _is_valid(adj):
    return _is_triangle_free(adj) and _is_not_3_colorable(adj)


def _try_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps triangle-free. Returns True if safe."""
    if (adj[u] >> v) & 1:
        return False
    # Adding (u,v) creates triangle if u and v have a common neighbor
    if adj[u] & adj[v]:
        return False
    return True


def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _greedy_add_edges(adj, deadline, rng):
    """
    Greedily add edges in random order while maintaining triangle-freeness.
    Returns the modified adj and new edge count.
    """
    cnt = _edge_count(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Get all missing edges
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                _add_edge(adj, u, v)
                cnt += 1
                improved = True
    return adj, cnt


def _local_search(adj, deadline, rng, min_edges):
    """
    Aggressive local search: try to add edges, and if stuck, try edge swaps
    (remove one edge, then greedily add more).
    Returns (adj, cnt) of best found.
    """
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    if best_cnt >= min_edges and _is_valid(adj):
        pass  # already valid

    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)

    # First, greedily add all possible edges
    cur_adj, cur_cnt = _greedy_add_edges(cur_adj, deadline, rng)

    if cur_cnt > best_cnt and _is_valid(cur_adj):
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]

    # Now try edge swaps to escape local optima
    improved = True
    swap_rounds = 0
    while improved and time.time() < deadline - 5.0 and swap_rounds < 50:
        improved = False
        swap_rounds += 1

        existing = [(u, v) for u in range(N) for v in range(u + 1, N)
                    if (cur_adj[u] >> v) & 1]
        if not existing:
            break

        # Try multiple random edge removals
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break

            rem_u, rem_v = rng.choice(existing)
            adj_copy = [a for a in cur_adj]
            _remove_edge(adj_copy, rem_u, rem_v)

            # Greedily add edges to the modified graph
            adj_copy, cnt_copy = _greedy_add_edges(adj_copy, deadline, rng)

            if cnt_copy > cur_cnt and _is_valid(adj_copy):
                cur_adj = adj_copy
                cur_cnt = cnt_copy
                improved = True

                if cur_cnt > best_cnt:
                    best_cnt = cur_cnt
                    best_adj = [a for a in cur_adj]

        # Also try pure add attempts
        missing = [(u, v) for u in range(N) for v in range(u + 1, N)
                   if not ((cur_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 5.0:
                break
            if _try_add_edge(cur_adj, u, v):
                _add_edge(cur_adj, u, v)
                cur_cnt += 1
                improved = True
                if cur_cnt > best_cnt:
                    best_cnt = cur_cnt
                    best_adj = [a for a in cur_adj]
                break

    return best_adj, best_cnt


def _perturb_grotzsch(base_adj, rng, num_remove=0):
    """
    Create a perturbed version of the Grötzsch graph by removing some edges.
    This allows exploration of different starting points.
    """
    adj = [a for a in base_adj]
    if num_remove > 0:
        # Get all edges in the base
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        # Remove random edges
        for _ in range(min(num_remove, len(edges))):
            if edges:
                idx = rng.randint(0, len(edges) - 1)
                u, v = edges[idx]
                _remove_edge(adj, u, v)
                edges.pop(idx)
    return adj


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin vs 150s kill

    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    base_cnt = _edge_count(base_adj)  # should be 20

    # Verify base is valid
    if not _is_valid(base_adj):
        # Fallback: return the base edges as-is (shouldn't happen)
        out = []
        for u in range(N):
            m = base_adj[u]
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                if v > u:
                    out.append([u, v])
        return out

    best_adj = [a for a in base_adj]
    best_cnt = base_cnt

    rng = random.Random(42)

    # Multiple trials with different strategies
    trial = 0
    while time.time() < deadline - 10.0:
        trial += 1

        # Vary the perturbation: sometimes start from base, sometimes perturb
        if trial % 3 == 0 and trial > 1:
            # Perturb: remove 1-3 random edges from base
            num_remove = rng.randint(1, 3)
            adj = _perturb_grotzsch(base_adj, rng, num_remove)
        else:
            adj = [a for a in base_adj]

        # Run local search
        result_adj, result_cnt = _local_search(adj, deadline, rng, best_cnt)

        if result_cnt > best_cnt and _is_valid(result_adj):
            best_cnt = result_cnt
            best_adj = [a for a in result_adj]

        # Early stop if we've found a very good solution
        if best_cnt >= 35:
            # Try a few more iterations but don't spend too much time
            if trial > 20:
                break

    # Final verification
    if not _is_valid(best_adj):
        best_adj = [a for a in base_adj]
        best_cnt = base_cnt

    # Convert to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])

    return out
# EVOLVE-BLOCK-END
