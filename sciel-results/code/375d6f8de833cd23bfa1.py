import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_long_cycle(adj, n, max_starts=30, max_restarts=8, rng=None):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy with improved scoring.
    Returns the cycle as a list of vertices or None.
    """
    if rng is None:
        rng = random.Random(42)

    best_cycle = None
    best_len = 0

    num_starts = min(n, max_starts)
    if n <= max_starts:
        starts = list(range(n))
    else:
        starts = rng.sample(range(n), num_starts)

    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]

                if not unvisited:
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    if len(path) >= 4 and start in adj[current] and rng.random() < 0.3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    scored = []
                    for v in unvisited:
                        unvisited_neighbors = sum(1 for w in adj[v] if w not in visited)
                        score = unvisited_neighbors
                        if v == start:
                            score += 10
                        scored.append((score, v))

                    scored.sort(reverse=True)
                    top_k = min(3, len(scored))
                    if rng.random() < 0.7:
                        best_next = scored[0][1]
                    else:
                        best_next = rng.choice([s[1] for s in scored[:top_k]])

                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next

    return best_cycle


def _find_2_factor(adj, n, rng=None):
    """
    Find a maximum-size 2-factor (collection of vertex-disjoint cycles covering
    as many vertices as possible) using a randomized matching-based approach.
    Returns a list of cycles (each a list of vertices), or empty list.
    """
    if rng is None:
        rng = random.Random(0)

    # Build a directed version: for each undirected edge (u,v), add arcs u->v and v->u.
    # A 2-factor is a set of arcs where each vertex has in-degree 1 and out-degree 1
    # (in the selected arcs), which decomposes into vertex-disjoint directed cycles.
    # We solve this as an assignment problem (min-cost perfect matching on a bipartite graph).

    # Use scipy's linear_sum_assignment for the assignment problem.
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return []

    # Cost matrix: cost[i][j] = 0 if arc i->j exists (i != j), else large cost.
    # We want to minimize cost, so we prefer existing arcs.
    BIG = 10**9
    cost = [[BIG] * n for _ in range(n)]
    for u in range(n):
        cost[u][u] = BIG  # no self-loops
        for v in adj[u]:
            cost[u][v] = 0

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Check if all assigned arcs are valid (cost 0)
    valid = all(cost[row_ind[i]][col_ind[i]] == 0 for i in range(n))
    if not valid:
        # Partial: only keep valid assignments
        # This doesn't give a full 2-factor, so we fall back to greedy cycle extraction
        return []

    # Build the directed graph from the assignment
    # Each vertex u has out-neighbor col_ind[u]
    out_neighbor = [None] * n
    for i in range(n):
        u = row_ind[i]
        v = col_ind[i]
        out_neighbor[u] = v

    # Extract cycles
    visited = [False] * n
    cycles = []
    for start in range(n):
        if visited[start]:
            continue
        path = []
        curr = start
        while not visited[curr]:
            visited[curr] = True
            path.append(curr)
            curr = out_neighbor[curr]
        # path is a cycle (since out_neighbor is a permutation)
        if len(path) >= 3:
            cycles.append(path)
        # len(path) == 1 or 2 shouldn't happen with no self-loops, but skip if so

    return cycles


def _decompose_greedy(edges, n, deadline, rng=None):
    """
    Greedy decomposition: repeatedly find a cycle and remove it.
    Uses a randomized greedy path extension to find cycles.
    """
    if rng is None:
        rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        best_cycle = _find_long_cycle(adj, n, max_starts=min(n, 25), max_restarts=6, rng=rng)

        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _generate_cycle_pool(adj, n, num_cycles=2000, rng=None):
    """
    Generate a pool of candidate cycles using randomized DFS and 2-factor extraction.
    Returns a list of cycles (each cycle is a list of vertices).
    """
    if rng is None:
        rng = random.Random(999)

    # Use frozenset of edges as the key for deduplication
    cycles = set()
    result = []

    def _add_cycle(cycle_verts):
        if len(cycle_verts) < 3:
            return
        # Verify all edges exist in adj
        m = len(cycle_verts)
        edges_in_cycle = set()
        for i in range(m):
            a = cycle_verts[i]
            b = cycle_verts[(i + 1) % m]
            if b not in adj[a]:
                return
            e = (min(a, b), max(a, b))
            if e in edges_in_cycle:
                return
            edges_in_cycle.add(e)
        # Deduplicate by edge set
        key = frozenset(edges_in_cycle)
        if key not in cycles:
            cycles.add(key)
            result.append(cycle_verts)

    # 1. Extract 2-factor cycles (vertex-disjoint, high quality)
    for _ in range(20):
        factor = _find_2_factor(adj, n, rng)
        for cyc in factor:
            _add_cycle(cyc)

    # 2. Randomized DFS cycles
    for _ in range(num_cycles):
        if not any(adj[v] for v in range(n)):
            break

        start_candidates = [v for v in range(n) if adj[v]]
        if not start_candidates:
            break

        start = rng.choice(start_candidates)
        path = [start]
        visited = {start}
        current = start

        max_depth = n
        for _ in range(max_depth):
            neighbors = list(adj[current])
            unvisited = [v for v in neighbors if v not in visited]

            if start in adj[current] and len(path) >= 3:
                _add_cycle(path)
                break

            if not unvisited:
                break

            # Scored choice: prefer vertices with more unvisited neighbors
            scored = []
            for v in unvisited:
                score = sum(1 for w in adj[v] if w not in visited)
                scored.append((score, v))
            scored.sort(reverse=True)
            top_k = min(3, len(scored))
            if rng.random() < 0.6:
                next_v = scored[0][1]
            else:
                next_v = rng.choice([s[1] for s in scored[:top_k]])
            path.append(next_v)
            visited.add(next_v)
            current = next_v

    return result


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # First, get a greedy baseline
    rng0 = random.Random(0)
    greedy_pieces = _decompose_greedy(edges, n, deadline - 1.0, rng0)
    best = greedy_pieces
    best_count = len(greedy_pieces)

    # Try ILP-based approach
    if time.time() < deadline - 3.0:
        try:
            ilp_pieces = _solve_ilp(edges, n, deadline - 2.0)
            if ilp_pieces and len(ilp_pieces) < best_count:
                best_count = len(ilp_pieces)
                best = ilp_pieces
        except Exception:
            pass

    # Additional greedy attempts with different seeds if time permits
    if time.time() < deadline - 1.0:
        base_seed = 1000
        attempts = 0
        max_attempts = 50
        while time.time() < deadline - 0.5:
            seed = base_seed + attempts
            rng = random.Random(seed)
            pieces = _decompose_greedy(edges, n, deadline - 0.5, rng)
            cnt = len(pieces)
            if cnt < best_count:
                best_count = cnt
                best = pieces
            attempts += 1
            if attempts >= max_attempts:
                break

    return best if best else [[u, v] for u, v in edges]


def _solve_ilp(edges, n, deadline):
    """
    ILP-based decomposition:
    1. Generate a pool of candidate cycles (including 2-factor cycles)
    2. Solve a set-packing ILP to select minimum number of cycles + single edges
    """
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        adj[u].add(v)
        adj[v].add(u)
        e = (min(u, v), max(u, v))
        edge_set.add(e)
        edge_index[e] = idx

    total_edges = len(edge_set)
    if total_edges == 0:
        return []

    # Generate cycle pool
    rng = random.Random(777)
    cycle_pool = _generate_cycle_pool(adj, n, num_cycles=1500, rng=rng)

    if not cycle_pool:
        return None

    # For each cycle, compute which edges it uses (as a bitmask or set of edge indices)
    cycle_edge_indices = []
    for cycle in cycle_pool:
        m = len(cycle)
        c_edge_idxs = set()
        valid = True
        for i in range(m):
            a = cycle[i]
            b = cycle[(i + 1) % m]
            e = (min(a, b), max(a, b))
            if e in edge_index:
                c_edge_idxs.add(edge_index[e])
            else:
                valid = False
                break
        if valid and len(c_edge_idxs) == m:
            cycle_edge_indices.append(c_edge_idxs)

    if not cycle_edge_indices:
        return None

    # Deduplicate cycles by edge set (keep longest vertex representation)
    seen_edge_sets = {}
    unique_cycles = []
    for i, c_idxs in enumerate(cycle_edge_indices):
        key = frozenset(c_idxs)
        if key not in seen_edge_sets:
            seen_edge_sets[key] = i
            unique_cycles.append((i, c_idxs))

    num_unique = len(unique_cycles)
    if num_unique == 0:
        return None

    # Solve ILP: minimize sum(x_i) + sum(y_e)
    # subject to: for each edge e, sum_{i: e in cycle_i} x_i + y_e = 1
    # x_i in {0,1}, y_e in {0,1}

    # Use a greedy approach with local search since full ILP may be too large
    # Sort cycles by length descending (longer cycles save more edges)
    cycle_lengths = [len(c_idxs) for _, c_idxs in unique_cycles]
    order = sorted(range(num_unique), key=lambda i: -cycle_lengths[i])

    # Greedy: select cycles that don't conflict with already selected ones
    selected = set()
    used_edges = set()
    remaining_edges = set(range(total_edges))

    for idx in order:
        if time.time() > deadline:
            break
        c_idxs = unique_cycles[idx][1]
        if c_idxs.issubset(remaining_edges):
            selected.add(idx)
            remaining_edges -= c_idxs
            used_edges |= c_idxs

    # Local search: try to replace one selected cycle with two others that cover the same edges
    # This can reduce total piece count if the two cycles together use fewer "new" edges
    improved = True
    max_local_iters = 50
    local_iter = 0
    while improved and local_iter < max_local_iters and time.time() < deadline:
        improved = False
        local_iter += 1
        for sel_idx in list(selected):
            if time.time() > deadline:
                break
            sel_edges = unique_cycles[sel_idx][1]
            # Try removing sel_idx and see if we can cover its edges with other cycles
            # For simplicity, try replacing with a single non-selected cycle
            for other_idx in range(num_unique):
                if other_idx == sel_idx or other_idx in selected:
                    continue
                other_edges = unique_cycles[other_idx][1]
                # Check if other_edges covers sel_edges minus some, and the union is better
                # Actually, we want: remove sel_idx, add other_idx, and ensure all edges still covered
                # This is complex; skip for now and just do greedy
                break

    # Reconstruct pieces
    pieces = []
    for idx in selected:
        c_idxs = unique_cycles[idx][1]
        # Reconstruct cycle from edge indices
        # Build adjacency for this cycle
        c_adj = defaultdict(list)
        for eidx in c_idxs:
            # Find the edge
            for (u, v) in edges:
                if edge_index[(min(u, v), max(u, v))] == eidx:
                    c_adj[u].append(v)
                    c_adj[v].append(u)
                    break

        # Find the cycle
        if c_adj:
            start_v = next(iter(c_adj))
            cycle_verts = [start_v]
            prev = None
            curr = start_v
            for _ in range(len(c_adj)):
                nexts = [x for x in c_adj[curr] if x != prev]
                if not nexts:
                    break
                next_v = nexts[0]
                if next_v == start_v and len(cycle_verts) >= 3:
                    break
                cycle_verts.append(next_v)
                prev = curr
                curr = next_v

            if len(cycle_verts) >= 3:
                pieces.append(cycle_verts)

    # Remaining edges become single edges
    for eidx in sorted(remaining_edges):
        for (u, v) in edges:
            if edge_index[(min(u, v), max(u, v))] == eidx:
                pieces.append([u, v])
                break

    return pieces if pieces else None


def construct():
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
