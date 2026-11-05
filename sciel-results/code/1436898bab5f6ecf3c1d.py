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

    # Try starting from each vertex with some randomization
    num_starts = min(n, max_starts)
    if n <= max_starts:
        starts = list(range(n))
    else:
        # Prioritize high-degree vertices
        degrees = [len(adj[v]) for v in range(n)]
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
                    # Check if we can close the cycle
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Heuristic: prefer neighbors with more unvisited connections
                    # Also consider closing early if path is already long
                    if len(path) >= 4 and start in adj[current] and rng.random() < 0.3:
                        # Close the cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    # Score each unvisited neighbor
                    scored = []
                    for v in unvisited:
                        # Count unvisited neighbors of v
                        unvisited_neighbors = sum(1 for w in adj[v] if w not in visited)
                        score = unvisited_neighbors
                        # Slight preference for closing to start if possible
                        if v == start:
                            score += 10
                        scored.append((score, v))

                    # Pick the best or randomly from top few
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

        # Try to find a long cycle
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
            # No cycle found, remove a single edge
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

    # Add any remaining edges as single edges
    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _generate_cycle_pool(adj, n, num_cycles=500, rng=None):
    """
    Generate a pool of candidate cycles using randomized DFS.
    Returns a list of cycles (each cycle is a list of vertices).
    """
    if rng is None:
        rng = random.Random(999)

    cycles = set()

    for _ in range(num_cycles):
        if not any(adj[v] for v in range(n)):
            break

        # Pick a random start vertex with neighbors
        start_candidates = [v for v in range(n) if adj[v]]
        if not start_candidates:
            break

        start = rng.choice(start_candidates)
        path = [start]
        visited = {start}
        current = start

        # Randomized DFS to find a cycle back to start
        max_depth = n
        found = False

        for _ in range(max_depth):
            neighbors = list(adj[current])
            unvisited = [v for v in neighbors if v not in visited]

            if start in adj[current] and len(path) >= 3:
                # Found a cycle
                cycle = tuple(sorted(tuple(sorted((path[i], path[(i+1) % len(path)]))) for i in range(len(path))))
                if len(path) >= 3:
                    cycles.add(frozenset(cycle))
                found = True
                break

            if not unvisited:
                break

            # Randomly pick next vertex
            next_v = rng.choice(unvisited)
            path.append(next_v)
            visited.add(next_v)
            current = next_v

    # Convert to list of vertex sequences
    result = []
    for cyc in cycles:
        # Reconstruct a valid cycle from the edge set
        if len(cyc) < 3:
            continue
        # Build adjacency for this cycle
        cyc_adj = defaultdict(list)
        for e in cyc:
            u, v = e
            cyc_adj[u].append(v)
            cyc_adj[v].append(u)

        # Find a Hamiltonian cycle in this subgraph
        if all(len(cyc_adj[v]) == 2 for v in cyc_adj) and len(cyc_adj) >= 3:
            # It's a simple cycle, extract vertices
            start_v = next(iter(cyc_adj))
            cycle_verts = [start_v]
            prev = None
            curr = start_v
            for _ in range(len(cyc_adj)):
                nexts = [x for x in cyc_adj[curr] if x != prev]
                if not nexts:
                    break
                next_v = nexts[0]
                if next_v == start_v and len(cycle_verts) >= 3:
                    break
                cycle_verts.append(next_v)
                prev = curr
                curr = next_v

            if len(cycle_verts) >= 3:
                result.append(cycle_verts)

    return result


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try multiple random strategies
    base_seed = 12345
    attempts = 0
    max_attempts = 150

    while time.time() < deadline - 1.0:
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

    # Try ILP-based approach if time permits
    if time.time() < deadline - 5.0:
        try:
            ilp_pieces = _solve_ilp(edges, n, deadline - 2.0)
            if ilp_pieces and len(ilp_pieces) < best_count:
                best_count = len(ilp_pieces)
                best = ilp_pieces
        except Exception:
            pass

    return best if best else [[u, v] for u, v in edges]


def _solve_ilp(edges, n, deadline):
    """
    ILP-based decomposition:
    1. Generate a pool of candidate cycles
    2. Solve ILP to select minimum number of cycles + single edges
    """
    from scipy.optimize import linprog

    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(tuple(sorted((u, v))))

    total_edges = len(edge_set)
    if total_edges == 0:
        return []

    # Generate cycle pool
    rng = random.Random(777)
    cycle_pool = _generate_cycle_pool(adj, n, num_cycles=300, rng=rng)

    if not cycle_pool:
        return None

    # For each cycle, compute which edges it uses
    cycle_edges = []
    for cycle in cycle_pool:
        m = len(cycle)
        c_edges = set()
        valid = True
        for i in range(m):
            a = cycle[i]
            b = cycle[(i + 1) % m]
            e = tuple(sorted((a, b)))
            if e in edge_set:
                c_edges.add(e)
            else:
                valid = False
                break
        if valid and len(c_edges) == m:
            cycle_edges.append(c_edges)

    if not cycle_edges:
        return None

    num_cycles = len(cycle_edges)
    num_edges = total_edges

    # ILP: minimize sum(x_i) + sum(y_j)
    # subject to: for each edge e, sum_{i: e in cycle_i} x_i + y_e = 1
    # x_i, y_e in {0, 1}

    # Use a greedy subset selection approach instead of full ILP for speed
    # Sort cycles by length (longer first)
    cycle_edges_sorted = sorted(cycle_edges, key=lambda ce: -len(ce))

    # Greedy selection with conflict resolution
    selected_cycles = []
    used_edges = set()
    remaining_edges = set(edge_set)

    for c_edges in cycle_edges_sorted:
        if time.time() > deadline:
            break
        if c_edges.issubset(remaining_edges):
            selected_cycles.append(c_edges)
            remaining_edges -= c_edges
            used_edges |= c_edges

    # Remaining edges become single edges
    pieces = []
    for c_edges in selected_cycles:
        # Reconstruct cycle from edges
        # Build adjacency
        c_adj = defaultdict(list)
        for u, v in c_edges:
            c_adj[u].append(v)
            c_adj[v].append(u)

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

    for e in remaining_edges:
        pieces.append(list(e))

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
