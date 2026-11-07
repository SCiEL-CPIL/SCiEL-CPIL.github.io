<<<<<<< SEARCH
"""Triangle-free diameter-four augmentation with simulated annealing."""
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj


def _bfs_all_dists(n, adj):
    """Return list of distance arrays from each source."""
    dists = []
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        dists.append(dist)
    return dists


def _max_diameter(dists):
    md = 0
    for dist in dists:
        for d in dist:
            if d == -1:
                return -1
            if d > md:
                md = d
    return md


def _is_triangle_free(adj, n, added_edges):
    """Check if adding these edges to base keeps triangle-free."""
    adj2 = [row[:] for row in adj]
    for u, v in added_edges:
        if adj2[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj2[u][w] and adj2[v][w]:
                return False
        adj2[u][v] = True
        adj2[v][u] = True
    return True


def _diameter_with_edges(n, base_adj, added_edges):
    """Compute diameter after adding edges; return -1 if invalid."""
    if not _is_triangle_free(base_adj, n, added_edges):
        return -1
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        adj[u][v] = True
        adj[v][u] = True
    dists = _bfs_all_dists(n, adj)
    return _max_diameter(dists)


def _precompute_valid_pairs(n, base_adj):
    """Find all pairs that can be added without creating a triangle."""
    valid = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v]:
                # Check no common neighbor
                ok = True
                for w in range(n):
                    if w != u and w != v and base_adj[u][w] and base_adj[v][w]:
                        ok = False
                        break
                if ok:
                    valid.append((u, v))
    return valid


def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])

    # Precompute all valid single-edge additions
    valid_pairs = _precompute_valid_pairs(n, base_adj)
    valid_set = set(valid_pairs)

    # Initial solution: known 8-edge solution
    seed_edges = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    best_edges = list(seed_edges)
    best_diam = _diameter_with_edges(n, base_adj, best_edges)

    # Simulated annealing: try to minimize number of edges
    # State: set of edges. Move: remove one edge, or swap one edge.
    random.seed(42)

    # First, try greedy reduction: can we remove any edge and still have diam <= 4?
    for attempt in range(200):
        if time.time() > deadline:
            break
        current = list(best_edges)
        improved = False
        for i in range(len(current)):
            candidate = [e for j, e in enumerate(current) if j != i]
            d = _diameter_with_edges(n, base_adj, candidate)
            if 0 <= d <= 4:
                best_edges = candidate
                best_diam = d
                improved = True
                break
        if not improved:
            break

    # Simulated annealing to try to find smaller solutions
    if len(best_edges) > 5:
        current = list(best_edges)
        current_diam = _diameter_with_edges(n, base_adj, current)
        T = 1.0
        cooling = 0.995
        min_T = 0.01

        while T > min_T and time.time() < deadline:
            # Choose a move
            move_type = random.random()
            if move_type < 0.4 and len(current) > 0:
                # Remove a random edge
                idx = random.randint(0, len(current) - 1)
                new_edges = [e for j, e in enumerate(current) if j != idx]
            elif move_type < 0.8 and len(current) > 0:
                # Swap: remove one, add one
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                new_edges = [e for j, e in enumerate(current) if j != idx]
                # Add a random valid edge not already present
                current_set = set(new_edges)
                candidates = [p for p in valid_pairs if p not in current_set]
                if candidates:
                    add_edge = random.choice(candidates)
                    new_edges.append(add_edge)
                else:
                    continue
            else:
                # Add a random edge
                current_set = set(current)
                candidates = [p for p in valid_pairs if p not in current_set]
                if candidates:
                    add_edge = random.choice(candidates)
                    new_edges = current + [add_edge]
                else:
                    continue

            new_diam = _diameter_with_edges(n, base_adj, new_edges)

            # Cost: prefer fewer edges, then valid diameter
            old_cost = len(current) * 100 + (0 if (0 <= current_diam <= 4) else 1000)
            new_cost = len(new_edges) * 100 + (0 if (0 <= new_diam <= 4) else 1000)
            delta = new_cost - old_cost

            if delta <= 0 or random.random() < min(1.0, max(0.0, 1.0 - delta / (T * 100))):
                current = new_edges
                current_diam = new_diam
                if 0 <= current_diam <= 4 and len(current) < len(best_edges):
                    best_edges = list(current)
                    best_diam = current_diam

            T *= cooling

    # Final greedy local search
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        # Try removing edges
        for i in range(len(best_edges)):
            candidate = [e for j, e in enumerate(best_edges) if j != i]
            d = _diameter_with_edges(n, base_adj, candidate)
            if 0 <= d <= 4:
                best_edges = candidate
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add another
        for i in range(len(best_edges)):
            if time.time() > deadline:
                break
            base = [e for j, e in enumerate(best_edges) if j != i]
            base_set = set(base)
            for cand in valid_pairs:
                if cand in base_set:
                    continue
                new_e = base + [cand]
                d = _diameter_with_edges(n, base_adj, new_e)
                if 0 <= d <= 4 and len(new_e) < len(best_edges):
                    best_edges = new_e
                    improved = True
                    break
            if improved:
                break

    # Validate final answer
    final_diam = _diameter_with_edges(n, base_adj, best_edges)
    if final_diam < 0 or final_diam > 4:
        # Fallback to seed
        best_edges = seed_edges

    return [list(e) for e in best_edges]
=======
"""Triangle-free diameter-four augmentation via bipartite 4-hop coverage."""
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj


def _bfs_diameter(n, adj):
    """Compute graph diameter via BFS from all sources."""
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d == -1:
                return -1
            if d > max_d:
                max_d = d
    return max_d


def _is_triangle_free(adj, n, added_edges):
    """Check if adding these edges keeps the graph triangle-free."""
    adj2 = [row[:] for row in adj]
    for u, v in added_edges:
        if adj2[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj2[u][w] and adj2[v][w]:
                return False
        adj2[u][v] = True
        adj2[v][u] = True
    return True


def _diameter_with_edges(n, base_adj, added_edges):
    """Compute diameter after adding edges; return -1 if invalid."""
    if not _is_triangle_free(base_adj, n, added_edges):
        return -1
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        adj[u][v] = True
        adj[v][u] = True
    return _bfs_diameter(n, adj)


def _precompute_valid_pairs(n, base_adj):
    """Find all non-adjacent pairs with no common neighbor (triangle-safe)."""
    valid = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v]:
                ok = True
                for w in range(n):
                    if w != u and w != v and base_adj[u][w] and base_adj[v][w]:
                        ok = False
                        break
                if ok:
                    valid.append((u, v))
    return valid


def _compute_reachability_4hop(n, adj):
    """
    For each vertex, compute the set of vertices reachable within 4 hops.
    Returns a list of sets (one per source vertex).
    """
    reach = [set() for _ in range(n)]
    for src in range(n):
        # BFS up to depth 4
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= 4:
                continue
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for v in range(n):
            if dist[v] >= 0 and dist[v] <= 4:
                reach[src].add(v)
    return reach


def _find_uncovered_pairs(n, reach):
    """Find pairs (u,v) where u is not in reach[v] or v is not in reach[u]."""
    uncovered = set()
    for u in range(n):
        for v in range(u + 1, n):
            if v not in reach[u]:
                uncovered.add((u, v))
    return uncovered


def _edge_covers_pairs(n, edge, adj, reach):
    """
    Given an edge (u,v), determine which currently-uncovered pairs it would cover.
    A pair (a,b) is covered if after adding edge (u,v), a can reach b in <= 4 hops.
    We approximate: adding edge (u,v) creates new paths through u-v.
    """
    # New paths: a -> ... -> u -> v -> ... -> b or a -> ... -> v -> u -> ... -> b
    # We compute the new reachability incrementally.
    covered = set()
    for a in range(n):
        for b in range(a + 1, n):
            if b in reach[a]:
                continue  # already covered
            # Check if a can reach u in <= k hops and v can reach b in <= (4-k-1) hops
            # Or a can reach v in <= k hops and u can reach b in <= (4-k-1) hops
            # This is expensive; use a simpler heuristic:
            # If a is adjacent to u or v, and b is adjacent to the other, it might help.
            # For exactness, we'd need full BFS, but that's too slow.
            # Heuristic: mark as potentially covered if a is close to {u,v} and b is close to {u,v}
            a_to_u = 1 if adj[a][u] else (2 if any(adj[a][x] and adj[x][u] for x in range(n) if x != a and x != u) else 3)
            a_to_v = 1 if adj[a][v] else (2 if any(adj[a][x] and adj[x][v] for x in range(n) if x != a and x != v) else 3)
            b_to_u = 1 if adj[b][u] else (2 if any(adj[b][x] and adj[x][u] for x in range(n) if x != b and x != u) else 3)
            b_to_v = 1 if adj[b][v] else (2 if any(adj[b][x] and adj[x][v] for x in range(n) if x != b and x != v) else 3)
            
            # Path a->u->v->b: a_to_u + 1 + b_to_v
            # Path a->v->u->b: a_to_v + 1 + b_to_u
            if a_to_u + 1 + b_to_v <= 4 or a_to_v + 1 + b_to_u <= 4:
                covered.add((a, b))
    return covered


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])
    valid_pairs = _precompute_valid_pairs(n, base_adj)
    valid_set = set(valid_pairs)

    # Fallback known solution (8 edges)
    seed_edges = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    best_edges = list(seed_edges)

    # Verify seed is valid
    seed_diam = _diameter_with_edges(n, base_adj, best_edges)
    if seed_diam < 0 or seed_diam > 4:
        return [list(e) for e in seed_edges]

    # Phase 1: Greedy set-cover style edge selection
    # Start with empty edge set, greedily add edges that cover the most uncovered pairs
    current_edges = []
    current_adj = [row[:] for row in base_adj]
    current_reach = _compute_reachability_4hop(n, current_adj)
    uncovered = _find_uncovered_pairs(n, current_reach)

    while uncovered and time.time() < deadline - 10.0:
        best_edge = None
        best_covered = set()
        best_count = 0

        for edge in valid_pairs:
            if edge in set(current_edges):
                continue
            # Quick check: would this edge create a triangle with existing edges?
            u, v = edge
            triangle_ok = True
            for w in range(n):
                if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                    triangle_ok = False
                    break
            if not triangle_ok:
                continue

            covered = _edge_covers_pairs(n, edge, current_adj, current_reach)
            if len(covered) > best_count:
                best_count = len(covered)
                best_edge = edge
                best_covered = covered

        if best_edge is None or best_count == 0:
            break

        current_edges.append(best_edge)
        current_adj[best_edge[0]][best_edge[1]] = True
        current_adj[best_edge[1]][best_edge[0]] = True
        current_reach = _compute_reachability_4hop(n, current_adj)
        uncovered = _find_uncovered_pairs(n, current_reach)

    # Verify greedy solution
    greedy_diam = _diameter_with_edges(n, base_adj, current_edges)
    if 0 <= greedy_diam <= 4:
        if len(current_edges) < len(best_edges):
            best_edges = list(current_edges)

    # Phase 2: Local search to reduce edge count
    if len(best_edges) > 4 and time.time() < deadline - 5.0:
        # Try removing each edge and check if diameter stays <= 4
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(best_edges)):
                candidate = [e for j, e in enumerate(best_edges) if j != i]
                d = _diameter_with_edges(n, base_adj, candidate)
                if 0 <= d <= 4:
                    best_edges = candidate
                    improved = True
                    break

    # Phase 3: Simulated annealing for further optimization
    if len(best_edges) > 3 and time.time() < deadline - 2.0:
        random.seed(123)
        current = list(best_edges)
        T = 2.0
        cooling = 0.99
        min_T = 0.05
        no_improve = 0

        while T > min_T and time.time() < deadline - 2.0 and no_improve < 500:
            move_type = random.random()
            if move_type < 0.5 and len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                new_edges = [e for j, e in enumerate(current) if j != idx]
            elif move_type < 0.9 and len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                new_edges = [e for j, e in enumerate(current) if j != idx]
                current_set = set(new_edges)
                candidates = [p for p in valid_pairs if p not in current_set]
                if candidates:
                    new_edges.append(random.choice(candidates))
                else:
                    continue
            else:
                current_set = set(current)
                candidates = [p for p in valid_pairs if p not in current_set]
                if candidates:
                    new_edges = current + [random.choice(candidates)]
                else:
                    continue

            new_diam = _diameter_with_edges(n, base_adj, new_edges)
            old_cost = len(current) * 10 + (0 if (0 <= _diameter_with_edges(n, base_adj, current) <= 4) else 100)
            new_cost = len(new_edges) * 10 + (0 if (0 <= new_diam <= 4) else 100)
            delta = new_cost - old_cost

            accept = False
            if delta <= 0:
                accept = True
            elif T > 0:
                accept = random.random() < min(1.0, max(0.0, 1.0 - delta / (T * 10)))

            if accept:
                current = new_edges
                if 0 <= new_diam <= 4 and len(current) < len(best_edges):
                    best_edges = list(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            T *= cooling

    # Final validation
    final_diam = _diameter_with_edges(n, base_adj, best_edges)
    if final_diam < 0 or final_diam > 4:
        best_edges = list(seed_edges)

    return [list(e) for e in best_edges]
>>>>>>> REPLACE
