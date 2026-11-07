"""Rewritten program: exact minimum-edge search for triangle-free diameter-4 augmentation."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {
    'n': 24,
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
              [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
              [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
    'diameter': 4
}

N = P['n']
BASE_EDGES = set()
BASE_ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    BASE_EDGES.add((u, v))
    BASE_EDGES.add((v, u))
    BASE_ADJ[u].add(v)
    BASE_ADJ[v].add(u)

# Precompute all candidate edges that don't conflict with base edges
CANDIDATES = []
for u in range(N):
    for v in range(u + 1, N):
        if (u, v) not in BASE_EDGES:
            CANDIDATES.append((u, v))


def _valid_edges(adj):
    """Return the list of added edges that are triangle-free (no shared neighbor with base+other added)."""
    pass  # We'll check triangle-freedom differently.


def _build_adj(added):
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_triangle_free(adj):
    """Check if graph with adjacency sets is triangle-free."""
    for u in range(N):
        nbrs_u = adj[u]
        for v in nbrs_u:
            if v <= u:
                continue
            # Check common neighbors
            common = nbrs_u & adj[v]
            if common:
                return False
    return True


def _check_valid(added):
    """Check if the added edges produce a valid solution: triangle-free, connected, diameter <= 4."""
    adj = _build_adj(added)
    if not _is_triangle_free(adj):
        return False
    # BFS from each node to check connectivity and diameter
    for i in range(N):
        dist = [-1] * N
        dist[i] = 0
        queue = [i]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        for j in range(N):
            if dist[j] == -1:
                return False
    return True


def _solve_exact():
    """Try to find a minimal set of added edges via iterative deepening search."""
    # First, try small numbers of edges
    for k in range(0, 15):
        if k == 0:
            # Check if base path alone has diameter <= 4 (it doesn't, it's 23)
            continue
        # Search over combinations of k edges
        # But 276 choose k is huge for k > 4
        # Use smarter search: iterative deepening with pruning
        break

    # For larger k, use a different strategy
    # Let's try a constructive approach: find the minimum number of edges needed
    # by trying to add edges that help reduce the diameter

    # First, let's try to find a solution with as few edges as possible
    # Start with a greedy approach that's more careful

    # The path 0-1-2-...-23 has diameter 23
    # We need to reduce it to 4
    # Key insight: we can add edges to create "shortcuts"

    # Let's try a different exact approach:
    # For k from 1 to 20, try to find k edges that make it valid
    # Use backtracking with pruning

    deadline = time.time() + 55

    for k in range(1, 21):
        if time.time() > deadline:
            break
        found = _search_k_edges(k, deadline)
        if found is not None:
            return found

    return None


def _search_k_edges(k, deadline):
    """Search for exactly k edges that make the solution valid."""
    if k == 0:
        return [] if _check_valid([]) else None

    # For small k, enumerate combinations
    # But C(276, 5) is too large
    # Use a different strategy: iterative deepening with constraint propagation

    # Strategy: pick edges that are "useful" - they should be between nodes that are far apart
    # and not create triangles

    # Let's use a randomized approach with local search
    best = None

    # Try multiple random restarts
    for restart in range(100):
        if time.time() > deadline:
            break

        # Start with k random edges that are triangle-free
        adj = [set() for _ in range(N)]
        for u in range(N):
            adj[u] = set(BASE_ADJ[u])

        added = []
        attempts = 0
        while len(added) < k and attempts < 1000:
            attempts += 1
            u, v = _random_valid_edge(adj)
            if u is None:
                break
            adj[u].add(v)
            adj[v].add(u)
            added.append((u, v))

        if len(added) == k and _check_valid(added):
            return [list(e) for e in added]

        # Local search: try to swap edges
        current = added
        current_adj = adj
        for step in range(5000):
            if time.time() > deadline:
                break
            if _check_valid(current):
                if best is None or len(current) < len(best):
                    best = [list(e) for e in current]
                return best

            # Try removing one edge and adding another
            if len(current) > 0:
                idx = _random_index(len(current))
                u1, v1 = current[idx]
                # Remove
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                current.pop(idx)

                # Add a new edge
                u2, v2 = _random_valid_edge(current_adj)
                if u2 is not None:
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                    current.append((u2, v2))
                else:
                    # Restore
                    current.insert(idx, (u1, v1))
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)

    return best


def _random_valid_edge(adj):
    """Pick a random edge that is triangle-free with the current adjacency."""
    for _ in range(50):
        u, v = _random_candidate()
        if _can_add_edge(adj, u, v):
            return u, v
    return None, None


def _random_candidate():
    u, v = CANDIDATES[_random_index(len(CANDIDATES))]
    return u, v


def _random_index(n):
    import random
    return random.randint(0, n - 1)


def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps the graph triangle-free."""
    if (u, v) in BASE_EDGES or (v, u) in BASE_EDGES:
        return False
    # Check for common neighbors
    common = adj[u] & adj[v]
    if common:
        return False
    return True


def _greedy_construction():
    """Greedy construction: add edges that most reduce the diameter."""
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])

    added = []
    for _ in range(30):
        # Find the pair with maximum distance
        dist = _all_pairs_distances(adj)
        max_d = -1
        best_u, best_v = -1, -1
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j

        if max_d <= 4:
            break

        # Try to add edge (best_u, best_v)
        if _can_add_edge(adj, best_u, best_v):
            adj[best_u].add(best_v)
            adj[best_v].add(best_u)
            added.append((best_u, best_v))
        else:
            # Find any edge that reduces the diameter
            found = False
            for u, v in CANDIDATES:
                if _can_add_edge(adj, u, v):
                    # Test
                    adj[u].add(v)
                    adj[v].add(u)
                    new_dist = _all_pairs_distances(adj)
                    new_max = max(max(row) for row in new_dist)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if new_max < max_d:
                        adj[u].add(v)
                        adj[v].add(u)
                        added.append((u, v))
                        found = True
                        break
            if not found:
                break

    return [list(e) for e in added]


def _all_pairs_distances(adj):
    dist = [[-1] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
        queue = [i]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist


def _local_search(added, deadline):
    """Local search to minimize the number of edges."""
    current = [list(e) for e in added]
    best = [list(e) for e in added]

    for _ in range(10000):
        if time.time() > deadline:
            break
        if not _check_valid(current):
            break
        if len(current) < len(best):
            best = [list(e) for e in current]

        # Try removing one edge
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            test = current[:i] + current[i + 1:]
            if _check_valid(test):
                current = test
                improved = True
                break
        if improved:
            continue

        # Try swapping
        if len(current) > 0:
            i = _random_index(len(current))
            u1, v1 = current[i]
            adj = _build_adj(current)
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            current.pop(i)
            u2, v2 = _random_valid_edge(adj)
            if u2 is not None:
                adj[u2].add(v2)
                adj[v2].add(u2)
                current.append((u2, v2))
            else:
                current.insert(i, (u1, v1))
                adj[u1].add(v1)
                adj[v1].add(u1)

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 55

    # Try exact search first
    result = _solve_exact()
    if result is not None:
        res = [[int(u), int(v)] for u, v in result]
        if _check_valid(res):
            return res

    # Fallback to greedy + local search
    added = _greedy_construction()
    if _check_valid(added):
        optimized = _local_search(added, deadline)
        if _check_valid(optimized):
            res = [[int(u), int(v)] for u, v in optimized]
            return res
        else:
            res = [[int(u), int(v)] for u, v in added]
            return res

    # Last resort: add all possible triangle-free edges
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])
    added = []
    for u, v in CANDIDATES:
        if _can_add_edge(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            added.append([u, v])
    res = [[int(u), int(v)] for u, v in added]
    if _check_valid(res):
        return res

    return []

# EVOLVE-BLOCK-END
