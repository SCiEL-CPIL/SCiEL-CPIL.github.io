import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.92

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION

    def _popcount(x):
        return bin(x).count('1')

    def _is_valid_add(adj, u, v):
        nv = adj[v]
        neigh_v = [i for i in range(N) if (nv >> i) & 1]
        for x, y in combinations(neigh_v, 2):
            common = adj[u] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False

        nu = adj[u]
        neigh_u = [i for i in range(N) if (nu >> i) & 1]
        for x, y in combinations(neigh_u, 2):
            common = adj[v] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False

        return True

    def _add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def _remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def _edge_count(adj):
        return sum(_popcount(a) for a in adj) // 2

    def _local_search_add(adj, max_iter=50):
        improved = True
        iters = 0
        while improved and time.time() < deadline and iters < max_iter:
            iters += 1
            improved = False
            pairs = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            random.shuffle(pairs)
            for u, v in pairs:
                if _is_valid_add(adj, u, v):
                    _add_edge(adj, u, v)
                    improved = True
                    if time.time() > deadline:
                        break
        return adj

    def _edge_swap_search(adj, max_time=5.0):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]

            if not missing:
                break

            random.shuffle(existing)
            random.shuffle(missing)

            for u1, v1 in existing:
                if time.time() > deadline:
                    break
                for u2, v2 in missing:
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue

                    _remove_edge(adj, u1, v1)

                    if _is_valid_add(adj, u2, v2):
                        _add_edge(adj, u2, v2)
                        improved = True
                        existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
                        missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
                        random.shuffle(existing)
                        random.shuffle(missing)
                        break
                    else:
                        _add_edge(adj, u1, v1)
                if improved:
                    break
        return adj

    def _spectral_init():
        # Use a spectral approach: start with a complete graph and remove edges
        # that violate K3,3, prioritizing edges in high-density regions
        adj = [0] * N
        # Start with complete graph
        for u in range(N):
            for v in range(u+1, N):
                _add_edge(adj, u, v)

        # Now iteratively remove edges that create K3,3
        # Priority: edges whose removal breaks the most K3,3 subgraphs
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find all K3,3 subgraphs
            k33s = []
            for a, b, c in combinations(range(N), 3):
                common = adj[a] & adj[b] & adj[c]
                common_list = [i for i in range(N) if (common >> i) & 1]
                if len(common_list) >= 3:
                    for d, e, f in combinations(common_list, 3):
                        # Check if {a,b,c} and {d,e,f} form a K3,3
                        # We need all 9 edges present
                        edges_needed = [
                            (a,d), (a,e), (a,f),
                            (b,d), (b,e), (b,f),
                            (c,d), (c,e), (c,f)
                        ]
                        all_present = True
                        for x, y in edges_needed:
                            if not ((adj[min(x,y)] >> max(x,y)) & 1):
                                all_present = False
                                break
                        if all_present:
                            k33s.append((a, b, c, d, e, f))

            if not k33s:
                break

            # Count how many times each edge appears in K3,3 subgraphs
            edge_counts = {}
            for k33 in k33s:
                a, b, c, d, e, f = k33
                edges_in_k33 = [
                    (a,d), (a,e), (a,f),
                    (b,d), (b,e), (b,f),
                    (c,d), (c,e), (c,f)
                ]
                for x, y in edges_in_k33:
                    key = (min(x,y), max(x,y))
                    edge_counts[key] = edge_counts.get(key, 0) + 1

            # Remove the edge that appears in the most K3,3 subgraphs
            if edge_counts:
                best_edge = max(edge_counts.keys(), key=lambda e: edge_counts[e])
                _remove_edge(adj, best_edge[0], best_edge[1])
                changed = True

        return adj

    def _greedy_init():
        adj = [0] * N
        all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if _is_valid_add(adj, u, v):
                _add_edge(adj, u, v)
            if time.time() > deadline:
                break
        return adj

    best_adj = None
    best_count = 0

    # Try spectral initialization first
    if time.time() < deadline:
        adj = _spectral_init()
        adj = _local_search_add(adj)
        adj = _edge_swap_search(adj, max_time=10.0)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    # Multiple restarts with greedy init
    restarts = 0
    while time.time() < deadline and restarts < 150:
        restarts += 1

        if best_adj is not None and restarts > 5:
            # Perturb best solution
            adj = best_adj[:]
            num_remove = random.randint(2, 6)
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
            random.shuffle(existing)
            for i in range(min(num_remove, len(existing))):
                u, v = existing[i]
                _remove_edge(adj, u, v)
        else:
            adj = _greedy_init()

        adj = _local_search_add(adj)
        adj = _edge_swap_search(adj, max_time=8.0)

        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
