"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # Precompute all-pairs shortest paths (unweighted)
    dist = [[n * 2 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if dist[cur][nb] > dist[cur][cur] + 1:
                    dist[cur][nb] = dist[cur][cur] + 1
                    queue.append(nb)

    # --- Vertex-splitting flow network for exact disjoint path verification ---
    # For a given branch set B, we build a flow network where:
    # - Each non-branch vertex v is split into v_in -> v_out with capacity 1
    # - Each edge (u,v) in original graph becomes arcs between the split nodes
    # - Branch vertices have unlimited capacity (they're endpoints)
    # - We need to route C(|B|,2) units of flow, one for each pair
    # 
    # Actually, for pairwise vertex-disjoint paths, we need a different approach.
    # We use a flow network per pair with vertex capacities, but that's O(k^2) flows.
    # 
    # Better: Use a single flow network with super-source/sink approach.
    # But pairwise disjoint paths is hard. Let's use the sequential BFS with better heuristics.

    def find_path_bfs(u, v, forbidden):
        """BFS from u to v avoiding forbidden vertices."""
        if u == v:
            return None
        parent = [-1] * n
        parent[u] = u
        queue = deque([u])
        while queue:
            cur = queue.popleft()
            if cur == v:
                path = []
                c = v
                while c != u:
                    path.append(c)
                    c = parent[c]
                path.append(u)
                path.reverse()
                return path
            for nb in adj[cur]:
                if parent[nb] == -1 and nb not in forbidden:
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path_bfs_shortest(u, v, forbidden):
        """BFS returning shortest path, with tie-breaking by preferring vertices with higher degree."""
        if u == v:
            return None
        parent = [-1] * n
        parent[u] = u
        queue = deque([u])
        while queue:
            cur = queue.popleft()
            if cur == v:
                path = []
                c = v
                while c != u:
                    path.append(c)
                    c = parent[c]
                path.append(u)
                path.reverse()
                return path
            # Sort neighbors by degree descending to prefer "hub" vertices
            neighbors = sorted([nb for nb in adj[cur] if parent[nb] == -1 and nb not in forbidden],
                               key=lambda x: degrees[x], reverse=True)
            for nb in neighbors:
                parent[nb] = cur
                queue.append(nb)
        return None

    def check_and_get_paths(B, num_restarts=200):
        """Try to find vertex-disjoint paths for all pairs in B."""
        B_set = set(B)
        k = len(B_set)
        if k < 2:
            return False, None

        pairs = list(itertools.combinations(sorted(B_set), 2))
        num_pairs = len(pairs)

        # Precompute for each pair: shortest path length in G - (B - {u,v})
        # Use this to order pairs (most constrained first)
        pair_costs = []
        for u, v in pairs:
            forbidden = B_set - {u, v}
            # Quick BFS to get distance
            d = n * 2
            if dist[u][v] <= n:  # connected in full graph
                # Check if there's a path avoiding other branches
                parent = [-1] * n
                parent[u] = u
                queue = deque([u])
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    if cur == v:
                        found = True
                        break
                    for nb in adj[cur]:
                        if parent[nb] == -1 and nb not in forbidden:
                            parent[nb] = cur
                            queue.append(nb)
                if found:
                    # Count path length
                    d = 0
                    c = v
                    while c != u:
                        c = parent[c]
                        d += 1
                else:
                    d = n * 2
            pair_costs.append((d, u, v))

        # Sort by cost ascending (most constrained first)
        pair_costs.sort()

        for attempt in range(num_restarts):
            used_interiors = set()
            paths = {}
            success = True

            # Use the sorted order, but add some randomness in later attempts
            if attempt == 0:
                ordered_pairs = [(u, v) for _, u, v in pair_costs]
            elif attempt == 1:
                # Reverse order
                ordered_pairs = [(u, v) for _, u, v in reversed(pair_costs)]
            else:
                # Random shuffle
                ordered_pairs = [(u, v) for _, u, v in pair_costs]
                random.shuffle(ordered_pairs)

            for u, v in ordered_pairs:
                forbidden = (B_set - {u, v}) | used_interiors
                path = find_path_bfs(u, v, forbidden)
                if path is None:
                    # Try with degree-preferred BFS
                    path = find_path_bfs_shortest(u, v, forbidden)
                if path is None:
                    success = False
                    break
                paths[(u, v)] = path
                for node in path[1:-1]:
                    used_interiors.add(node)

            if success:
                return True, paths

        return False, None

    # --- Step 1: Find a good initial branch set ---
    # Strategy: Start with high-degree vertices and expand greedily

    # Sort vertices by degree descending
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    best_B = set()
    best_paths = None
    best_score = 0

    # Try multiple starting seeds
    seeds_to_try = []
    # Top degree vertices
    seeds_to_try.append(set(vertices_by_degree[:5]))
    # Clique-based
    # Find a large clique using greedy coloring bound
    max_clique = []

    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if time.time() > deadline - 15.0:
            return
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        if len(P) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        union_p_x = P.union(X)
        if not union_p_x:
            return
        best_pivot = None
        max_uncovered = -1
        for u in union_p_x:
            uncovered = len(P.difference(adj[u]))
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_pivot = u
        if best_pivot is None:
            return
        for v in list(P.difference(adj[best_pivot])):
            bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)

    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    if len(max_clique) >= 2:
        seeds_to_try.append(set(max_clique))

    # Also try: vertices with high "connectivity" (degree * avg_neighbor_degree)
    def connectivity_score(v):
        if not adj[v]:
            return 0
        return degrees[v] * sum(degrees[u] for u in adj[v]) / len(adj[v])

    vertices_by_connectivity = sorted(range(n), key=lambda x: connectivity_score(x), reverse=True)
    seeds_to_try.append(set(vertices_by_connectivity[:5]))

    # Try each seed
    for seed in seeds_to_try:
        if time.time() > deadline - 10.0:
            break
        valid, paths = check_and_get_paths(seed, num_restarts=50)
        if valid and len(seed) > best_score:
            best_B = set(seed)
            best_paths = paths
            best_score = len(seed)

    # If no valid seed found, use a minimal valid set
    if best_score < 2:
        # Find any two adjacent vertices
        for u, v in edges:
            best_B = {u, v}
            best_paths = {(u, v): [u, v]}
            best_score = 2
            break

    # --- Step 2: Greedy expansion ---
    current_B = set(best_B)
    current_paths = best_paths

    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False

        # Try adding each candidate vertex (sorted by degree desc)
        candidates = [v for v in vertices_by_degree if v not in current_B]

        for v in candidates:
            if time.time() > deadline - 8.0:
                break
            new_B = current_B | {v}
            valid, paths = check_and_get_paths(new_B, num_restarts=100)
            if valid:
                current_B = new_B
                current_paths = paths
                if len(current_B) > best_score:
                    best_B = set(current_B)
                    best_paths = paths
                    best_score = len(current_B)
                improved = True
                break  # Restart with new B

        if not improved:
            # Try swaps: remove one from B, add one not in B
            if len(current_B) > 2:
                # Remove lowest degree vertex in B
                b_sorted = sorted(current_B, key=lambda x: degrees[x])
                candidates_not_in_b = [v for v in vertices_by_degree if v not in current_B]

                for remove_v in b_sorted[:2]:
                    for add_v in candidates_not_in_b[:5]:
                        if time.time() > deadline - 8.0:
                            break
                        swap_B = (current_B - {remove_v}) | {add_v}
                        valid, paths = check_and_get_paths(swap_B, num_restarts=50)
                        if valid:
                            current_B = swap_B
                            current_paths = paths
                            improved = True
                            break
                    if improved:
                        break

    # --- Step 3: Try larger sets via random sampling ---
    if time.time() < deadline - 10.0:
        # Try random subsets of size best_score + 1 and best_score + 2
        for target_size in [best_score + 1, best_score + 2]:
            if target_size > n:
                break
            for _ in range(30):
                if time.time() > deadline - 10.0:
                    break
                # Random sample of target_size vertices, biased toward high degree
                weights = [degrees[v] + 1 for v in range(n)]
                sample = random.sample(range(n), target_size)
                valid, paths = check_and_get_paths(sample, num_restarts=30)
                if valid and target_size > best_score:
                    best_B = set(sample)
                    best_paths = paths
                    best_score = target_size

    # --- Step 4: ILP for branch selection with connectivity constraints ---
    if time.time() < deadline - 15.0:
        # ILP: Maximize sum(x_v) subject to cut constraints
        # For each subset S with |S| < n, the number of branches in S must be <= 
        # the max number of vertex-disjoint paths that can be routed... 
        # This is complex. Instead, use a simpler approach:
        # Maximize sum(x_v) - penalty * sum_{u<v} dist[u][v] * x_u * x_v
        # The penalty encourages clustering.

        num_vars = n
        c = np.zeros(num_vars)
        # Objective: maximize sum(x_v) - lambda * sum dist[u][v] x_u x_v
        # Linearize: minimize -sum(x_v) + lambda * sum dist[u][v] y_{uv}
        # where y_{uv} >= x_u + x_v - 1, y_{uv} <= x_u, y_{uv} <= x_v

        # Actually, let's just use the greedy result and try to improve with local search
        pass

    # --- Final verification and output ---
    if best_paths is None:
        valid, paths = check_and_get_paths(best_B, num_restarts=500)
        if valid:
            best_paths = paths
        else:
            # Fallback: use direct edges for a clique
            best_B = set(max_clique) if len(max_clique) >= 2 else {0, 1}
            best_paths = {}
            for u, v in itertools.combinations(sorted(best_B), 2):
                if v in adj[u]:
                    best_paths[(u, v)] = [u, v]
                else:
                    # Find any path
                    p = find_path_bfs(u, v, set())
                    if p:
                        best_paths[(u, v)] = p
                    else:
                        best_paths[(u, v)] = [u, v]  # Shouldn't happen

    branches = sorted([int(v) for v in best_B])
    paths_list = []
    for u, v in itertools.combinations(branches, 2):
        key = (u, v)
        if key not in best_paths:
            key = (v, u)
        if key in best_paths:
            p = best_paths[key]
        else:
            p = [u, v]
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}

# EVOLVE-BLOCK-END
