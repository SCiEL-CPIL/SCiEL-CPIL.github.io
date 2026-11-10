<<<<<<< SEARCH
import time
from collections import deque
from itertools import combinations
=======
import time
from collections import deque
from itertools import combinations
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    best_branches = None
    best_paths = None

    def bfs_path(start, end, forbidden, adj):
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for v in adj[u]:
                if v in visited:
                    continue
                if v in forbidden and v != end:
                    continue
                visited.add(v)
                parent[v] = u
                queue.append(v)
        return None

    def try_branch_set(branches):
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        # Try multiple orderings and different path-finding strategies
        import random
        rng = random.Random(42)

        # Strategy 1: Greedy with BFS shortest path
        def find_paths_greedy(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths

            # Try each remaining pair
            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                forbidden = used_int | (branch_set - {a, b})
                path = bfs_path(a, b, forbidden, adj)
                if path is not None:
                    interior = set(path[1:-1])
                    if not (interior & used_int):
                        new_used = used_int | interior
                        result = find_paths_greedy(rest, new_used, current_paths + [path])
                        if result is not None:
                            return result
            return None

        # Strategy 2: Greedy but try to find longer paths when shortest fails
        def bfs_path_length(start, end, forbidden, adj, max_len):
            """Find a path of length <= max_len from start to end avoiding forbidden."""
            if start == end:
                return [start]
            if max_len <= 1:
                if end in adj[start] and end not in forbidden:
                    return [start, end]
                return None

            visited = {start}
            parent = {start: None}
            depth = {start: 0}
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if depth[u] >= max_len:
                    continue
                if u == end:
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                for v in adj[u]:
                    if v in visited:
                        continue
                    if v in forbidden and v != end:
                        continue
                    visited.add(v)
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    queue.append(v)
            return None

        def find_paths_greedy_long(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths

            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                forbidden = used_int | (branch_set - {a, b})
                # Try increasing path lengths
                for max_len in range(2, n + 1):
                    if time.time() > deadline:
                        return None
                    path = bfs_path_length(a, b, forbidden, adj, max_len)
                    if path is not None:
                        interior = set(path[1:-1])
                        if not (interior & used_int):
                            new_used = used_int | interior
                            result = find_paths_greedy_long(rest, new_used, current_paths + [path])
                            if result is not None:
                                return result
                    # If we found any path at this length but couldn't complete, 
                    # try next length (the path might block others)
                    # Actually, let's just try the shortest that works
                    break
            return None

        # Try natural order with greedy
        result = find_paths_greedy(pairs, set(), [])
        if result is not None:
            return result

        # Try sorted by "difficulty" (shorter available paths first)
        def pair_difficulty(pair):
            a, b = pair
            forbidden = branch_set - {a, b}
            path = bfs_path(a, b, forbidden, adj)
            return len(path) if path else 999

        sorted_pairs = sorted(pairs, key=pair_difficulty)
        result = find_paths_greedy(sorted_pairs, set(), [])
        if result is not None:
            return result

        # Try randomized orderings
        for attempt in range(10):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_greedy(shuffled, set(), [])
            if result is not None:
                return result

        # Try with longer path search
        for attempt in range(5):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_greedy_long(shuffled, set(), [])
            if result is not None:
                return result

        return None
=======
def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    best_branches = None
    best_paths = None

    def bfs_path(start, end, forbidden, adj):
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for v in adj[u]:
                if v in visited:
                    continue
                if v in forbidden and v != end:
                    continue
                visited.add(v)
                parent[v] = u
                queue.append(v)
        return None

    def max_flow_disjoint_paths(branches):
        """
        Try to find vertex-disjoint paths between all pairs of branches using max-flow.
        Returns list of paths if successful, else None.
        """
        k = len(branches)
        branch_set = set(branches)
        num_pairs = k * (k - 1) // 2
        
        # Node splitting: each vertex v becomes v_in -> v_out with capacity 1
        # For branch vertices, capacity is k (they can be endpoints of up to k-1 paths, 
        # but since each path uses 2 endpoints, a branch is endpoint of k-1 paths.
        # Actually, each branch is in k-1 pairs. So it needs capacity k-1 for its "out" side 
        # and k-1 for its "in" side? No, vertex-disjoint means each non-branch vertex 
        # is used at most once. Branch vertices are used as endpoints.
        
        # Let's use a standard construction for vertex-disjoint paths with specified terminals.
        # This is a bit tricky with max-flow because we need paths between ALL pairs.
        
        # Alternative: Use the fact that we need a set of vertex-disjoint paths that cover all pairs.
        # This is equivalent to finding a clique in the "path graph" where paths are vertex-disjoint.
        # But we can also view it as: can we route flow from each branch to every other branch?
        
        # Actually, the problem is: find vertex-disjoint paths P_{ij} for all i < j in branches.
        # This is a classic problem. One approach is to use max-flow with node splitting.
        
        # Let's try a simpler approach: for each pair, find a path, ensuring disjointness.
        # But max-flow can solve this if we model it correctly.
        
        # Construction:
        # - Split each vertex v into v_in and v_out with edge v_in -> v_out capacity 1 (or inf for branches?)
        # - For each edge (u, v) in original graph, add edges u_out -> v_in and v_out -> u_in with capacity inf
        # - For each pair (i, j) of branches, we need a path from i to j.
        # - This is a multi-commodity flow problem, which is hard.
        
        # However, since all paths must be vertex-disjoint, we can use a single max-flow if we
        # can reduce it. But multi-commodity flow is NP-hard in general.
        
        # Let's stick with the greedy approach but improve it.
        # Or, we can try to use a different strategy: 
        # 1. Find a large clique in the graph.
        # 2. Check if the clique has vertex-disjoint paths (which it does, since adjacent vertices have path of length 1).
        # 3. Try to expand the clique by adding vertices that are "close" to existing branches.
        
        # For now, let's implement a better greedy with backtracking.
        return None

    def try_branch_set(branches):
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        import random
        rng = random.Random(42)

        def find_paths_greedy(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths

            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                forbidden = used_int | (branch_set - {a, b})
                path = bfs_path(a, b, forbidden, adj)
                if path is not None:
                    interior = set(path[1:-1])
                    if not (interior & used_int):
                        new_used = used_int | interior
                        result = find_paths_greedy(rest, new_used, current_paths + [path])
                        if result is not None:
                            return result
            return None

        # Try natural order with greedy
        result = find_paths_greedy(pairs, set(), [])
        if result is not None:
            return result

        # Try sorted by "difficulty" (shorter available paths first)
        def pair_difficulty(pair):
            a, b = pair
            forbidden = branch_set - {a, b}
            path = bfs_path(a, b, forbidden, adj)
            return len(path) if path else 999

        sorted_pairs = sorted(pairs, key=pair_difficulty)
        result = find_paths_greedy(sorted_pairs, set(), [])
        if result is not None:
            return result

        # Try randomized orderings
        for attempt in range(10):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths_greedy(shuffled, set(), [])
            if result is not None:
                return result

        return None
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Find initial clique
    def find_max_clique_greedy(adj, n):
        best = []
        for start in range(n):
            if time.time() > deadline:
                break
            clique = [start]
            candidates = set(adj[start])
            while candidates:
                best_v = max(candidates, key=lambda v: len(adj[v] & set(clique)))
                if len(adj[best_v] & set(clique)) < len(clique):
                    break
                clique.append(best_v)
                candidates = candidates & adj[best_v]
                candidates = candidates - set(clique)
            if len(clique) > len(best):
                best = clique[:]
        return best

    initial_clique = find_max_clique_greedy(adj, n)
    paths_result = try_branch_set(initial_clique)
    if paths_result is not None:
        best_branches = initial_clique[:]
        best_paths = paths_result

    # Greedy expansion
    if best_branches is not None:
        current_branches = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            current_set = set(current_branches)
            candidates = [v for v in range(n) if v not in current_set]
            candidates.sort(key=lambda v: len(adj[v]), reverse=True)

            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                paths_result = try_branch_set(new_branches)
                if paths_result is not None:
                    current_branches = new_branches
                    best_branches = current_branches[:]
                    best_paths = paths_result
                    expanded = True
                    break

    # Random search for larger sets
    import random
    rng2 = random.Random(123)

    for size in range(n, 1, -1):
        if time.time() > deadline:
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            random_set = rng2.sample(range(n), size)
            paths_result = try_branch_set(random_set)
            if paths_result is not None:
                best_branches = random_set[:]
                best_paths = paths_result
                break

    # Local search: try swapping vertices in the best set
    if best_branches is not None and time.time() < deadline:
        current_branches = best_branches[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current_branches)
            for i in range(len(current_branches)):
                if time.time() > deadline:
                    break
                for v in range(n):
                    if v in current_set:
                        continue
                    # Try swapping current_branches[i] with v
                    new_branches = current_branches[:i] + [v] + current_branches[i+1:]
                    paths_result = try_branch_set(new_branches)
                    if paths_result is not None:
                        current_branches = new_branches
                        best_branches = current_branches[:]
                        best_paths = paths_result
                        improved = True
                        break
                if improved:
                    break
=======
    # Find initial clique using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P, X, best_clique):
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique.extend(R)
            return
        if time.time() > deadline:
            return
        
        # Choose pivot to minimize branching
        # Pivot should be in P ∪ X
        pivot_candidates = P | X
        if pivot_candidates:
            # Choose vertex with max degree in P
            u = max(pivot_candidates, key=lambda v: len(adj[v] & P))
            candidates = P - adj[u]
        else:
            candidates = P
        
        for v in list(candidates):
            if time.time() > deadline:
                return
            bron_kerbosch(R + [v], P & adj[v], X & adj[v], best_clique)
            P.remove(v)
            X.add(v)
    
    best_clique = []
    bron_kerbosch([], set(range(n)), set(), best_clique)
    initial_clique = best_clique[:]
    
    if not initial_clique or len(initial_clique) < 2:
        initial_clique = [0, 1]
    
    paths_result = try_branch_set(initial_clique)
    if paths_result is not None:
        best_branches = initial_clique[:]
        best_paths = paths_result

    # Greedy expansion
    if best_branches is not None:
        current_branches = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            current_set = set(current_branches)
            candidates = [v for v in range(n) if v not in current_set]
            # Sort by number of connections to current branches (descending)
            candidates.sort(key=lambda v: len(adj[v] & current_set), reverse=True)
            candidates.sort(key=lambda v: len(adj[v]), reverse=True)

            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                paths_result = try_branch_set(new_branches)
                if paths_result is not None:
                    current_branches = new_branches
                    best_branches = current_branches[:]
                    best_paths = paths_result
                    expanded = True
                    break

    # Local search: try swapping vertices in the best set
    if best_branches is not None and time.time() < deadline:
        current_branches = best_branches[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current_branches)
            for i in range(len(current_branches)):
                if time.time() > deadline:
                    break
                # Sort candidates by how well they connect to the rest of the set
                other_branches = set(current_branches) - {current_branches[i]}
                candidates = [v for v in range(n) if v not in current_set]
                candidates.sort(key=lambda v: len(adj[v] & other_branches), reverse=True)
                
                for v in candidates:
                    new_branches = current_branches[:i] + [v] + current_branches[i+1:]
                    paths_result = try_branch_set(new_branches)
                    if paths_result is not None:
                        current_branches = new_branches
                        best_branches = current_branches[:]
                        best_paths = paths_result
                        improved = True
                        break
                if improved:
                    break
>>>>>>> REPLACE
