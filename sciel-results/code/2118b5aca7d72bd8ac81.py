import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # ---------- Max-flow verification with vertex splitting ----------
    # For a candidate branch set B, check if internally disjoint paths exist
    # for all pairs in B. We model this as: each non-branch vertex has capacity 1.
    # We use a single-commodity reduction: for each pair (u,v) in B, we need
    # a path using distinct interior vertices. We verify by trying to route
    # all paths simultaneously using a flow network.

    def verify_and_extract(branches_list):
        """Verify internally disjoint paths for all pairs in branches_list.
        Returns (True, paths_dict) or (False, None)."""
        k = len(branches_list)
        if k < 2:
            return False, None
        branches_set = set(branches_list)
        pairs = list(combinations(range(k), 2))

        # Build a flow network with vertex splitting for non-branch vertices
        # Each non-branch vertex v becomes v_in -> v_out with capacity 1
        # Branch vertices have infinite capacity
        # For each pair (i,j), we add a super source connected to branch_i
        # and super sink from branch_j, but we need ALL pairs simultaneously.

        # Strategy: Use iterative path finding with max-flow for feasibility
        # First, check if the graph minus branches has enough connectivity
        # Then greedily assign paths using max-flow for each pair

        # Simpler approach: greedy path assignment with max-flow verification
        # For each pair, find a path that doesn't use already-assigned interior vertices
        # Use BFS on the residual graph

        assigned_interior = set()
        paths_dict = {}

        # Sort pairs by "difficulty" - pairs with fewer available paths first
        pair_order = []
        for i, j in pairs:
            s, t = branches_list[i], branches_list[j]
            # Count available interior vertices that are adjacent to both s and t
            avail = 0
            for w in adj[s]:
                if w not in branches_set and w != t and t in adj[w] and w not in assigned_interior:
                    avail += 1
            # Also count direct edge
            if t in adj[s]:
                avail += 1
            pair_order.append((avail, i, j))
        pair_order.sort(key=lambda x: x[0])

        for _, i, j in pair_order:
            s, t = branches_list[i], branches_list[j]
            forbidden = assigned_interior | (branches_set - {s, t})

            # Try to find a path from s to t avoiding forbidden interior vertices
            # Use BFS with the constraint that interior vertices must not be in forbidden
            path = _bfs_path(s, t, forbidden, adj)
            if path is None:
                return False, None

            interior = set(path[1:-1])
            if interior & assigned_interior:
                return False, None
            assigned_interior |= interior
            paths_dict[(i, j)] = path

        return True, paths_dict

    def _bfs_path(s, t, forbidden, adj):
        """BFS from s to t, interior vertices must not be in forbidden."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == t:
                    # Found path
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nb in forbidden:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    # ---------- Greedy construction ----------
    # Start with a good seed, then try to add more branches

    # Score each vertex by how well it connects to others
    # A vertex is "good" if it has high degree and connects to many other high-degree vertices
    vertex_scores = []
    for v in range(n):
        score = degrees[v]
        # Bonus for connecting to other high-degree vertices
        for nb in adj[v]:
            score += 0.5 * degrees[nb]
        vertex_scores.append((score, v))
    vertex_scores.sort(key=lambda x: x[0], reverse=True)

    # Try multiple starting pairs and grow
    best_valid = None
    best_valid_size = 0

    # Try growing from top-scored vertices
    candidate_order = [v for _, v in vertex_scores]

    for start_idx in range(min(5, n)):
        if time.time() > deadline - 10:
            break

        # Start with two top vertices
        init_branches = [candidate_order[start_idx]]
        for other in candidate_order:
            if other == init_branches[0]:
                continue
            # Check if there's a path between them
            forbidden = set()
            path = _bfs_path(init_branches[0], other, forbidden, adj)
            if path is not None:
                init_branches.append(other)
                break

        if len(init_branches) < 2:
            continue

        # Try to grow by adding vertices
        current = list(init_branches)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for v in candidate_order:
                if v in current:
                    continue
                new_branches = current + [v]
                valid, paths = verify_and_extract(new_branches)
                if valid:
                    current = new_branches
                    if len(current) > best_valid_size:
                        best_valid_size = len(current)
                        best_valid = (list(current), paths)
                    improved = True
                    break

        # Try to improve by swapping
        if time.time() < deadline - 10:
            current = best_valid[0] if best_valid else current
            swap_improved = True
            while swap_improved and time.time() < deadline - 10:
                swap_improved = False
                for i in range(len(current)):
                    for v in candidate_order:
                        if v in current:
                            continue
                        new_branches = current[:i] + current[i+1:] + [v]
                        valid, paths = verify_and_extract(new_branches)
                        if valid:
                            current = new_branches
                            if len(current) > best_valid_size:
                                best_valid_size = len(current)
                                best_valid = (list(current), paths)
                            swap_improved = True
                            break
                    if swap_improved:
                        break

    # ---------- Fallback: try all pairs ----------
    if best_valid is None or best_valid_size < 2:
        for u, v in edges:
            if u != v:
                best_valid = ([u, v], {(0, 1): [u, v]})
                best_valid_size = 2
                break

    if best_valid is None:
        u, v = edges[0]
        best_valid = ([u, v], {(0, 1): [u, v]})
        best_valid_size = 2

    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            p = paths_dict.get((i, j)) or paths_dict.get((j, i))
            if p is None:
                p = [branches[i], branches[j]]
            if p[0] != branches[i]:
                p = p[::-1]
            output_paths.append(list(p))

    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
