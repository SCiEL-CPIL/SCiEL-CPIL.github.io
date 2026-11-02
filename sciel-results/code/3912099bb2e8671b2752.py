"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)
    N1 = len(vectors_v1)
    N = N4 + N1

    if N4 == 0:
        return vectors_v1

    # Build combined vector matrix
    all_vectors = np.vstack([vectors_v4, vectors_v1])

    # Compute conflict matrix for the combined set: conflict if dot > 2
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Partition by parity of sum of coordinates
    sums = all_vectors.sum(axis=1)
    parity = sums % 2  # 0 or 1

    # For each parity class, check if the induced conflict graph is bipartite
    # If bipartite, use Kőnig's theorem: max independent set = n - min vertex cover
    # min vertex cover = max matching in bipartite graph

    def max_bipartite_matching(adj_matrix, n_left, n_right):
        """
        Find maximum matching in a bipartite graph using Hopcroft-Karp.
        adj_matrix: (n_left x n_right) boolean array
        Returns: (left_match, right_match) arrays where
                 left_match[i] = matched right node or -1
                 right_match[j] = matched left node or -1
        """
        nL = n_left
        nR = n_right
        left_match = np.full(nL, -1, dtype=np.int64)
        right_match = np.full(nR, -1, dtype=np.int64)

        # BFS to find augmenting paths
        def bfs():
            dist = np.full(nL, -1, dtype=np.int32)
            queue = deque()
            for u in range(nL):
                if left_match[u] == -1:
                    dist[u] = 0
                    queue.append(u)
            found = False
            while queue:
                u = queue.popleft()
                for v in np.where(adj_matrix[u])[0]:
                    w = right_match[v]
                    if w == -1:
                        found = True
                    elif dist[w] == -1:
                        dist[w] = dist[u] + 1
                        queue.append(w)
            return dist, found

        def dfs(u, dist, visited):
            for v in np.where(adj_matrix[u])[0]:
                w = right_match[v]
                if w == -1 or (dist[w] == dist[u] + 1 and not visited[w] and dfs(w, dist, visited)):
                    left_match[u] = v
                    right_match[v] = u
                    return True
            dist[u] = -1
            return False

        while time.time() < deadline - 3:
            dist, found = bfs()
            if not found:
                break
            visited = np.zeros(nR, dtype=bool)
            for u in range(nL):
                if left_match[u] == -1:
                    dfs(u, dist, visited)

        return left_match, right_match

    def solve_bipartite_mis(indices, adj_submatrix):
        """
        Given indices into the global graph and the sub-adjacency matrix,
        find maximum independent set using bipartite matching.
        Returns: boolean array of length len(indices) indicating MIS membership.
        """
        n = len(indices)
        if n == 0:
            return np.array([], dtype=bool)

        # Try to 2-color the graph
        color = np.full(n, -1, dtype=np.int32)
        left_nodes = []
        right_nodes = []
        is_bipartite = True

        for i in range(n):
            if color[i] != -1:
                continue
            color[i] = 0
            left_nodes.append(i)
            queue = deque([i])
            while queue and is_bipartite:
                u = queue.popleft()
                for v in np.where(adj_submatrix[u])[0]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        if color[v] == 0:
                            left_nodes.append(v)
                        else:
                            right_nodes.append(v)
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break

        if not is_bipartite:
            # Fall back to greedy
            in_set = np.zeros(n, dtype=bool)
            alive = np.ones(n, dtype=bool)
            degrees = adj_submatrix.sum(axis=1).astype(np.float64)
            while np.any(alive):
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
                chosen = cands[0]
                in_set[chosen] = True
                nbrs = np.where(adj_submatrix[chosen])[0]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= adj_submatrix[rem][:, to_remove].sum(axis=1)
            return in_set

        # Build bipartite adjacency: left_nodes x right_nodes
        left_idx_map = {node: i for i, node in enumerate(left_nodes)}
        right_idx_map = {node: i for i, node in enumerate(right_nodes)}

        nL = len(left_nodes)
        nR = len(right_nodes)
        bip_adj = np.zeros((nL, nR), dtype=bool)
        for i, u in enumerate(left_nodes):
            for v in np.where(adj_submatrix[u])[0]:
                if v in right_idx_map:
                    bip_adj[i, right_idx_map[v]] = True

        # Find maximum matching
        left_match, right_match = max_bipartite_matching(bip_adj, nL, nR)

        # Minimum vertex cover from matching (Kőnig's theorem)
        # Unmatched left nodes are in the cover
        # Then do DFS on alternating paths from unmatched left nodes
        unmatched_left = [i for i in range(nL) if left_match[i] == -1]

        # Build residual graph for alternating paths
        # From left to right: unmatched edges
        # From right to left: matched edges
        in_cover_left = np.zeros(nL, dtype=bool)
        in_cover_right = np.zeros(nR, dtype=bool)

        # Mark unmatched left nodes
        for i in unmatched_left:
            in_cover_left[i] = True

        # BFS/DFS on alternating paths
        visited_left = np.zeros(nL, dtype=bool)
        visited_right = np.zeros(nR, dtype=bool)
        queue = deque()
        for i in unmatched_left:
            visited_left[i] = True
            queue.append(('L', i))

        while queue:
            side, idx = queue.popleft()
            if side == 'L':
                # From left node, go along unmatched edges to right
                for j in np.where(bip_adj[idx])[0]:
                    if left_match[idx] != j:  # unmatched edge
                        if not visited_right[j]:
                            visited_right[j] = True
                            in_cover_right[j] = True
                            queue.append(('R', j))
            else:
                # From right node, go along matched edge to left
                i = right_match[idx]
                if i != -1 and not visited_left[i]:
                    visited_left[i] = True
                    in_cover_left[i] = True
                    queue.append(('L', i))

        # Maximum independent set = complement of minimum vertex cover
        mis_in_subgraph = np.zeros(n, dtype=bool)
        for i, node in enumerate(left_nodes):
            if not in_cover_left[i]:
                mis_in_subgraph[node] = True
        for j, node in enumerate(right_nodes):
            if not in_cover_right[j]:
                mis_in_subgraph[node] = True

        return mis_in_subgraph

    # Solve for each parity class
    best_in_set = np.zeros(N, dtype=bool)

    for p in [0, 1]:
        class_indices = np.where(parity == p)[0]
        n_class = len(class_indices)
        if n_class == 0:
            continue

        # Build sub-adjacency matrix for this class
        sub_adj = conflicts[np.ix_(class_indices, class_indices)]

        # Solve MIS for this class
        mis_local = solve_bipartite_mis(class_indices, sub_adj)
        best_in_set[class_indices[mis_local]] = True

    # Also try a global greedy as fallback/comparison
    rng = np.random.default_rng(42)

    def greedy_min_degree_global():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        neighbors_list = [np.where(conflicts[i])[0] for i in range(N)]
        neighbor_counts = np.array([len(n) for n in neighbors_list], dtype=np.int32)
        degrees = neighbor_counts.copy().astype(np.float64)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors_list[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    greedy_result = greedy_min_degree_global()
    if greedy_result.sum() > best_in_set.sum():
        best_in_set = greedy_result

    # Polish: add any vertices that don't conflict with the current set
    def fast_greedy_add(in_set_init, max_time=5.0):
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time

        while time.time() < polish_deadline:
            not_in_mask = ~polish_set
            if not np.any(not_in_mask):
                break
            not_in_indices = np.where(not_in_mask)[0]
            counts = conflicts[not_in_indices] @ polish_set
            addable_mask = counts == 0
            if not np.any(addable_mask):
                break
            add_indices = not_in_indices[addable_mask]
            polish_set[add_indices] = True
            polish_size += len(add_indices)

        return polish_set, polish_size

    polished, polished_size = fast_greedy_add(best_in_set, max_time=2.0)
    if polished_size > best_in_set.sum():
        best_in_set = polished

    # Final result
    final_indices = np.where(best_in_set)[0]
    final_vectors = all_vectors[final_indices]

    return final_vectors

# EVOLVE-BLOCK-END
