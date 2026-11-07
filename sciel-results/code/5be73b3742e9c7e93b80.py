"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0

    n = P['n']

    # Build initial adjacency from path
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True

    def bfs_all_pairs(adj_mat):
        dist_matrix = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj_mat[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            dist_matrix.append(dist)
        return dist_matrix

    def get_diameter(dist_matrix):
        max_diam = 0
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][j] > max_diam:
                    max_diam = dist_matrix[i][j]
        return max_diam

    def is_triangle_free(adj_mat, u, v):
        for w in range(n):
            if w != u and w != v and adj_mat[u][w] and adj_mat[v][w]:
                return False
        return True

    def check_valid(added_edges):
        if not added_edges:
            return get_diameter(bfs_all_pairs(adj)) <= 4

        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if adj[a][b]:
                return False

        tmp_adj = [row[:] for row in adj]
        for u, v in added_edges:
            tmp_adj[u][v] = True
            tmp_adj[v][u] = True

        for u in range(n):
            for v in range(u + 1, n):
                if tmp_adj[u][v]:
                    for w in range(n):
                        if w != u and w != v and tmp_adj[u][w] and tmp_adj[v][w]:
                            return False

        dist_matrix = bfs_all_pairs(tmp_adj)
        return get_diameter(dist_matrix) <= 4

    # Generate all candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v] and is_triangle_free(adj, u, v):
                candidates.append((u, v))

    m = len(candidates)

    base_dist = bfs_all_pairs(adj)
    pairs_needing = []
    for i in range(n):
        for j in range(i + 1, n):
            if base_dist[i][j] > 4:
                pairs_needing.append((i, j))

    if not pairs_needing:
        return []

    # For each candidate edge, compute which pairs it covers
    edge_covers = {}
    for idx, (u, v) in enumerate(candidates):
        covers = set()
        for (i, j) in pairs_needing:
            d1 = base_dist[i][u] + 1 + base_dist[v][j] if base_dist[i][u] != -1 and base_dist[v][j] != -1 else float('inf')
            d2 = base_dist[i][v] + 1 + base_dist[u][j] if base_dist[i][v] != -1 and base_dist[u][j] != -1 else float('inf')
            if min(base_dist[i][j], d1, d2) <= 4:
                covers.add((i, j))
        edge_covers[idx] = covers

    # Backtracking search
    def backtrack_search(k, cand_list, cand_indices, used_set, current_edges, uncovered):
        if len(current_edges) == k:
            return current_edges if not uncovered else None

        if time.time() > deadline - 2:
            return None

        remaining_needed = k - len(current_edges)
        remaining_cands = len(cand_indices) - len(used_set)
        if remaining_cands < remaining_needed:
            return None

        for pos in range(len(cand_indices)):
            idx = cand_indices[pos]
            if idx in used_set:
                continue
            if time.time() > deadline - 2:
                return None

            u, v = cand_list[idx]

            # Check triangle-free with current edges
            tri_ok = True
            for (su, sv) in current_edges:
                for w in range(n):
                    if w != u and w != v:
                        if adj[u][w] and adj[v][w]:
                            tri_ok = False
                            break
                        has_uw = (min(u,w), max(u,w)) in used_set or adj[u][w]
                        has_vw = (min(v,w), max(v,w)) in used_set or adj[v][w]
                        if has_uw and has_vw:
                            tri_ok = False
                            break
                if not tri_ok:
                    break

            if not tri_ok:
                continue

            new_uncovered = uncovered - edge_covers[idx]
            new_used = used_set | {idx}
            new_edges = current_edges + [[u, v]]

            result = backtrack_search(k, cand_list, cand_indices, new_used, new_edges, new_uncovered)
            if result is not None:
                return result

        return None

    best_solution = None
    for k in range(5, 9):
        if time.time() > deadline - 5:
            break

        useful = [(idx, len(edge_covers[idx])) for idx in range(m) if edge_covers[idx]]
        if len(useful) < k:
            continue

        useful.sort(key=lambda x: -x[1])
        cand_indices = [x[0] for x in useful]

        result = backtrack_search(k, candidates, cand_indices, set(), [], set(pairs_needing))
        if result is not None:
            if check_valid(result):
                best_solution = result
                break

    if best_solution is not None:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(best_solution)):
                trial = best_solution[:i] + best_solution[i+1:]
                if check_valid(trial):
                    best_solution = trial
                    improved = True
                    break
        return best_solution

    # Fallback: hardcoded
    hardcoded_edges = [
        [12, 1], [12, 3], [12, 6], [12, 9],
        [12, 15], [12, 18], [12, 21], [12, 23]
    ]
    if check_valid(hardcoded_edges):
        return hardcoded_edges

    # Greedy fallback
    current_adj = [row[:] for row in adj]
    current_dist = bfs_all_pairs(current_adj)
    uncovered = set(pairs_needing)
    selected = []
    selected_set = set()

    while uncovered and time.time() < deadline - 3:
        best_edge = None
        best_count = 0
        for idx, (u, v) in enumerate(candidates):
            key = (min(u,v), max(u,v))
            if key in selected_set:
                continue
            if not is_triangle_free(current_adj, u, v):
                continue
            count = 0
            for (i, j) in uncovered:
                d1 = current_dist[i][u] + 1 + current_dist[v][j] if current_dist[i][u] != -1 and current_dist[v][j] != -1 else float('inf')
                d2 = current_dist[i][v] + 1 + current_dist[u][j] if current_dist[i][v] != -1 and current_dist[u][j] != -1 else float('inf')
                if min(current_dist[i][j], d1, d2) <= 4:
                    count += 1
            if count > best_count:
                best_count = count
                best_edge = (u, v, idx)

        if best_edge is None or best_count == 0:
            break

        u, v, idx = best_edge
        selected.append([u, v])
        selected_set.add((min(u,v), max(u,v)))
        current_adj[u][v] = True
        current_adj[v][u] = True
        current_dist = bfs_all_pairs(current_adj)
        uncovered = set()
        for i in range(n):
            for j in range(i + 1, n):
                if current_dist[i][j] > 4:
                    uncovered.add((i, j))

    if not uncovered and check_valid(selected):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(selected)):
                trial = selected[:i] + selected[i+1:]
                if check_valid(trial):
                    selected = trial
                    improved = True
                    break
        return selected

    if check_valid(hardcoded_edges):
        return hardcoded_edges

    return []

# EVOLVE-BLOCK-END
