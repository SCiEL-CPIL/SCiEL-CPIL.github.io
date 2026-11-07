"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    degree = [len(adj[v]) for v in range(n)]

    # Bron-Kerbosch for maximum clique
    max_clique = []

    def bronk(R, P_set, X_set):
        nonlocal max_clique
        if time.time() > deadline - 10:
            return
        if not P_set and not X_set:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        U = P_set | X_set
        if not U:
            return
        pivot = None
        max_inter = -1
        for u in U:
            inter = len(P_set & adj[u])
            if inter > max_inter:
                max_inter = inter
                pivot = u
        candidates = list(P_set - adj[pivot])
        for v in candidates:
            bronk(R | {v}, P_set & adj[v], X_set & adj[v])
            P_set.discard(v)
            X_set.add(v)
            if time.time() > deadline - 10:
                return

    all_vertices = set(range(n))
    bronk(set(), all_vertices, set())

    if len(max_clique) < 2:
        max_clique = [edges[0][0], edges[0][1]]

    # Core: find vertex-disjoint interior paths for all pairs in branch set
    def find_paths(branches, max_path_len=8):
        """Find vertex-disjoint interior paths for all unordered pairs.
        Returns dict {(i,j): path} or None."""
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(range(k), 2))
        branch_set = set(branches)
        num_pairs = len(pairs)

        # For each pair, precompute candidate paths (shortest first)
        def bfs_paths(src, dst, forbidden, max_len):
            """Find up to 15 candidate paths from src to dst avoiding forbidden."""
            if src == dst:
                return [[src]]
            paths = []
            visited = {src}
            queue = deque([(src, [src])])
            while queue:
                node, path = queue.popleft()
                if len(path) > max_len:
                    continue
                if node == dst:
                    paths.append(list(path))
                    if len(paths) >= 15:
                        break
                    continue
                for nb in sorted(adj[node]):
                    if nb in forbidden or nb in visited:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
            paths.sort(key=len)
            return paths[:15]

        # Precompute candidates for each pair
        candidates = {}
        for idx, (i, j) in enumerate(pairs):
            bi, bj = branches[i], branches[j]
            forbidden = branch_set - {bi, bj}
            cands = bfs_paths(bi, bj, forbidden, max_path_len)
            if not cands:
                return None
            candidates[idx] = cands

        # Order pairs by number of candidates (most constrained first)
        order = sorted(range(num_pairs), key=lambda idx: len(candidates[idx]))

        assignment = [None] * num_pairs
        used_interior = set()

        def backtrack(pos):
            if time.time() > deadline - 5:
                return False
            if pos == num_pairs:
                return True
            pair_idx = order[pos]
            i, j = pairs[pair_idx]
            bi, bj = branches[i], branches[j]
            for path in candidates[pair_idx]:
                interior = set(path[1:-1])
                if interior & used_interior:
                    continue
                assignment[pair_idx] = path
                used_interior |= interior
                if backtrack(pos + 1):
                    return True
                used_interior -= interior
                assignment[pair_idx] = None
            return False

        if backtrack(0):
            result = {}
            for idx in range(num_pairs):
                result[idx] = assignment[idx]
            return result
        return None

    # Greedy extension: start from a seed, try adding vertices one by one
    def greedy_extend(seed_branches, candidate_pool, max_attempts=500):
        current = list(seed_branches)
        paths = find_paths(current)
        if paths is None:
            return None, None

        best = (list(current), paths)
        attempts = 0
        improved = True
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            if time.time() > deadline - 15:
                break
            for v in candidate_pool:
                if v in current:
                    continue
                if time.time() > deadline - 10:
                    break
                new_branches = current + [v]
                new_paths = find_paths(new_branches)
                if new_paths is not None:
                    current = new_branches
                    best = (list(current), new_paths)
                    improved = True
                    break
        return best

    # Main search
    best_branches = list(max_clique)
    best_paths = find_paths(best_branches)
    best_size = len(best_branches)

    if best_paths is None:
        # Fallback
        best_paths = {}
        pairs = list(combinations(range(len(best_branches)), 2))
        for idx, (i, j) in enumerate(pairs):
            bi, bj = best_branches[i], best_branches[j]
            best_paths[idx] = [bi, bj]

    candidate_pool = sorted(range(n), key=lambda v: degree[v], reverse=True)

    # Strategy 1: Greedy from max clique
    result = greedy_extend(best_branches, candidate_pool)
    if result[0] is not None and len(result[0]) > best_size:
        best_branches, best_paths = result
        best_size = len(best_branches)

    # Strategy 2: Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        seed_size = random.randint(2, 4)
        seed_vertices = random.sample(candidate_pool[:12], min(seed_size, 12))
        result = greedy_extend(list(seed_vertices), candidate_pool)
        if result[0] is not None and len(result[0]) > best_size:
            best_branches, best_paths = result
            best_size = len(result[0])

    # Strategy 3: Local search - swap
    if time.time() < deadline - 25:
        for _ in range(8):
            if time.time() > deadline - 12:
                break
            cur = list(best_branches)
            improved = False
            for bi in range(len(cur)):
                if time.time() > deadline - 10:
                    break
                old_b = cur[bi]
                for v in candidate_pool:
                    if v in cur:
                        continue
                    new_branches = cur[:bi] + [v] + cur[bi+1:]
                    new_paths = find_paths(new_branches)
                    if new_paths is not None:
                        best_branches = new_branches
                        best_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # Convert to output format
    if best_paths is None:
        u, v = edges[0]
        best_branches = [int(u), int(v)]
        best_paths = {(0, 1): [int(u), int(v)]}

    pairs = list(combinations(range(len(best_branches)), 2))
    paths_out = []
    for idx, (i, j) in enumerate(pairs):
        path = best_paths.get(idx)
        if path is None:
            # Fallback: direct edge or shortest path
            bi, bj = best_branches[i], best_branches[j]
            if (bi, bj) in edge_set:
                path = [bi, bj]
            else:
                path = [bi, bj]
        paths_out.append([int(x) for x in path])

    branches_out = [int(b) for b in best_branches]

    if len(branches_out) < 2:
        u, v = edges[0]
        branches_out = [int(u), int(v)]
        paths_out = [[int(u), int(v)]]

    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
