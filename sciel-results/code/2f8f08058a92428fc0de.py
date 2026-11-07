import time
import itertools
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()
    deadline = start_time + 85.0

    def time_left():
        return deadline - time.time()

    # Build adjacency as bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)

    def find_max_clique():
        """Bron-Kerbosch with pivot to find largest maximal clique."""
        max_clique = []
        max_clique_id = 0

        def bronk(R, P, X):
            nonlocal max_clique, max_clique_id
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                    max_clique_id += 1
                return

            if P or X:
                # Choose pivot with max degree into P
                pivot = max(P | X, key=lambda v: bin(adj_mask[v] & P).count('1'))
            else:
                return

            candidates = [v for v in P if not (adj_mask[pivot] & (1 << v))]
            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bronk(new_R, new_P, new_X)
                P = P - {v}
                X = X | {v}

        bronk(frozenset(), frozenset(range(n)), frozenset())
        return max_clique

    def check_disjoint_paths(branches, max_path_len=6, max_paths_per_pair=20):
        """
        Check if 'branches' can be connected by internally disjoint paths.
        Returns list of paths if successful, else None.
        """
        if len(branches) < 2:
            return []

        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))

        # For each pair, find up to max_paths_per_pair short simple paths
        candidates = {}

        for u, v in pairs:
            paths = []
            # BFS to find shortest paths first
            # Use a queue of (path, visited)
            queue = deque()
            queue.append(([u], {u}))
            found = 0
            while queue and found < max_paths_per_pair:
                path, visited = queue.popleft()
                curr = path[-1]
                if len(path) - 1 > max_path_len:
                    continue
                if curr == v and len(path) > 1:
                    paths.append(list(path))
                    found += 1
                    continue
                if len(path) - 1 >= max_path_len:
                    continue
                for nb in sorted(adj[curr]):
                    if nb == v:
                        if len(path) + 1 - 1 <= max_path_len:
                            paths.append(path + [nb])
                            found += 1
                        continue
                    if nb not in visited and nb not in branch_set:
                        queue.append(path + [nb], visited | {nb})
                        if found >= max_paths_per_pair:
                            break

            paths.sort(key=lambda x: (len(x), x))
            candidates[(u, v)] = paths[:max_paths_per_pair]

        # Check if any pair has no candidates
        for u, v in pairs:
            if not candidates[(u, v)]:
                return None

        # Order pairs by number of candidates (most constrained first)
        pair_order = sorted(pairs, key=lambda p: len(candidates[p]))

        used_internal = set()
        assignment = {}

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pair_order):
                return True

            u, v = pair_order[idx]
            for path in candidates[(u, v)]:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & branch_set:
                    continue

                assignment[(u, v)] = path
                used_internal.update(internal)

                if backtrack(idx + 1):
                    return True

                used_internal.difference_update(internal)
                del assignment[(u, v)]

            return False

        if backtrack(0):
            return [assignment[p] for p in pairs]

        return None

    # Phase 1: Find Max Clique
    max_clique = find_max_clique()
    best_size = len(max_clique)
    best_branches = list(max_clique)
    best_paths = None

    # If max clique is large enough, it's a valid subdivision
    if best_size >= 2:
        pairs = list(itertools.combinations(best_branches, 2))
        best_paths = [[u, v] for u, v in pairs]

    # Phase 2: Try to expand
    current_branches = list(best_branches)
    current_paths = best_paths

    # Try adding vertices one by one
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break

        added = False
        # Candidates to add: vertices not in current_branches
        # Sort by degree to prefer well-connected vertices
        candidates = [v for v in range(n) if v not in set(current_branches)]
        candidates.sort(key=lambda v: -len(adj[v]))

        for v in candidates:
            if time.time() > deadline - 10:
                break

            new_branches = current_branches + [v]

            # Determine max_path_len based on size
            # Smaller max_path_len is faster
            mpl = 6 if len(new_branches) <= 6 else 4

            paths = check_disjoint_paths(new_branches, max_path_len=mpl, max_paths_per_pair=10)

            if paths is not None:
                current_branches = new_branches
                best_size = len(current_branches)
                best_branches = list(current_branches)
                best_paths = paths
                current_paths = paths
                added = True
                break  # Restart candidate selection with new set

        if not added:
            # Try swapping: replace one branch with another
            # This is more complex; for now, break
            break

    # Final validation and formatting
    if best_size >= 2 and best_paths is not None:
        branches = [int(b) for b in best_branches]
        formatted_paths = []
        for p in best_paths:
            formatted_paths.append([int(x) for x in p])

        return {'branches': branches, 'paths': formatted_paths}

    # Fallback: Any edge
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}
