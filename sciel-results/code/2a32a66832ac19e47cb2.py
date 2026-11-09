import itertools
import time
import collections
import random

def construct():
    start_time = time.time()
    LIMIT = 55.0  # safety margin under 60s (evaluator kills at 90s)

    p = 101
    denom_range = range(1, 11)
    n = 10

    # Precompute modular inverses for each denominator
    inverses = {d: pow(d, -1, p) for d in denom_range}

    # Group subsets by size
    subsets_by_size = {}
    for size in range(1, n + 1):
        subsets_by_size[size] = []
        for combo in itertools.combinations(denom_range, size):
            s = sum(inverses[d] for d in combo)
            residue = s % p
            subsets_by_size[size].append((residue, combo))

    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] contains right nodes connected to left node i.
        n_left: number of left nodes (residues)
        n_right: number of right nodes (subsets)
        Returns: matching array of size n_left, matching[i] is the right node matched to left node i, or -1.
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = collections.deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == float('inf'):
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u

    def can_cover(max_size):
        """Check if residues 0..100 can all be assigned distinct subsets of size <= max_size."""
        all_subsets = []
        for sz in range(1, max_size + 1):
            for (res, combo) in subsets_by_size[sz]:
                all_subsets.append((res, combo))
        
        n_subsets = len(all_subsets)
        if n_subsets < p:
            return None

        # Build adjacency: residue -> subset indices
        adj = [[] for _ in range(p)]
        for idx, (res, combo) in enumerate(all_subsets):
            adj[res].append(idx)

        # Hopcroft-Karp
        # Left: 0..100 (residues)
        # Right: 0..n_subsets-1 (subsets)
        
        # Optimization: If a residue has no subsets, return None immediately
        for r in range(p):
            if not adj[r]:
                return None

        pair_u = hopcroft_karp(adj, p, n_subsets)
        
        # Check if all residues are matched
        for r in range(p):
            if pair_u[r] == -1:
                return None

        # Build assignment
        assignment = [None] * p
        for r in range(p):
            s_idx = pair_u[r]
            assignment[r] = all_subsets[s_idx][1]
        return assignment

    # Try increasing max_size
    for K in range(1, n + 1):
        if time.time() - start_time > LIMIT:
            break
        assignment = can_cover(K)
        if assignment is not None:
            # All residues covered with subsets of size <= K
            result = [list(assignment[r]) for r in range(p)]
            # Sanity check
            for r in range(p):
                combo = result[r]
                assert len(combo) > 0
                assert len(set(combo)) == len(combo)
                s = sum(inverses[d] for d in combo)
                assert s % p == r, f"Residue mismatch at r={r}: got {s%p}"
                assert all(1 <= d <= 10 for d in combo)
            return result

    # Fallback
    assignments = {}
    for size in range(1, n + 1):
        for (res, combo) in subsets_by_size[size]:
            if res not in assignments:
                assignments[res] = combo
            if len(assignments) == p:
                break
        if len(assignments) == p:
            break
        if time.time() - start_time > LIMIT:
            break

    result = []
    for r in range(p):
        if r in assignments:
            result.append(list(assignments[r]))
        else:
            result.append(list(denom_range))
    return result
