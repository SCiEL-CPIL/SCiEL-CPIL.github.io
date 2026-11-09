import time
import random
from collections import deque
from itertools import combinations
import networkx as nx

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # ---------- Bron-Kerbosch with pivot to find max clique ----------
    best_clique = []

    def bron_kerbosch(R, Pset, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not Pset and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Pivot
        candidates_for_pivot = list(Pset) + list(X)
        u = None
        max_neighbors = -1
        for v in candidates_for_pivot:
            count = len(adj[v] & Pset)
            if count > max_neighbors:
                max_neighbors = count
                u = v
        if u is None:
            candidates = list(Pset)
        else:
            candidates = [v for v in Pset if v not in adj[u]]
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        for v in candidates:
            if time.time() > deadline:
                return
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v])
            Pset = Pset - {v}
            X = X | {v}

    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset())

    # Greedy + random restarts to find good cliques
    def greedy_clique(start_order):
        remaining = set(start_order)
        clique = set()
        while remaining:
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.add(best_v)
            remaining = remaining & adj[best_v]
        return clique

    for trial in range(100):
        if time.time() > deadline - 30:
            break
        shuffled = list(range(n))
        random.shuffle(shuffled)
        c = greedy_clique(shuffled)
        if len(c) > len(best_clique):
            best_clique = list(c)

    if not best_clique:
        best_clique = [0, 1]

    # ---------- Path finding ----------
    def find_path(s, t, forbidden):
        """BFS for shortest path from s to t avoiding forbidden set (interior only)."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == t:
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

    def find_subdivision(branches):
        """
        Attempts to find internally disjoint paths for all pairs in 'branches'.
        Uses a backtracking approach with randomized ordering.
        Returns (True, paths_dict) or (False, None).
        """
        k = len(branches)
        if k < 2:
            return False, None
        
        branches_set = set(branches)
        pairs = list(combinations(range(k), 2))
        
        # Pre-compute all possible shortest paths for each pair avoiding other branches
        # This helps in choosing "easy" paths first
        pair_candidates = {}
        for i, j in pairs:
            s, t = branches[i], branches[j]
            forbidden = branches_set - {s, t}
            p = find_path(s, t, forbidden)
            if p is None:
                return False, None
            pair_candidates[(i, j)] = p

        # Heuristic: Order pairs by the "tightness" of their path
        # Shorter paths are more constrained, so assign them first
        sorted_pairs = sorted(pairs, key=lambda x: len(pair_candidates[x]))

        # Backtracking search
        def backtrack(idx, used_interior, assigned_paths):
            if time.time() > deadline:
                return False
            if idx == len(sorted_pairs):
                return True
            
            i, j = sorted_pairs[idx]
            s, t = branches[i], branches[j]
            
            # Generate candidate paths for this pair, avoiding current used_interior
            # We try the shortest path first, then longer ones if needed
            # To keep it fast, we only try a few random variations if the first fails
            
            candidates = []
            # 1. The pre-computed shortest path (if still valid)
            p_short = pair_candidates[(i, j)]
            if not (set(p_short[1:-1]) & used_interior):
                candidates.append(p_short)
            
            # 2. Find a new path avoiding used_interior
            p_new = find_path(s, t, used_interior)
            if p_new and p_new not in candidates:
                candidates.append(p_new)
                
            # 3. If we have time, try a few random walks to find alternative paths
            # This helps escape local minima
            if len(candidates) < 5:
                for _ in range(10):
                    # Randomized BFS/DFS
                    visited = {s}
                    parent = {s: None}
                    queue = deque([s])
                    found = False
                    while queue and not found:
                        cur = queue.popleft()
                        # Shuffle neighbors for randomization
                        nbs = list(adj[cur])
                        random.shuffle(nbs)
                        for nb in nbs:
                            if nb == t:
                                # Reconstruct
                                path = [t]
                                node = cur
                                while node is not None:
                                    path.append(node)
                                    node = parent[node]
                                path.reverse()
                                if not (set(path[1:-1]) & used_interior):
                                    candidates.append(path)
                                    found = True
                                    break
                            if nb in used_interior:
                                continue
                            if nb not in visited:
                                visited.add(nb)
                                parent[nb] = cur
                                queue.append(nb)
                    if time.time() > deadline:
                        return False

            # Try each candidate
            # Sort candidates by length (shorter is generally better for disjointness)
            candidates.sort(key=lambda x: len(x))
            
            for p in candidates:
                interior = set(p[1:-1])
                if interior & used_interior:
                    continue
                
                # Assign this path
                assigned_paths[(i, j)] = p
                new_used = used_interior | interior
                
                if backtrack(idx + 1, new_used, assigned_paths):
                    return True
                
                # Undo
                assigned_paths.pop((i, j), None)
            
            return False

        assigned_paths = {}
        # Try a few different random orderings of the sorted pairs to break ties
        success = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            # Shuffle the sorted_pairs slightly to break ties in length
            shuffled_pairs = sorted_pairs.copy()
            random.shuffle(shuffled_pairs)
            # Re-sort by length but keep some randomness among same-length paths
            # Actually, just shuffling the list of pairs is enough to change the backtracking order
            assigned_paths = {}
            if backtrack(0, set(), assigned_paths):
                success = True
                break
        
        if success:
            # Normalize keys to (min, max) for consistency
            final_paths = {}
            for (i, j), p in assigned_paths.items():
                final_paths[(min(i, j), max(i, j))] = p
            return True, final_paths
        else:
            return False, None

    # ---------- Search for maximum valid subdivision ----------
    best_valid = None
    best_valid_size = 0

    # 1. Try the max clique
    if len(best_clique) >= 2:
        valid, paths = find_subdivision(best_clique)
        if valid:
            best_valid = (best_clique, paths)
            best_valid_size = len(best_clique)

    # 2. Try to grow from the max clique
    if best_valid is not None:
        current_branches = list(best_valid[0])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each vertex not in current_branches
            for v in range(n):
                if v in current_branches:
                    continue
                new_branches = current_branches + [v]
                valid, new_paths = find_subdivision(new_branches)
                if valid:
                    current_branches = new_branches
                    best_valid = (current_branches, new_paths)
                    best_valid_size = len(current_branches)
                    improved = True
                    break # Restart the loop with the new set

    # 3. If max clique didn't work, try subsets
    if best_valid is None or best_valid_size < 2:
        for size in range(len(best_clique) - 1, 1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in combinations(best_clique, size):
                valid, paths = find_subdivision(list(subset))
                if valid:
                    best_valid = (list(subset), paths)
                    best_valid_size = size
                    found = True
                    break
            if found:
                break

    # 4. Fallback: Find any valid pair
    if best_valid is None or len(best_valid[0]) < 2:
        for u, v in edges:
            if u != v:
                best_valid = ([u, v], {(0, 1): [u, v]})
                best_valid_size = 2
                break

    if best_valid is None:
        u, v = edges[0]
        best_valid = ([u, v], {(0, 1): [u, v]})

    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    branch_to_idx = {b: i for i, b in enumerate(branches)}

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            key = (i, j)
            p = paths_dict.get(key)
            if p is None:
                # Should not happen
                p = [branches[i], branches[j]]
            
            # Ensure path starts and ends with correct branches
            if p[0] != branches[i]:
                p = p[::-1]
            if p[0] != branches[i] or p[-1] != branches[j]:
                # Fallback if direction is still wrong
                p = [branches[i]] + p[1:-1] + [branches[j]]
            
            output_paths.append(list(p))

    # Final type safety
    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
