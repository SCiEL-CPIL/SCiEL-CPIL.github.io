"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all-pairs shortest distances for ordering heuristic
    dist = [[n] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v in edges:
        dist[u][v] = 1
        dist[v][u] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    def check_time():
        return time.time() < deadline
    
    def bfs_path(u, v, forbidden, max_len=None):
        """Find a path from u to v avoiding forbidden nodes, optionally with max length."""
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]
        
        visited = {u: None}
        queue = deque([(u, 0)])
        
        while queue:
            node, depth = queue.popleft()
            if max_len is not None and depth > max_len:
                continue
            if node == v:
                # Reconstruct path
                path = [v]
                while path[-1] != u:
                    path.append(visited[path[-1]])
                path.reverse()
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited[neighbor] = node
                    queue.append((neighbor, depth + 1))
        
        return None
    
    def bfs_all_paths(u, v, forbidden, max_paths=5):
        """Find multiple simple paths from u to v avoiding forbidden nodes."""
        if u in forbidden or v in forbidden:
            return []
        if u == v:
            return [[u, v]]
        
        paths = []
        # Use DFS to find multiple paths
        def dfs(current, target, visited, path):
            if len(paths) >= max_paths:
                return
            if current == target:
                paths.append(list(path))
                return
            for neighbor in adj[current]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, visited, path)
                    path.pop()
                    visited.remove(neighbor)
                    if len(paths) >= max_paths:
                        return
        
        dfs(u, v, {u}, [u])
        return paths
    
    def is_valid_solution(branches, paths_dict):
        """Validate that the solution satisfies all constraints."""
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
        
        # Check all pairs have paths
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict and (v, u) not in paths_dict:
                return False
        
        used_interiors = set()
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] not in (u, v) or path[-1] not in (u, v):
                return False
            if path[0] == path[-1]:
                return False
            if len(set(path)) != len(path):
                return False
            
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
            # Check interiors
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
        
        return True
    
    def try_find_paths_for_branches(branches, num_attempts=20):
        """
        Given a set of branch vertices, find vertex-disjoint paths between all pairs.
        Uses distance-based ordering and multiple path options.
        Returns dict mapping (u,v) -> path, or None if not possible.
        """
        if len(branches) < 2:
            return {}
        
        pairs = list(combinations(branches, 2))
        
        for attempt in range(num_attempts):
            if not check_time():
                return None
            
            if attempt == 0:
                # First attempt: order by distance (closest pairs first)
                pairs_sorted = sorted(pairs, key=lambda p: dist[p[0]][p[1]])
            else:
                # Subsequent attempts: random shuffle
                pairs_sorted = list(pairs)
                random.shuffle(pairs_sorted)
            
            candidate_paths = {}
            used_interiors = set()
            
            success = True
            for u, v in pairs_sorted:
                # Build forbidden set for this path
                path_forbidden = set(used_interiors)
                # All branches except u and v are forbidden
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                
                # Try shortest path first
                p = bfs_path(u, v, path_forbidden)
                
                if p is None:
                    # Try with longer paths
                    p = bfs_path(u, v, path_forbidden, max_len=6)
                
                if p is None:
                    # Try finding any path via DFS
                    all_paths = bfs_all_paths(u, v, path_forbidden, max_paths=3)
                    if all_paths:
                        p = all_paths[0]
                
                if p is None:
                    success = False
                    break
                
                candidate_paths[(u, v)] = p
                used_interiors.update(p[1:-1])
            
            if success:
                return candidate_paths
        
        return None
    
    def max_clique():
        """Find maximum clique in the graph using Bron-Kerbosch with pivot."""
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if not check_time():
                return
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & (P | X)))
            else:
                pivot = None
            
            candidates = set(P)
            if pivot is not None:
                candidates -= adj[pivot]
            
            for v in list(candidates):
                if not check_time():
                    return
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return best_clique
    
    def score_candidate(b, current_branches):
        """Score how good a candidate branch vertex is."""
        # Higher degree is better, but also consider connectivity to existing branches
        degree = len(adj[b])
        # Count how many existing branches are adjacent or close
        close_count = sum(1 for cb in current_branches if dist[b][cb] <= 2)
        return degree * 10 + close_count
    
    def greedy_expand(start_branches):
        """
        Start with initial branches and greedily add more.
        Returns (branches, paths_dict).
        """
        current_branches = list(start_branches)
        
        # First find paths for initial branches
        current_paths = try_find_paths_for_branches(current_branches, num_attempts=30)
        if current_paths is None:
            return [], {}
        
        # Greedily add more branches
        for _ in range(200):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current_branches]
            if not candidates:
                break
            
            # Score and sort candidates
            candidates.sort(key=lambda x: score_candidate(x, current_branches), reverse=True)
            
            added = False
            for new_b in candidates[:12]:
                # Try to add new branch
                new_branches = current_branches + [new_b]
                
                # Find paths for all pairs
                new_paths = try_find_paths_for_branches(new_branches, num_attempts=20)
                
                if new_paths is not None:
                    if is_valid_solution(new_branches, new_paths):
                        current_branches = new_branches
                        current_paths = new_paths
                        added = True
                        break
            
            if not added:
                break
        
        return current_branches, current_paths
    
    def local_search(branches, paths_dict):
        """
        Try to improve the solution by swapping branches.
        Returns improved (branches, paths_dict).
        """
        best = (list(branches), dict(paths_dict))
        best_size = len(branches)
        
        for iteration in range(150):
            if not check_time():
                break
            
            if len(branches) < 3:
                break
            
            # Try removing a random branch and adding a different one
            remove_idx = random.randint(0, len(branches) - 1)
            remaining = [b for i, b in enumerate(branches) if i != remove_idx]
            
            # Try to add a new branch, scored by potential
            candidates = [i for i in range(n) if i not in remaining]
            candidates.sort(key=lambda x: score_candidate(x, remaining), reverse=True)
            
            improved = False
            for new_b in candidates[:15]:
                new_branches = remaining + [new_b]
                new_paths = try_find_paths_for_branches(new_branches, num_attempts=15)
                
                if new_paths is not None and is_valid_solution(new_branches, new_paths):
                    if len(new_branches) > best_size:
                        best = (new_branches, new_paths)
                        best_size = len(new_branches)
                        branches, paths_dict = new_branches, new_paths
                    improved = True
                    break
            
            if not improved:
                continue
        
        return best
    
    # Strategy 1: Use maximum clique
    clique = max_clique()
    
    best_branches = []
    best_paths = {}
    
    if len(clique) >= 2:
        # Sort clique by degree (descending)
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)
        
        # Try subsets from largest to smallest
        for size in range(min(len(clique_sorted), 18), 1, -1):
            if not check_time():
                break
            
            subset = clique_sorted[:size]
            paths = try_find_paths_for_branches(subset, num_attempts=40)
            if paths is not None and is_valid_solution(subset, paths):
                if len(subset) > len(best_branches):
                    best_branches = subset
                    best_paths = paths
                break
    
    # Strategy 2: Greedy expansion from various starting points
    # Start with high-degree edges
    sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
    
    for u, v in sorted_edges[:20]:
        if not check_time():
            break
        
        branches, paths = greedy_expand([u, v])
        if len(branches) > len(best_branches):
            best_branches = branches
            best_paths = paths
    
    # Strategy 3: Random restarts with edge-based init
    for restart in range(60):
        if not check_time():
            break
        
        # Random initialization with an edge
        if edges:
            u, v = random.choice(edges)
            branches, paths = greedy_expand([u, v])
            if len(branches) > len(best_branches):
                best_branches = branches
                best_paths = paths
    
    # Strategy 4: Local search improvement
    if len(best_branches) >= 2:
        best_branches, best_paths = local_search(best_branches, best_paths)
    
    # Strategy 5: Aggressive add - try to add branches to current best
    if len(best_branches) >= 2 and check_time():
        current = list(best_branches)
        current_paths = dict(best_paths)
        
        for _ in range(80):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current]
            if not candidates:
                break
            
            candidates.sort(key=lambda x: score_candidate(x, current), reverse=True)
            
            added = False
            for new_b in candidates[:10]:
                new_branches = current + [new_b]
                new_paths = try_find_paths_for_branches(new_branches, num_attempts=20)
                
                if new_paths is not None and is_valid_solution(new_branches, new_paths):
                    current = new_branches
                    current_paths = new_paths
                    added = True
                    break
            
            if not added:
                break
        
        if len(current) > len(best_branches):
            best_branches = current
            best_paths = current_paths
    
    # Build final output
    if len(best_branches) < 2:
        # Fallback
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    # Construct paths list in consistent order
    final_paths = []
    for u, v in combinations(sorted(best_branches), 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])
    
    return {
        'branches': [int(x) for x in sorted(best_branches)],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
