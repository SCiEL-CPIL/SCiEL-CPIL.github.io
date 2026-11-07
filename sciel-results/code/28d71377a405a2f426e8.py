"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
import numpy as np

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

    def check_time():
        return time.time() < deadline

    def bfs_shortest_path(u, v, forbidden):
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]
        visited = {u: None}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            if node == v:
                path = [v]
                while path[-1] != u:
                    path.append(visited[path[-1]])
                path.reverse()
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited[neighbor] = node
                    queue.append(neighbor)
        return None

    def find_disjoint_paths_for_branches(branches, max_attempts=100):
        """
        Try to find vertex-disjoint paths for all pairs of branches.
        Uses randomized greedy with multiple restarts.
        Returns dict mapping (u,v) -> path, or None.
        """
        if len(branches) < 2:
            return {}
        
        pairs = list(combinations(sorted(branches), 2))
        best_result = None
        
        for attempt in range(max_attempts):
            if not check_time():
                break
            
            # Random ordering of pairs
            shuffled_pairs = list(pairs)
            random.shuffle(shuffled_pairs)
            
            candidate_paths = {}
            used_interiors = set()
            success = True
            
            for u, v in shuffled_pairs:
                path_forbidden = set(used_interiors)
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                
                p = bfs_shortest_path(u, v, path_forbidden)
                if p is None:
                    success = False
                    break
                
                candidate_paths[(u, v)] = p
                used_interiors.update(p[1:-1])
            
            if success:
                if best_result is None or len(used_interiors) < len(set(x for pp in best_result.values() for x in pp[1:-1])):
                    best_result = candidate_paths
                    if len(used_interiors) == 0:
                        break  # All paths are direct edges, optimal
        
        return best_result

    def is_valid_solution(branches, paths_dict):
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
        
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
            
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return False
            
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
        
        return True

    # Find maximum clique using Bron-Kerbosch with pivot
    def max_clique():
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if not check_time():
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
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

    # Main strategy: Use ILP-like approach via exhaustive search over clique subsets
    # Since n=24, we can be more systematic
    
    # First find a large clique to serve as candidate branches
    clique = max_clique()
    
    best_branches = []
    best_paths = {}
    
    if len(clique) >= 2:
        # Sort by degree for better path finding
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)
        
        # Try subsets from largest to smallest
        for size in range(len(clique_sorted), 1, -1):
            if not check_time():
                break
            
            # Try the top 'size' vertices of the sorted clique
            subset = clique_sorted[:size]
            paths = find_disjoint_paths_for_branches(subset, max_attempts=50)
            if paths is not None and is_valid_solution(subset, paths):
                best_branches = subset
                best_paths = paths
                break
        
        # If the top subset didn't work, try other subsets of the same size
        if len(best_branches) < size:
            # Try random subsets of the clique of the same size
            for trial in range(50):
                if not check_time():
                    break
                if len(clique_sorted) <= size:
                    break
                subset = random.sample(clique_sorted, size)
                paths = find_disjoint_paths_for_branches(subset, max_attempts=30)
                if paths is not None and is_valid_solution(subset, paths):
                    if len(subset) > len(best_branches):
                        best_branches = subset
                        best_paths = paths
                        break

    # If clique approach didn't give good results, try greedy construction
    if len(best_branches) < 3:
        # Greedy: start with high-degree edge, add vertices
        sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        
        for u, v in sorted_edges[:30]:
            if not check_time():
                break
            
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]}
            
            for _ in range(150):
                if not check_time():
                    break
                
                candidates = [i for i in range(n) if i not in current_branches]
                candidates.sort(key=lambda x: len(adj[x]), reverse=True)
                
                added = False
                for new_b in candidates[:15]:
                    new_branches = current_branches + [new_b]
                    # Try to find paths for new configuration
                    new_paths = find_disjoint_paths_for_branches(new_branches, max_attempts=20)
                    if new_paths is not None and is_valid_solution(new_branches, new_paths):
                        current_branches = new_branches
                        current_paths = new_paths
                        added = True
                        break
                
                if not added:
                    break
            
            if len(current_branches) > len(best_branches):
                best_branches = current_branches
                best_paths = current_paths

    # Local search: try to improve by swapping branches
    if len(best_branches) >= 2:
        improved = True
        while improved:
            if not check_time():
                break
            improved = False
            
            # Try removing a random branch and adding a different one
            if len(best_branches) < 3:
                break
            
            for trial in range(50):
                if not check_time():
                    break
                
                remove_idx = random.randint(0, len(best_branches) - 1)
                removed = best_branches[remove_idx]
                remaining = [b for i, b in enumerate(best_branches) if i != remove_idx]
                
                # Remove paths involving the removed branch
                reduced_paths = {}
                for key, path in best_paths.items():
                    if removed not in key:
                        reduced_paths[key] = path
                
                # Try to add a new branch
                candidates = [i for i in range(n) if i not in remaining]
                random.shuffle(candidates)
                
                for new_b in candidates[:10]:
                    new_branches = remaining + [new_b]
                    new_paths = find_disjoint_paths_for_branches(new_branches, max_attempts=30)
                    if new_paths is not None and is_valid_solution(new_branches, new_paths):
                        if len(new_branches) > len(best_branches):
                            best_branches = new_branches
                            best_paths = new_paths
                            improved = True
                        break

    # Multiple random restarts
    for restart in range(20):
        if not check_time():
            break
        
        if edges:
            u, v = random.choice(edges)
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]}
        else:
            break
        
        for _ in range(80):
            if not check_time():
                break
            
            candidates = [i for i in range(n) if i not in current_branches]
            random.shuffle(candidates)
            
            added = False
            for new_b in candidates[:10]:
                new_branches = current_branches + [new_b]
                new_paths = find_disjoint_paths_for_branches(new_branches, max_attempts=15)
                if new_paths is not None and is_valid_solution(new_branches, new_paths):
                    current_branches = new_branches
                    current_paths = new_paths
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_branches) > len(best_branches):
            best_branches = current_branches
            best_paths = current_paths

    # Build final output
    if len(best_branches) < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    final_paths = []
    for u, v in combinations(sorted(best_branches), 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])
    
    return {
        'branches': [int(x) for x in best_branches],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
