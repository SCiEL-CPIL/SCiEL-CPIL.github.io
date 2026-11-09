"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import deque
import sys
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    def find_all_shortest_paths(src, dst, forbidden):
        """Find all shortest paths from src to dst avoiding forbidden vertices."""
        if src == dst:
            return [[src]]
        
        # BFS to find distance and predecessors
        dist = {src: 0}
        preds = {src: []}
        queue = deque([src])
        
        while queue:
            curr = queue.popleft()
            if curr == dst:
                continue
            for nb in adj[curr]:
                if nb in forbidden or nb in dist:
                    continue
                dist[nb] = dist[curr] + 1
                preds[nb] = preds.get(nb, []) + [curr]
                queue.append(nb)
        
        if dst not in dist:
            return []
        
        # Backtrack to find all paths
        paths = []
        def backtrack(curr, path):
            if curr == dst:
                paths.append(path + [dst])
                return
            for p in preds.get(curr, []):
                backtrack(p, path + [curr])
        
        # Start from src
        # We need to be careful not to include forbidden nodes in intermediate steps
        # The BFS already avoided them.
        # But we must ensure the path doesn't go through other branches.
        # 'forbidden' should contain all branches except src and dst.
        
        # Actually, the standard BFS finds *some* path. To find *all* shortest paths,
        # we need to reconstruct them.
        
        # Let's do a simpler DFS limited to shortest path length
        target_dist = dist[dst]
        
        def dfs(curr, path, remaining):
            if curr == dst:
                if remaining == 0:
                    paths.append(path + [dst])
                return
            if remaining == 0:
                return
            for nb in adj[curr]:
                if nb in forbidden or nb in path:
                    continue
                if dist.get(nb, float('inf')) == dist[curr] + 1:
                    # Only move to nodes that are on a shortest path
                    dfs(nb, path + [nb], remaining - 1)
        
        dfs(src, [src], target_dist)
        return paths

    def verify_subset(branches):
        """
        Verify if the given branches can be connected by pairwise internally disjoint paths.
        Uses backtracking on the assignment of paths to pairs.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        used_interior = set()
        paths = {}
        
        # Generate all pairs
        pairs = list(combinations(range(k), 2))
        
        # Order pairs by "difficulty": shortest path length first
        def get_difficulty(i, j):
            b_i, b_j = branches[i], branches[j]
            forbidden = branch_set - {b_i, b_j}
            paths_list = find_all_shortest_paths(b_i, b_j, forbidden)
            if not paths_list:
                return float('inf')
            return len(paths_list[0])
        
        # Sort pairs by difficulty (shortest path first)
        pairs.sort(key=lambda p: get_difficulty(p[0], p[1]))
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            i, j = pairs[idx]
            src, dst = branches[i], branches[j]
            forbidden = (branch_set - {src, dst}) | used_interior
            
            # Find all valid shortest paths
            # We might need longer paths if shortest are blocked, but for now try shortest first.
            # If no shortest path works, we should try longer paths.
            # For efficiency, let's try to find *any* path, preferring short ones.
            
            # We will collect candidate paths
            candidates = []
            
            # BFS to find shortest path avoiding forbidden
            visited = {src}
            queue = deque([(src, [src])])
            found_shortest = False
            shortest_len = float('inf')
            
            while queue:
                curr, path = queue.popleft()
                if len(path) > shortest_len:
                    break # BFS expands in layers, so if we've found a shortest, we can stop if we want just one.
                          # But we want options. Let's collect all paths of length <= some limit.
                
                for nb in adj[curr]:
                    if nb == dst:
                        candidate_path = path + [dst]
                        # Check if interior is free
                        valid = True
                        for v in candidate_path[1:-1]:
                            if v in forbidden:
                                valid = False
                                break
                        if valid:
                            candidates.append(candidate_path)
                            shortest_len = len(candidate_path)
                            found_shortest = True
                        continue
                    
                    if nb in visited or nb in forbidden:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
                
                # If we found a shortest path, we might break early if we only need one,
                # but for backtracking, having multiple options is better.
                # However, collecting ALL paths is expensive.
                # Let's just collect the first few shortest paths found.
                if found_shortest and len(candidates) >= 5:
                     # Stop expanding if we have enough options
                     pass 
            
            # If no candidates found, try longer paths?
            # For now, if no shortest path works, it might still be possible with a longer one.
            # But usually, if the shortest is blocked by used_interior, a longer one might exist.
            # Let's try a simple DFS for any path if BFS (shortest) fails.
            if not candidates:
                # Try to find any path
                visited = {src}
                queue = deque([(src, [src])])
                while queue:
                    curr, path = queue.popleft()
                    for nb in adj[curr]:
                        if nb == dst:
                            candidate_path = path + [dst]
                            valid = True
                            for v in candidate_path[1:-1]:
                                if v in forbidden:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(candidate_path)
                                break
                        if nb in visited or nb in forbidden:
                            continue
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
                    if candidates:
                        break

            if not candidates:
                return False
            
            # Shuffle candidates to add randomness?
            random.shuffle(candidates)
            
            for path in candidates:
                interior = set(path[1:-1])
                # Check again just in case
                if interior & used_interior:
                    continue
                if interior & (branch_set - {src, dst}):
                    continue
                
                used_interior.update(interior)
                paths[(i, j)] = path
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                used_interior.difference_update(interior)
                del paths[(i, j)]
            
            return False
        
        if backtrack(0):
            result_paths = [list(paths[p]) for p in pairs]
            return {'branches': list(branches), 'paths': result_paths}
        
        return None

    # Strategy: Randomized Greedy with Backtracking
    degrees = [len(adj[v]) for v in range(n)]
    sorted_vertices = sorted(range(n), key=lambda v: -degrees[v])
    
    best_result = None
    best_k = 0
    
    # Try multiple random restarts
    for _ in range(20):
        # Create a random permutation of vertices, biased towards high degree
        candidates = sorted_vertices[:]
        random.shuffle(candidates)
        
        current_set = []
        
        for v in candidates:
            if v in current_set:
                continue
            trial_set = current_set + [v]
            if len(trial_set) >= 2:
                result = verify_subset(trial_set)
                if result is not None:
                    current_set = trial_set
                    if len(current_set) > best_k:
                        best_k = len(current_set)
                        best_result = result
                    # Optional: stop if we reached a very high k?
                    # No, keep trying to add more.
        
        # Also try a greedy approach on the sorted list
        # (Already covered by one of the permutations potentially, but let's be explicit)
    
    # Specific greedy attempt
    current_set = []
    for v in sorted_vertices:
        if v in current_set:
            continue
        trial_set = current_set + [v]
        if len(trial_set) >= 2:
            result = verify_subset(trial_set)
            if result is not None:
                current_set = trial_set
                if len(current_set) > best_k:
                    best_k = len(current_set)
                    best_result = result

    if best_result is not None:
        return best_result
    
    # Fallback: find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
