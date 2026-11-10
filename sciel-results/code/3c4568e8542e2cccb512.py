"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    def find_disjoint_paths(branches):
        """Find vertex-disjoint paths between all pairs of branches using max-flow."""
        k = len(branches)
        if k < 2:
            return None
        
        # Map branch vertices to indices
        branch_to_idx = {b: i for i, b in enumerate(branches)}
        
        # For each pair of branches, we need a path
        # Use a flow network where each non-branch vertex has capacity 1
        # Branch vertices have infinite capacity (they can be used multiple times as endpoints)
        
        # We'll use a different approach: for each pair, find a path that avoids
        # all other paths' internal vertices. We do this iteratively with backtracking.
        
        # First, let's try a simpler approach: find shortest paths and check for conflicts
        paths = {}
        used_internal = set()
        
        # Sort pairs by some heuristic to find paths in a good order
        pairs = list(combinations(range(k), 2))
        
        # Try to find paths for all pairs
        def try_find_paths(pairs, used_internal, paths, depth=0):
            if time.time() > deadline:
                return None
            if not pairs:
                return paths
            
            # Choose a pair to process (try different heuristics)
            # For now, just take the first pair
            i, j = pairs[0]
            remaining = pairs[1:]
            
            # Find a path from branches[i] to branches[j] avoiding used_internal
            # and not going through any other branch
            path = find_path_avoiding(branches[i], branches[j], used_internal, 
                                      set(branches) - {branches[i], branches[j]})
            
            if path is None:
                return None
            
            # Get internal vertices of this path
            internal = set(path[1:-1])
            
            # Check no conflicts
            if internal & used_internal:
                return None
            
            new_used = used_internal | internal
            new_paths = dict(paths)
            new_paths[(i, j)] = path
            
            result = try_find_paths(remaining, new_used, new_paths, depth + 1)
            if result is not None:
                return result
            
            # Backtrack: try finding a different path
            # For now, this is a simple version without full backtracking
            return None
        
        def find_path_avoiding(start, end, forbidden, avoid_branches):
            """Find a path from start to end avoiding forbidden vertices and other branches."""
            if start == end:
                return [start]
            if start in forbidden or end in forbidden:
                return None
            
            visited = {start}
            queue = deque([(start, [start])])
            
            while queue:
                node, path = queue.popleft()
                if node == end:
                    return path
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in forbidden:
                        continue
                    if neighbor in avoid_branches:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            
            return None
        
        result = try_find_paths(pairs, set(), {})
        return result
    
    def try_branch_set(branches):
        """Try to find disjoint paths for the given branch set."""
        paths = find_disjoint_paths(branches)
        if paths is None:
            return None
        
        # Convert to the required format
        result_paths = []
        for (i, j), path in paths.items():
            result_paths.append(list(path))
        
        return {'branches': list(branches), 'paths': result_paths}
    
    # Start with greedy growth from high-degree vertices
    degree = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
    
    best_result = None
    best_k = 0
    
    # Try to grow a branch set greedily
    current_branches = [sorted_vertices[0]]
    
    for candidate in sorted_vertices[1:]:
        if time.time() > deadline:
            break
        
        new_branches = current_branches + [candidate]
        result = try_branch_set(new_branches)
        
        if result is not None:
            current_branches = new_branches
            best_result = result
            best_k = len(current_branches)
        else:
            # Try skipping this candidate and see if we can find a better set
            # For now, just continue without it
            pass
    
    # If we didn't find anything, try a simpler approach
    if best_result is None:
        # Find a triangle
        es = {tuple(e) for e in edges}
        for row in combinations(range(n), 3):
            if all(e in es for e in combinations(row, 2)):
                return {'branches': list(row), 'paths': [list(e) for e in combinations(row, 2)]}
        
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    return best_result

# EVOLVE-BLOCK-END
