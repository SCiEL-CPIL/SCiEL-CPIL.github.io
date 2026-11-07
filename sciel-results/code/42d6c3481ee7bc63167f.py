"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

# The graph is provided in the seed. We use it directly.
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and matrix
    adj_list = [[] for _ in range(n)]
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj[u][v] = True
        adj[v][u] = True
    
    # Function to check if a set of vertices forms a clique
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if not adj[vertices[i]][vertices[j]]:
                    return False
        return True
    
    # Find max clique
    max_clique_size = 0
    max_clique_vertices = []
    
    for k in range(min(n, 10), 2, -1):
        found = False
        for comb in itertools.combinations(range(n), k):
            if is_clique(comb):
                max_clique_size = k
                max_clique_vertices = list(comb)
                found = True
                break
        if found:
            break
    
    if max_clique_size < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    # Build initial result from max clique (all direct edges)
    def build_clique_result(clique):
        paths = []
        for pair in itertools.combinations(clique, 2):
            paths.append([int(pair[0]), int(pair[1])])
        return {'branches': [int(v) for v in clique], 'paths': paths}
    
    best_result = build_clique_result(max_clique_vertices)
    
    # Now try to find larger clique subdivisions
    # Strategy: For k from max_clique_size+1 up to some limit, try to find k vertices
    # with internally disjoint paths between all pairs
    
    def find_path_avoiding(src, dst, forbidden, max_len=6):
        """Find a simple path from src to dst avoiding forbidden vertices (except endpoints)."""
        if src == dst:
            return None
        if forbidden and (src in forbidden or dst in forbidden):
            return None
        
        # BFS with path tracking
        visited = set(forbidden)
        visited.add(src)
        queue = deque([(src, [src])])
        
        while queue:
            node, path = queue.popleft()
            if len(path) > max_len:
                continue
            for neighbor in adj_list[node]:
                if neighbor == dst:
                    return path + [dst]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None
    
    def try_subdivision(branches, time_limit):
        """Try to find internally disjoint paths for all pairs of branches."""
        n_branches = len(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(range(n_branches), 2))
        paths = {}
        used_interiors = set()
        
        # Order pairs by difficulty: start with pairs that have direct edges
        def pair_difficulty(pair):
            i, j = pair
            if adj[branches[i]][branches[j]]:
                return 0
            return 1
        
        pairs_sorted = sorted(pairs, key=pair_difficulty)
        
        for pi, pj in pairs_sorted:
            if time.time() > time_limit:
                return None
            
            b_i, b_j = branches[pi], branches[pj]
            
            # Try multiple random orderings of forbidden vertices to find a path
            best_path = None
            for attempt in range(20):
                if time.time() > time_limit:
                    return None
                
                # Forbid all branch vertices except the two endpoints, plus used interiors
                forbidden = set(branch_set - {b_i, b_j})
                forbidden.update(used_interiors)
                
                path = find_path_avoiding(b_i, b_j, forbidden, max_len=5)
                if path is not None:
                    # Check if interior vertices are not already used
                    interior = set(path[1:-1])
                    if interior & used_interiors:
                        continue
                    best_path = path
                    break
            
            if best_path is None:
                return None
            
            paths[(pi, pj)] = best_path
            used_interiors.update(best_path[1:-1])
        
        # Build result
        result_paths = []
        for pi, pj in pairs:
            result_paths.append([int(v) for v in paths[(pi, pj)]])
        
        return {'branches': [int(v) for v in branches], 'paths': result_paths}
    
    # Try to extend the clique with additional vertices
    # First, try all combinations of size max_clique_size+1
    for k in range(max_clique_size + 1, min(n, 8) + 1):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate candidate sets: start with max clique and add one vertex at a time
        # or try random combinations
        candidates = set()
        
        # Add vertices adjacent to many clique members
        if max_clique_size >= 3:
            for v in range(n):
                if v in max_clique_vertices:
                    continue
                # Count how many clique members are adjacent to v
                adj_count = sum(1 for c in max_clique_vertices if adj[c][v])
                if adj_count >= max_clique_size - 1:
                    candidates.add(tuple(sorted(max_clique_vertices + [v])))
        
        # Also try random combinations
        for _ in range(500):
            if time.time() > DEADLINE - 5:
                break
            sample = random.sample(range(n), k)
            candidates.add(tuple(sorted(sample)))
        
        for cand in candidates:
            if time.time() > DEADLINE - 3:
                break
            
            # Quick check: at least some pairs should be connected
            cand_list = list(cand)
            result = try_subdivision(cand_list, time.time() + 0.5)
            if result is not None:
                if len(result['branches']) > len(best_result['branches']):
                    best_result = result
        
        if len(best_result['branches']) >= k:
            # Found a valid subdivision of size k, try to go bigger
            continue
    
    # Final validation and type conversion
    final_branches = [int(v) for v in best_result['branches']]
    final_paths = [[int(v) for v in path] for path in best_result['paths']]
    
    return {'branches': final_branches, 'paths': final_paths}

# EVOLVE-BLOCK-END
