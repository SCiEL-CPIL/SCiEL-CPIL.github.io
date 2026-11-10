"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Find a good initial clique (branch set)
    # Using a simple greedy max clique heuristic
    def find_max_clique():
        best = []
        # Try starting from each vertex
        for start in range(n):
            clique = [start]
            candidates = [v for v in adj[start] if v > start]
            # Greedy expansion
            while candidates:
                # Pick vertex with max degree in candidates
                best_cand = max(candidates, key=lambda v: len(adj[v].intersection(candidates)))
                clique.append(best_cand)
                candidates = [v for v in candidates if v in adj[best_cand]]
            if len(clique) > len(best):
                best = clique
        return best

    initial_branches = find_max_clique()
    
    # 2. Define path construction function
    def try_construct_paths(branches):
        if len(branches) < 2:
            return True, []
        
        blocked = set()
        paths = {}
        
        # Generate all pairs
        pairs = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                pairs.append((branches[i], branches[j]))
                
        # Sort pairs by shortest path length in original graph to prioritize "easy" connections?
        # Or random order to avoid bias? Let's try sorting by distance.
        def get_dist(u, v, block):
            # BFS
            if u == v: return 0
            if v in block: return float('inf')
            visited = {u: 0}
            queue = [u]
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if nb not in visited and nb not in block:
                        visited[nb] = visited[curr] + 1
                        if nb == v:
                            return visited[nb]
                        queue.append(nb)
            return float('inf')

        # We need to be careful: the "blocked" set changes as we add paths.
        # A static sort might be bad. Let's just iterate and pick the shortest available path at each step.
        remaining_pairs = list(pairs)
        
        while remaining_pairs:
            best_pair = None
            best_path = None
            best_len = float('inf')
            
            for u, v in remaining_pairs:
                # Find shortest path avoiding blocked
                # BFS
                if u in blocked or v in blocked:
                    continue
                parent = {u: None}
                queue = [u]
                found = False
                while queue:
                    curr = queue.pop(0)
                    if curr == v:
                        found = True
                        break
                    for nb in adj[curr]:
                        if nb not in parent and nb not in blocked:
                            parent[nb] = curr
                            queue.append(nb)
                
                if found:
                    # Reconstruct path
                    path = []
                    curr = v
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    if len(path) - 1 < best_len:
                        best_len = len(path) - 1
                        best_pair = (u, v)
                        best_path = path
            
            if best_pair is None:
                return False, None
            
            u, v = best_pair
            # Add interior to blocked
            for node in best_path[1:-1]:
                blocked.add(node)
            
            paths[(u, v)] = best_path
            paths[(v, u)] = best_path # Just in case, though we use unordered
            
            remaining_pairs.remove(best_pair)
            
        return True, paths

    # 3. Greedy Expansion
    branches = list(initial_branches)
    candidates = [v for v in range(n) if v not in branches]
    
    # Sort candidates by degree descending
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    
    improved = True
    while improved:
        improved = False
        for v in candidates:
            new_branches = branches + [v]
            success, _ = try_construct_paths(new_branches)
            if success:
                branches = new_branches
                candidates.remove(v)
                improved = True
                break # Restart with new branch set
        
    # 4. Final Path Construction
    success, path_dict = try_construct_paths(branches)
    
    if not success or len(branches) < 2:
        # Fallback to just an edge
        u, v = P['edges'][0]
        branches = [int(u), int(v)]
        path_dict = {(u, v): [u, v], (v, u): [u, v]}
        
    # Format output
    final_paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            # Ensure we have the path
            if (u, v) in path_dict:
                p = path_dict[(u, v)]
            elif (v, u) in path_dict:
                p = path_dict[(v, u)]
            else:
                # Should not happen if success was True
                p = [u, v]
            final_paths.append([int(x) for x in p])
            
    return {'branches': [int(b) for b in branches], 'paths': final_paths}

# EVOLVE-BLOCK-END
