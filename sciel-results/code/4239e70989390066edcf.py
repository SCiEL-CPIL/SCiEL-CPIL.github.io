"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 50
    
    def is_clique(verts):
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if verts[i] not in adj[verts[j]]:
                    return False
        return True
    
    def find_good_clique():
        """Find a large clique using greedy with multiple orderings."""
        best = []
        # Try different orderings
        orderings = [
            sorted(range(n), key=lambda x: -len(adj[x])),
            sorted(range(n)),
            sorted(range(n), key=lambda x: len(adj[x])),
        ]
        for order in orderings:
            clique = []
            for v in order:
                if all(v in adj[u] for u in clique):
                    clique.append(v)
            if len(clique) > len(best):
                best = clique
        # Try to extend
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if v not in best:
                    new_best = best + [v]
                    if is_clique(new_best):
                        best = new_best
                        improved = True
        return best
    
    def try_add_vertex(new_v, branches, path_interiors):
        """
        Try to find internally disjoint paths from new_v to each branch.
        Returns list of paths or None.
        """
        if time.time() > deadline:
            return None
        
        blocked = set(branches) | path_interiors
        # new_v must not be in blocked
        if new_v in blocked:
            return None
        
        # Greedy: for each branch, find shortest path avoiding blocked and used interiors
        interior_used = set()
        paths_found = []
        
        for b in branches:
            # BFS from new_v to b
            visited = {new_v}
            parent = {new_v: None}
            queue = deque([new_v])
            found = False
            while queue:
                u = queue.popleft()
                if u == b:
                    found = True
                    break
                for w in adj[u]:
                    if w not in visited and w not in blocked:
                        if w != b and w in interior_used:
                            continue
                        visited.add(w)
                        parent[w] = u
                        queue.append(w)
            
            if not found:
                return None
            
            # Reconstruct path
            path = [b]
            cur = b
            while parent[cur] is not None:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            
            # Mark interior vertices
            for v in path[1:-1]:
                interior_used.add(v)
            
            paths_found.append(path)
        
        return paths_found
    
    # Step 1: Find initial clique
    branches = find_good_clique()
    
    if len(branches) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Step 2: Build initial paths
    paths_dict = {}
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            u, v = branches[i], branches[j]
            paths_dict[(min(u,v), max(u,v))] = [u, v]
    
    path_interiors = set()  # empty for direct edges
    
    # Step 3: Try to add more vertices
    candidates = [v for v in range(n) if v not in branches]
    candidates.sort(key=lambda x: -len(adj[x]))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for new_v in candidates[:]:
            if time.time() > deadline:
                break
            if new_v in branches:
                continue
            
            new_paths = try_add_vertex(new_v, branches, path_interiors)
            
            if new_paths is not None:
                branches.append(new_v)
                candidates.remove(new_v)
                
                for path in new_paths:
                    u, v = path[0], path[-1]
                    key = (min(u, v), max(u, v))
                    paths_dict[key] = path
                
                for path in new_paths:
                    for v in path[1:-1]:
                        path_interiors.add(v)
                
                improved = True
                candidates.sort(key=lambda x: -len(adj[x]))
                break
    
    # Build final result
    final_branches = [int(b) for b in branches]
    final_paths = []
    for i in range(len(final_branches)):
        for j in range(i + 1, len(final_branches)):
            u, v = final_branches[i], final_branches[j]
            key = (min(u, v), max(u, v))
            path = paths_dict[key]
            final_paths.append([int(x) for x in path])
    
    return {'branches': final_branches, 'paths': final_paths}
# EVOLVE-BLOCK-END

