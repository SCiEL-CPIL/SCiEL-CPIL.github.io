"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build adjacency set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def is_adjacent(u, v):
        return (u, v) in edge_set
    
    def find_vertex_disjoint_paths(branches, internal_pool):
        """
        Given a set of branch vertices, try to find vertex-disjoint paths between
        all pairs of branches using only vertices from internal_pool as interior.
        Returns True if successful, and modifies internal_pool to remove used vertices.
        """
        if len(branches) < 2:
            return True
        
        # Sort branches for deterministic ordering
        branches_sorted = sorted(branches)
        pairs = []
        for i in range(len(branches_sorted)):
            for j in range(i+1, len(branches_sorted)):
                pairs.append((branches_sorted[i], branches_sorted[j]))
        
        # For each pair, find a shortest path avoiding other branches and used internals
        used_internals = set()
        
        for (u, v) in pairs:
            # BFS from u to v, avoiding all branch vertices except u and v
            # and avoiding used_internals
            if u == v:
                continue
            
            # Check direct edge
            if is_adjacent(u, v):
                continue  # Direct edge, no internal needed
            
            # BFS
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            found = False
            
            while queue and not found:
                curr = queue.popleft()
                for nb in sorted(adj[curr]):
                    if nb == v:
                        found = True
                        parent[v] = curr
                        break
                    if nb in visited:
                        continue
                    # nb must not be a branch (except we're looking for v)
                    if nb in branches and nb != u and nb != v:
                        continue
                    if nb in used_internals:
                        continue
                    visited.add(nb)
                    parent[nb] = curr
                    queue.append(nb)
            
            if not found:
                return False
            
            # Reconstruct path
            path = []
            curr = v
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            # Mark internal vertices as used
            for vertex in path[1:-1]:
                used_internals.add(vertex)
        
        # Update internal_pool
        internal_pool -= used_internals
        return True
    
    def find_direct_paths(branches):
        """
        Try to find vertex-disjoint paths where each path is either a direct edge
        or uses unused internal vertices. Returns dict of paths or None.
        """
        branches_sorted = sorted(branches)
        pairs = []
        for i in range(len(branches_sorted)):
            for j in range(i+1, len(branches_sorted)):
                pairs.append((branches_sorted[i], branches_sorted[j]))
        
        used_internals = set()
        paths = {}
        
        for (u, v) in pairs:
            if is_adjacent(u, v):
                paths[(u, v)] = [u, v]
                continue
            
            # BFS avoiding branches and used internals
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            found = False
            
            while queue and not found:
                curr = queue.popleft()
                for nb in sorted(adj[curr]):
                    if nb == v:
                        found = True
                        parent[v] = curr
                        break
                    if nb in visited:
                        continue
                    if nb in branches and nb != u and nb != v:
                        continue
                    if nb in used_internals:
                        continue
                    visited.add(nb)
                    parent[nb] = curr
                    queue.append(nb)
            
            if not found:
                return None
            
            path = []
            curr = v
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            for vertex in path[1:-1]:
                used_internals.add(vertex)
            
            paths[(u, v)] = path
        
        return paths
    
    # Strategy: Greedily build a set of branches, verifying at each step that
    # vertex-disjoint paths exist between all pairs.
    
    all_vertices = list(range(n))
    all_vertices.sort(key=lambda x: len(adj[x]), reverse=True)  # High degree first
    
    best_branches = None
    best_paths = None
    
    # Try different starting points
    for start_idx in range(min(6, len(all_vertices))):
        if time.time() > deadline:
            break
        
        current_branches = [all_vertices[start_idx]]
        remaining = [v for v in all_vertices if v != all_vertices[start_idx]]
        
        # Greedily add vertices
        for _ in range(len(remaining)):
            if time.time() > deadline:
                break
            
            best_candidate = None
            for cand in remaining[:]:
                if time.time() > deadline:
                    break
                
                test_branches = current_branches + [cand]
                paths = find_direct_paths(test_branches)
                
                if paths is not None:
                    best_candidate = cand
                    break
            
            if best_candidate is None:
                break
            
            current_branches.append(best_candidate)
            remaining.remove(best_candidate)
            
            # Update best if improved
            if best_branches is None or len(current_branches) > len(best_branches):
                best_branches = list(current_branches)
                best_paths = find_direct_paths(best_branches)
        
        # Try to improve from current best
        if best_branches is not None:
            # Attempt to add more by trying all remaining vertices
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for cand in all_vertices:
                    if cand in best_branches:
                        continue
                    test_branches = best_branches + [cand]
                    paths = find_direct_paths(test_branches)
                    if paths is not None:
                        best_branches = list(test_branches)
                        best_paths = paths
                        improved = True
                        break
    
    # Fallback: find largest clique
    if best_branches is None or len(best_branches) < 2:
        # Find largest clique using Bron-Kerbosch
        def bron_kerbosch(R, P_set, X, best_clique):
            if len(P_set) == 0 and len(X) == 0:
                if len(R) > len(best_clique[0]):
                    best_clique[0] = list(R)
                return
            
            # Choose pivot
            pivot_candidates = P_set | X
            if not pivot_candidates:
                if len(R) > len(best_clique[0]):
                    best_clique[0] = list(R)
                return
            
            pivot = max(pivot_candidates, key=lambda x: len(adj[x] & P_set))
            candidates = P_set - adj[pivot]
            
            for v in list(candidates):
                if time.time() > deadline:
                    return
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], best_clique)
                P_set.remove(v)
                X.add(v)
        
        best_clique = [[]]
        bron_kerbosch(set(), set(all_vertices), set(), best_clique)
        
        if len(best_clique[0]) >= 2:
            best_branches = list(best_clique[0])
            best_paths = find_direct_paths(best_branches)
        
        if best_paths is None:
            # Last resort: just two adjacent vertices
            u, v = edges[0]
            best_branches = [u, v]
            best_paths = {(u, v): [u, v]}
    
    # Convert to required format
    branches_sorted = sorted(best_branches)
    result_paths = {}
    for i in range(len(branches_sorted)):
        for j in range(i+1, len(branches_sorted)):
            u, v = branches_sorted[i], branches_sorted[j]
            result_paths[(u, v)] = best_paths.get((u, v), best_paths.get((v, u), [u, v]))
    
    # Build output
    out_paths = []
    for i in range(len(branches_sorted)):
        for j in range(i+1, len(branches_sorted)):
            u, v = branches_sorted[i], branches_sorted[j]
            path = result_paths.get((u, v), [u, v])
            # Ensure path starts at u and ends at v
            if path[0] != u:
                path = list(reversed(path))
            out_paths.append(list(path))
    
    return {
        'branches': [int(b) for b in branches_sorted],
        'paths': out_paths
    }
# EVOLVE-BLOCK-END
