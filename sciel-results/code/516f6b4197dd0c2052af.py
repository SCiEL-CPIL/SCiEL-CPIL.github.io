import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe, limit is 150s
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def min_vertex_cover(edges, n, time_limit):
        # edges: list of tuples
        # n: number of vertices
        # Returns: list of vertices in the cover
        
        if not edges:
            return []
        
        # Adjacency list for fast removal
        # We will maintain the set of active edges and active vertices
        # A simpler state: set of edges.
        
        best_cover = None
        best_size = float('inf')
        
        # Initial upper bound: Greedy
        # Greedy: pick vertex with max degree, remove it and its edges
        greedy_edges = set(edges)
        greedy_cover = []
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        active_vertices = set(range(n))
        # Remove isolated vertices initially? They don't matter.
        
        # Greedy heuristic
        temp_edges = set(edges)
        temp_adj = {i: set(adj[i]) for i in range(n)}
        while temp_edges:
            # Find vertex with max degree in temp_edges
            max_deg = -1
            max_v = None
            for u, v in temp_edges:
                # We need degrees. Let's compute on the fly or maintain.
                # For small n, on-the-fly is fine.
                pass
            
            # Better greedy:
            # Count degrees
            degs = {i: 0 for i in active_vertices}
            for u, v in temp_edges:
                if u in degs: degs[u] += 1
                if v in degs: degs[v] += 1
                
            # Pick max
            if not degs: break
            max_v = max(degs, key=lambda x: degs[x])
            greedy_cover.append(max_v)
            
            # Remove max_v and its incident edges
            new_temp_edges = set()
            for u, v in temp_edges:
                if u == max_v or v == max_v:
                    continue
                new_temp_edges.add((u, v))
            temp_edges = new_temp_edges
            
            # Update active vertices? Not strictly needed for edge set logic, but good for pruning.
            # We just need to ensure we don't pick isolated vertices again.
            
        best_cover = greedy_cover
        best_size = len(greedy_cover)
        
        # Branch and Bound
        # State: (current_edges, current_size, current_cover)
        # To avoid copying large sets, we can use a stack or recursion with modification.
        # Given n=40, recursion depth is small.
        
        def bbs(edges_set, size, cover):
            nonlocal best_size, best_cover, deadline
            
            if time.time() > deadline:
                return
                
            if not edges_set:
                if size < best_size:
                    best_size = size
                    best_cover = list(cover)
                return
                
            # Lower Bound: Maximal Matching
            # Greedy maximal matching
            remaining = set(edges_set)
            matching_size = 0
            while remaining:
                # Pick an edge
                e = next(iter(remaining))
                u, v = e
                matching_size += 1
                # Remove all edges incident to u or v
                to_remove = set()
                for eu, ev in remaining:
                    if eu == u or eu == v or ev == u or ev == v:
                        to_remove.add((eu, ev))
                remaining -= to_remove
                
            lb = size + matching_size
            if lb >= best_size:
                return
            
            # Branch
            # Pick an edge to branch on. 
            # Heuristic: Pick edge with highest degree sum? Or just first.
            # Let's pick an edge (u,v) where u has max degree in edges_set?
            # Or simply pick the first edge.
            # To improve pruning, pick an edge (u,v) such that u and v have high degrees.
            # Let's compute degrees.
            degs = {}
            for u, v in edges_set:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1
            
            # Find edge with max max(degs[u], degs[v])?
            # Or just pick one with high degree.
            # Let's pick the edge (u,v) with the largest degs[u] + degs[v].
            max_score = -1
            branch_edge = None
            for u, v in edges_set:
                score = degs.get(u, 0) + degs.get(v, 0)
                if score > max_score:
                    max_score = score
                    branch_edge = (u, v)
                    
            if branch_edge is None:
                return
                
            u, v = branch_edge
            
            # Branch 1: Include u
            new_edges_1 = set()
            for eu, ev in edges_set:
                if eu != u and ev != u:
                    new_edges_1.add((eu, ev))
            bbs(new_edges_1, size + 1, cover + [u])
            
            # Branch 2: Include v
            new_edges_2 = set()
            for eu, ev in edges_set:
                if eu != v and ev != v:
                    new_edges_2.add((eu, ev))
            bbs(new_edges_2, size + 1, cover + [v])
            
        # Start B&B
        initial_edges = set(edges)
        bbs(initial_edges, 0, [])
        
        return best_cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cover = min_vertex_cover(edges, n, deadline)
        out.append(sorted(cover))
        
    return out
