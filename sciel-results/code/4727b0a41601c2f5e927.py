import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    # Initialize with a random K4-free graph
    # Start with empty graph and add edges
    # Or start with K6,6,6 and remove edges
    
    # Let's start with a random graph and fix K4s
    # Or better: Start with K6,6,6 and remove random edges to break symmetry
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.add((min(u,v), max(u,v)))
    
    # Randomly remove 20% of edges
    edges_list = list(edges)
    random.shuffle(edges_list)
    remove_count = len(edges_list) // 5
    for _ in range(remove_count):
        edges.discard(edges_list.pop())
        
    current_edges = edges
    current_score = 12 # Upper bound from K6,6,6
    
    def get_triangles(edge_set):
        triangles = []
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find triangles
        # Iterate over edges (u, v)
        for u, v in edge_set:
            common = adj[u] & adj[v]
            for w in common:
                if u < v < w:
                    triangles.append((u, v, w))
        return triangles

    def min_hitting_set(triangles):
        if not triangles:
            return 0
            
        # Convert to bitmasks for faster intersection?
        # Or just use sets
        # Branch and Bound
        best = n
        # Sort triangles by size? All size 3.
        # Maybe sort by frequency of vertices?
        
        # Precompute vertex frequencies
        freq = [0]*n
        for t in triangles:
            for v in t:
                freq[v] += 1
                
        # Heuristic: pick triangle with highest sum of freqs
        # Or just pick first
        
        def bb(removed_mask, remaining_tris):
            nonlocal best
            if not remaining_tris:
                pop = bin(removed_mask).count('1')
                if pop < best:
                    best = pop
                return
                
            # Pruning
            # Lower bound: 
            # Each vertex can cover at most max_freq triangles?
            # Better: max triangles a single vertex can cover in remaining_tris
            # If max_cover == 0, impossible? No, remaining_tris not empty.
            
            # Simple bound: len(remaining_tris) is upper bound on needed removals? No.
            # If we have T triangles, and each vertex covers at most C triangles, we need at least ceil(T/C) vertices.
            # Calculate max coverage per vertex in remaining_tris
            max_cov = 0
            for v in range(n):
                if removed_mask & (1 << v):
                    continue
                cov = 0
                for t in remaining_tris:
                    if v in t:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return # Should not happen if remaining_tris not empty and no vertex covers them?
                
            lb = (len(remaining_tris) + max_cov - 1) // max_cov
            cur_pop = bin(removed_mask).count('1')
            if cur_pop + lb >= best:
                return
                
            # Pick a triangle to branch on
            # Pick one with highest "weight" (sum of freqs of its vertices not removed)
            tri = max(remaining_tris, key=lambda t: sum(freq[v] for v in t if not (removed_mask & (1<<v))))
            
            for v in tri:
                if removed_mask & (1 << v):
                    continue
                new_removed = removed_mask | (1 << v)
                # Filter remaining triangles
                new_remaining = [t for t in remaining_tris if v not in t]
                bb(new_removed, new_remaining)
                
        bb(0, triangles)
        return best

    def get_score(edge_set):
        tris = get_triangles(edge_set)
        if not tris:
            return 18
        mh = min_hitting_set(tris)
        return n - mh

    # Local Search
    best_edges = current_edges
    best_score = get_score(current_edges)
    
    while time.time() < deadline:
        # Generate neighbor
        # 50% remove edge, 50% add edge
        if random.random() < 0.5 and len(current_edges) > 0:
            e = random.choice(list(current_edges))
            new_edges = set(current_edges)
            new_edges.discard(e)
        else:
            # Try to add an edge
            # Pick random non-edge
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            non_edges = [e for e in all_edges if e not in current_edges]
            if not non_edges:
                continue
            e = random.choice(non_edges)
            new_edges = set(current_edges)
            new_edges.add(e)
            
            # Check K4 constraint
            u, v = e
            # Check if adding (u,v) creates a K4
            # Need x, y such that u-x, u-y, v-x, v-y, x-y exist
            adj = {i: set() for i in range(n)}
            for a, b in new_edges:
                adj[a].add(b)
                adj[b].add(a)
                
            common = adj[u] & adj[v]
            k4_found = False
            for x in common:
                for y in common:
                    if x < y and (x, y) in new_edges:
                        k4_found = True
                        break
                if k4_found:
                    break
            if k4_found:
                continue
                
        new_score = get_score(new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
            current_edges = new_edges
        elif new_score == best_score and random.random() < 0.1:
            current_edges = new_edges
        else:
            # Keep current or revert?
            # Simple hill climbing: revert if worse
            pass
            
    return [[u, v] for u, v in best_edges]
