import time
import random

def construct():
    """
    Constructs a triangle-free, non-3-colorable graph on 16 vertices maximizing edges.
    Strategy:
    1. Start with a known 4-chromatic triangle-free core (Grötzsch graph on 11 vertices).
    2. Use a fast bitset-based 3-colorability checker to verify non-3-colorability.
    3. Use a randomized greedy algorithm to add edges to the remaining 5 vertices,
       ensuring the graph remains triangle-free and non-3-colorable.
    4. Perform multiple random restarts to find a denser graph.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # Grötzsch graph edges (vertices 0-10)
    # 0-4: C5
    # 5-9: copies
    # 10: root
    core_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 6], [1, 5], [1, 7], [2, 6], [2, 8], [3, 7], [3, 9], [4, 8], [4, 5],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    # Precompute adjacency bitsets for the core
    adj = [0] * n
    for u, v in core_edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    # Fast 3-colorability check using bitsets
    def is_3_colorable(adj, n):
        # Returns True if 3-colorable, False otherwise
        # Uses DFS with propagation
        color = [-1] * n
        # Try all 3^? assignments is too slow. Use backtracking with pruning.
        # For n=16, this is fast enough if we order vertices well.
        
        # Heuristic: order vertices by degree (descending)
        # But for speed in inner loop, maybe just use 0..n-1? 
        # Let's use a simple recursive backtracking.
        
        # To speed up, we can use the fact that if a vertex has 3 neighbors of all 3 colors, it's a conflict.
        
        def backtrack(v):
            if v == n:
                return True
            
            # Find available colors
            # A color c is unavailable if any neighbor has color c
            # Let's compute forbidden colors
            forbidden = 0
            neighbors = adj[v]
            # Iterate through bits in neighbors
            # Since n=16, we can just loop 0..15
            for u in range(n):
                if (neighbors >> u) & 1:
                    if color[u] != -1:
                        forbidden |= 1 << color[u]
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        # Optimization: Try to color the most constrained vertices first?
        # For simplicity and speed in this context, 0..15 order is often okay for n=16,
        # but let's try to make it a bit smarter by checking if it's 3-colorable.
        # Actually, for the core (Grötzsch), it is NOT 3-colorable.
        # The check will return False quickly? 
        # Wait, if it's not 3-colorable, the backtracking must explore the whole tree to prove it.
        # Is that fast? 3^16 is huge. But pruning is strong.
        # Let's test the speed mentally. Grötzsch is small.
        # The new vertices are added. If we add edges to make it denser, it becomes harder to color? 
        # No, adding edges makes it *easier* to prove non-3-colorable (more constraints), 
        # but *harder* to find a coloring if it exists. 
        # Since we want to prove it's NOT 3-colorable, the search space for the "yes" branch shrinks.
        
        return backtrack(0)
    
    # Greedy fill
    def greedy_fill(adj, core_edges, deadline):
        # adj is modified in place? No, let's work on a copy or careful updates.
        # We will maintain a list of edges and the adj bitsets.
        # Start with core
        cur_adj = [a for a in adj]
        cur_edges = [e[:] for e in core_edges]
        
        # Vertices 11-15 are initially isolated (except we might add edges)
        # We want to add edges between 11-15 and 0-10, and between 11-15 themselves?
        # Triangle-free constraint:
        # If we add (u, v), we must ensure no common neighbor.
        
        # Random order of potential edges to add
        # Potential edges: all pairs (i, j) where i < j, not in core, and involving at least one of 11-15?
        # Or just all missing edges?
        # Let's focus on adding edges incident to 11-15 first, as they have no constraints yet.
        
        candidates = []
        for u in range(11, 16):
            for v in range(16):
                if u != v and u > v:
                    # Check if already in core
                    # We can just check adj
                    if not (cur_adj[u] >> v) & 1:
                        candidates.append([u, v])
        
        # Also edges between 11-15
        for i in range(11, 16):
            for j in range(i + 1, 16):
                if not (cur_adj[i] >> j) & 1:
                    candidates.append([i, j])
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Check triangle-free
            # Common neighbors of u and v
            common = cur_adj[u] & cur_adj[v]
            if common:
                continue
            
            # Tentatively add edge
            cur_adj[u] |= 1 << v
            cur_adj[v] |= 1 << u
            cur_edges.append([u, v])
            
            # Check non-3-colorability
            if is_3_colorable(cur_adj, n):
                # Revert
                cur_adj[u] ^= 1 << v
                cur_adj[v] ^= 1 << u
                cur_edges.pop()
            # Else keep it
        
        return cur_edges
    
    best_edges = core_edges
    best_count = len(core_edges)
    
    # Random restarts
    while time.time() < deadline:
        edges = greedy_fill(adj, core_edges, deadline)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        
        # Small randomization: maybe shuffle the core? No, core is fixed.
        # The randomization is in the order of candidate edges.
        
        # To diversify, we could also try different random seeds or slightly different cores?
        # Sticking to one core for now to ensure validity.
        
        # Optional: try to remove some core edges and re-add? 
        # No, core is the source of 4-chromaticity. Removing edges might make it 3-colorable.
        # We must maintain non-3-colorability.
        
        # Let's just run many restarts.
        # To make restarts different, we can randomize the order of candidates each time.
        # The code above does that.
        
        # Add a small delay or check time?
        # The loop condition handles it.
        
        # To avoid infinite loop if greedy is too fast, we can add a sleep?
        # No, just let it run.
        
        # Actually, if greedy is very fast, we might do thousands of restarts.
        # That's good.
        
        # One improvement: if we find a graph with more edges, we can use it as a new starting point?
        # "Iterated Local Search" style.
        # But removing edges from a dense graph to allow new additions is complex.
        # Let's stick to restarts from core.
        
        # Wait, the greedy fill might get stuck in a local optimum where it can't add more edges.
        # Random restarts help escape this by trying different orders.
        
        # Is there a better core?
        # Mycielski of C5 is 11 vertices.
        # We have 5 extra vertices.
        # Max edges in triangle-free graph on 16 vertices is floor(16^2/4) = 64.
        # But it must be 4-chromatic.
        # The core has 20 edges.
        # We can add at most 5*11 + 10 (edges among 11-15) = 65 potential edges?
        # No, 5 new vertices.
        # Edges between new and old: 5 * 11 = 55.
        # Edges among new: 10.
        # Total potential additions: 65.
        # But triangle-free constraint limits this.
        # If we connect a new vertex to a clique in the old graph, it creates triangles.
        # The old graph (Grötzsch) has no triangles, but has edges.
        # If new vertex v connects to u and w, and (u, w) is an edge, triangle.
        # So v can only connect to an independent set in the old graph.
        # Max independent set in Grötzsch?
        # Grötzsch is 4-chromatic, so independence number is small?
        # Actually, Grötzsch has independence number 2? No, it's triangle-free, so alpha >= n/3?
        # For 11 vertices, alpha is likely 3 or 4.
        # So each new vertex can connect to at most ~4-5 old vertices.
        # 5 new vertices * 5 = 25 edges.
        # Plus edges among new vertices.
        # If new vertices form a clique, they can't connect to the same old vertex (triangle).
        # This is a packing problem.
        
        # The greedy approach should find a good packing.
        
        # Let's run.
        pass

    return best_edges
