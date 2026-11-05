"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns edges of a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Construct the Clebsch Graph (16 vertices, 40 edges)
    # Vertices are 4-bit vectors (0..15).
    # Edges exist if Hamming distance is 2.
    def hamming_dist(u, v):
        return bin(u ^ v).count('1')
        
    clebsch_edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if hamming_dist(u, v) == 2:
                clebsch_edges.append([u, v])
                
    # Verify Clebsch graph properties (optional, but good for sanity)
    # It is triangle-free, 5-regular, and 4-chromatic.
    
    best_edges = clebsch_edges
    best_count = len(best_edges)
    
    # 2. Local Search to see if we can add any edges
    # We represent the graph as a set of edges for fast lookup.
    def make_edge_set(edges):
        return set((min(u, v), max(u, v)) for u, v in edges)
        
    current_edge_set = make_edge_set(best_edges)
    
    def is_triangle_free_set(edge_set, u, v):
        """Check if adding edge (u, v) to the graph represented by edge_set creates a triangle."""
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        # A triangle is created if there is a vertex w such that (u,w) and (v,w) are in edge_set.
        # We can check common neighbors.
        neighbors_u = {x for (a, b) in edge_set if a == u for x in [b] if x != v} | \
                      {x for (a, b) in edge_set if b == u for x in [a] if x != v}
        neighbors_v = {x for (a, b) in edge_set if a == v for x in [b] if x != u} | \
                      {x for (a, b) in edge_set if b == v for x in [a] if x != u}
        
        if neighbors_u & neighbors_v:
            return False
        return True

    def is_3_colorable(edge_set, max_nodes=16):
        """
        Exact backtracking 3-colorability check.
        Returns True if 3-colorable, False if not.
        Uses MRV (Minimum Remaining Values) heuristic.
        """
        # Build adjacency list
        adj = [[] for _ in range(max_nodes)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
            
        # Precompute degrees for initial ordering, but MRV is dynamic
        # State: colors[0..15], -1 is uncolored
        colors = [-1] * max_nodes
        
        def backtrack(node):
            if node == max_nodes:
                return True
            # Try to color node 'node'
            # Available colors are those not used by neighbors
            used_colors = set()
            for neighbor in adj[node]:
                if colors[neighbor] != -1:
                    used_colors.add(colors[neighbor])
            
            for c in range(3):
                if c not in used_colors:
                    colors[node] = c
                    if backtrack(node + 1):
                        return True
                    colors[node] = -1
            return False
        
        # Heuristic: Order nodes by degree descending to fail fast?
        # Or use recursive MRV. For 16 nodes, simple order might suffice,
        # but let's try a static order based on degree first, as MRV is expensive to compute dynamically in pure Python every step.
        # Actually, for 16 nodes, even naive backtracking is fast if the graph is dense.
        # Clebsch is 4-chromatic, so it will fail. The search space is pruned heavily.
        
        # Let's try ordering by degree descending (higher degree nodes first)
        order = sorted(range(max_nodes), key=lambda x: -len(adj[x]))
        
        def backtrack_order(idx):
            if idx == max_nodes:
                return True
            node = order[idx]
            used_colors = set()
            for neighbor in adj[node]:
                if colors[neighbor] != -1:
                    used_colors.add(colors[neighbor])
            
            for c in range(3):
                if c not in used_colors:
                    colors[node] = c
                    if backtrack_order(idx + 1):
                        return True
                    colors[node] = -1
            return False
        
        return backtrack_order(0)

    # Search loop: Try to add an edge, check triangle-freeness.
    # If triangle-free, check 3-colorability.
    # If not 3-colorable and has more edges, it's a better solution.
    # If 3-colorable, we could still keep it temporarily? 
    # The task requires the FINAL result to be not 3-colorable.
    # However, a valid path might involve making it 3-colorable to add more edges, 
    # then finding a subset or configuration that is 4-chromatic.
    # Given the strict requirement "returns a ... graph that is not 3-colourable",
    # we should only accept states that are NOT 3-colorable as "best".
    
    # We will try all possible missing edges.
    all_possible_edges = [(u, v) for u in range(16) for v in range(u + 1, 16)]
    missing_edges = [e for e in all_possible_edges if e not in current_edge_set]
    
    # Sort missing edges by some heuristic? Random?
    # Let's iterate through them.
    
    improved = False
    # We might do a few rounds of search.
    # Since n=16 is small, checking all pairs is cheap.
    
    # Let's try to add each missing edge one by one to the BEST graph found so far.
    # This is a greedy hill-climbing.
    
    for edge in missing_edges:
        if time.time() > deadline:
            break
            
        u, v = edge
        if not is_triangle_free_set(current_edge_set, u, v):
            continue
            
        # Try adding it
        new_edge_set = set(current_edge_set)
        new_edge_set.add((u, v))
        
        # Check 3-colorability
        if not is_3_colorable(new_edge_set):
            # If it's not 3-colorable, it's a valid candidate.
            # Since we added an edge to a valid candidate (assuming best_edges was valid),
            # it is also triangle-free.
            best_edges = [list(e) for e in new_edge_set]
            best_count = len(best_edges)
            current_edge_set = new_edge_set
            improved = True
            
    # If we improved, we should restart the search from the new best, as new missing edges might now be addable.
    if improved:
        # Re-evaluate missing edges based on new state
        missing_edges = [e for e in all_possible_edges if e not in current_edge_set]
        for edge in missing_edges:
            if time.time() > deadline:
                break
            u, v = edge
            if not is_triangle_free_set(current_edge_set, u, v):
                continue
            new_edge_set = set(current_edge_set)
            new_edge_set.add((u, v))
            if not is_3_colorable(new_edge_set):
                best_edges = [list(e) for e in new_edge_set]
                best_count = len(best_edges)
                current_edge_set = new_edge_set

    # If we have time, we could try a randomized search or swap, 
    # but for n=16, the Clebsch graph is likely optimal or near-optimal.
    # The known maximum for triangle-free 4-chromatic on 16 vertices is 40 (Clebsch).
    # Is it possible to have 41?
    # Mantel's theorem says max edges for triangle-free is floor(n^2/4) = 64.
    # But we need chromatic number >= 4.
    # If we add an edge to Clebsch, does it become 3-colorable?
    # The Clebsch graph is the unique (16, 5, 2) Moore graph? No, Clebsch is (16, 5, 2) cage? No.
    # Clebsch graph has 40 edges.
    # If we add an edge, it might create a triangle (blocked) or become 3-colorable.
    # My search above checks this.
    
    return best_edges
# EVOLVE-BLOCK-END
