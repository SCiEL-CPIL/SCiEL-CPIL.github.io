import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 18
    # We represent the bipartite graph B with parts A (0..8) and B (9..17) in the line graph?
    # No, the line graph has 18 vertices.
    # The bipartite graph has E edges. We need E=18.
    # Let the bipartite graph have vertices 0..11 (6+6) or similar.
    # Let's use a bipartite graph with parts A and B.
    # We need to ensure max degree <= 3 to avoid K4 in line graph?
    # Wait, if max degree is 4, line graph has K4.
    # So we MUST restrict to max degree <= 3 in the bipartite graph.
    
    # Let's use a bipartite graph with 6+6 vertices (12 vertices).
    # Max edges 18 (3-regular).
    # This ensures K4-free line graph.
    
    # State: adjacency matrix of bipartite graph (6x6)
    # 1 if edge exists.
    
    def get_line_graph_edges(adj_b):
        # adj_b is 6x6 boolean matrix
        # Vertices of line graph are edges of adj_b.
        # Map (i, j) to vertex index.
        edges_list = []
        for i in range(6):
            for j in range(6):
                if adj_b[i][j]:
                    edges_list.append((i, j))
        
        # Map edge to index
        edge_to_idx = {e: idx for idx, e in enumerate(edges_list)}
        
        # Line graph edges
        lg_edges = []
        # Two edges in B are adjacent in L(B) if they share a vertex.
        # (i1, j1) and (i2, j2) share vertex if i1==i2 or j1==j2.
        for u_idx in range(len(edges_list)):
            i1, j1 = edges_list[u_idx]
            for v_idx in range(u_idx + 1, len(edges_list)):
                i2, j2 = edges_list[v_idx]
                if i1 == i2 or j1 == j2:
                    lg_edges.append([u_idx, v_idx])
        return lg_edges

    def is_k4_free_lg(lg_edges):
        # Check if line graph is K4 free.
        # This is equivalent to checking if B has no K_{1,4} or K_3,3?
        # No, K4 in L(B) <=> 4 edges in B pairwise adjacent.
        # In bipartite B, this means a star K_{1,4} or a K_{2,2} with 4 edges?
        # K_{2,2} has 4 edges. Are they pairwise adjacent?
        # Edges: (a1,b1), (a1,b2), (a2,b1), (a2,b2).
        # (a1,b1) and (a2,b2) do not share vertex.
        # So K_{2,2} does NOT form a K4 in L(B).
        # Only stars K_{1,4} form K4 in L(B).
        # So we just need to ensure max degree in B is <= 3.
        return True # If we enforce deg <= 3 in B, this is always true.

    def get_largest_triangle_free_lg(lg_edges):
        # Build adjacency for line graph
        n_v = len(lg_edges)
        if n_v == 0:
            return 0
        adj = [0] * n_v
        for u, v in lg_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Find largest triangle-free induced subgraph
        # Use the same logic as before, but n=18 is small.
        # We can use the existing get_largest_triangle_free logic.
        # But we need to pass adj.
        
        # Re-use the logic from the seed but optimized for bitmask.
        def max_tf(adj):
            n = len(adj)
            for k in range(n, 0, -1):
                for subset in itertools.combinations(range(n), k):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    
                    is_tf = True
                    for u in subset:
                        for v in subset:
                            if u >= v: continue
                            if adj[u] & (1 << v):
                                if adj[u] & adj[v] & mask:
                                    is_tf = False
                                    break
                        if not is_tf:
                            break
                    if is_tf:
                        return k
            return 0
        return max_tf(adj)

    # Initialize random 3-regular bipartite graph?
    # Or just random with deg <= 3.
    
    best_l = 18
    best_bipartite = None
    
    def gen_random_bipartite():
        # Random bipartite graph with 6+6 vertices, deg <= 3.
        # We want 18 edges.
        # Try to generate 3-regular.
        # Random permutation for right side.
        # For each left node, pick 3 distinct right nodes.
        # This doesn't guarantee right nodes have deg 3.
        # We can use a greedy repair or just accept any deg <= 3 with 18 edges.
        
        adj = [[0]*6 for _ in range(6)]
        edges = 0
        # Add random edges until 18, checking deg <= 3.
        while edges < 18:
            i = random.randint(0, 5)
            j = random.randint(0, 5)
            if adj[i][j] == 0:
                deg_i = sum(adj[i])
                deg_j = sum(adj[j])
                if deg_i < 3 and deg_j < 3:
                    adj[i][j] = 1
                    edges += 1
        return adj

    # Initial
    adj_b = gen_random_bipartite()
    # Ensure it has 18 edges? The loop above might get stuck if not enough capacity.
    # 6*3 = 18 capacity. So it should fill up.
    
    lg_edges = get_line_graph_edges(adj_b)
    # lg_edges should have 18 vertices.
    l = get_largest_triangle_free_lg(lg_edges)
    best_l = l
    best_bipartite = [row[:] for row in adj_b]
    
    while time.time() < deadline:
        # Move: Swap an edge in bipartite graph.
        # Pick an existing edge (i,j) and a non-edge (i', j') or (i, j') or (i', j).
        # Let's try moving an edge from (i,j) to (i, j2) where j2 != j.
        
        i = random.randint(0, 5)
        j = random.randint(0, 5)
        
        # Find an edge to move from row i?
        # Or just pick a random edge (i,j) that exists.
        if adj_b[i][j] == 0:
            continue
            
        # Try to move it to (i, j2)
        j2 = random.randint(0, 5)
        if j2 == j:
            continue
        if adj_b[i][j2] == 1:
            continue
            
        # Check degree constraints
        deg_j = sum(adj_b[k][j] for k in range(6))
        deg_j2 = sum(adj_b[k][j2] for k in range(6))
        
        # If we move (i,j) to (i,j2):
        # deg_j decreases by 1.
        # deg_j2 increases by 1.
        # deg_i stays same.
        
        if deg_j2 >= 3:
            continue
            
        # Apply move
        adj_b[i][j] = 0
        adj_b[i][j2] = 1
        
        lg_edges = get_line_graph_edges(adj_b)
        l = get_largest_triangle_free_lg(lg_edges)
        
        if l < best_l:
            best_l = l
            best_bipartite = [row[:] for row in adj_b]
        else:
            # Revert
            adj_b[i][j2] = 0
            adj_b[i][j] = 1

    # Convert best_bipartite to line graph edges
    final_lg_edges = get_line_graph_edges(best_bipartite)
    return final_lg_edges

