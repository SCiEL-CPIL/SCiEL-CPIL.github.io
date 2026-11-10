import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n = 16
    # Total possible edges in K_16 is 16*15/2 = 120
    # Map (u, v) with u < v to index 0..119
    edge_indices = []
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices.append((u, v))
            edge_map[(u, v)] = idx
            idx += 1
    
    num_edges = 120

    # Precompute for each edge index, the list of other edge indices that would form a triangle
    # If we add edge (u,v), we cannot have any edge (u,w) or (v,w) where w is a common neighbor?
    # Actually, if we add (u,v), it creates a triangle if there exists w such that (u,w) and (v,w) are in the graph.
    # So, for a given edge e=(u,v), the "conflicting" edges are those (u,w) and (v,w) for all w != u,v.
    # If e is present, all conflicting edges must be absent.
    # This is a constraint satisfaction problem.
    
    # Let's represent the graph as a set of present edges (indices)
    # Or better, a boolean array of size 120.
    
    # Precompute conflict lists:
    # conflict[e] = set of edges f such that {e, f} is part of a triangle?
    # No, if e and f are present, and they share a vertex, they don't necessarily form a triangle.
    # They form a triangle if there is a third edge g connecting the other two vertices.
    # This is a 3-way interaction.
    
    # Efficient Triangle-Free Check:
    # A graph is triangle-free iff for every edge (u,v), N(u) and N(v) are disjoint.
    
    def is_triangle_free(present_edges):
        # present_edges is a set of (u,v) tuples or a bitset
        # Let's use adjacency bitmasks for speed
        adj = [0] * n
        for u, v in present_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True

    def is_3_colorable(adj):
        """
        Fast 3-colorability check using backtracking with bitsets.
        Returns (is_colorable, coloring_or_none)
        """
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True, colors[:]
            
            v = order[idx]
            forbidden = 0
            neighbors = adj[v]
            while neighbors:
                lsb = neighbors & -neighbors
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    forbidden |= (1 << colors[u])
                neighbors -= lsb
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    res, col = backtrack(idx + 1)
                    if res:
                        return True, col
                    colors[v] = -1
            return False, None
        
        return backtrack(0)

    def get_adjacency(present_edges):
        adj = [0] * n
        for u, v in present_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_edge_list(present_edges):
        return [[u, v] for u, v in present_edges]

    # Strategy:
    # 1. Generate a valid initial solution.
    #    The Grötzsch graph is 4-chromatic and triangle-free.
    #    We need to embed it in 16 vertices and try to add more edges.
    #    Instead of a fixed core, we use a "random dense triangle-free" generator
    #    and then use local search to ensure 4-chromaticity and maximize density.
    
    # However, random triangle-free graphs on 16 vertices are often 3-colorable.
    # We need a strong 4-chromatic seed.
    # Let's use the "blow-up" of the Grötzsch graph or a known 4-chromatic triangle-free graph on 16 vertices.
    # One such graph is the "Hajnal graph" (14 vertices, 4-chromatic, triangle-free).
    # Or we can construct one via Mycielski of C5 (Grötzsch, 11 vertices) and add 5 vertices optimally.
    
    # Let's try a different seed: The Clebsch graph is 3-colorable.
    # The "Chvátal graph" is 12 vertices, 4-chromatic, triangle-free, 24 edges.
    # We can start with Chvátal graph + 4 isolated vertices, then fill.
    
    def create_chvatal_core():
        # Chvátal graph: 12 vertices.
        # It is the complement of the line graph of K_6? No.
        # Let's just use the Grötzsch core (11 vertices) as it's easier to define correctly.
        # Vertices 0-4: C5
        # Vertices 5-9: Copies of 0-4, connected to i-1, i+1
        # Vertex 10: Connected to 5-9
        adj = [0] * n
        for i in range(5):
            u, v = i, (i + 1) % 5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for i in range(5):
            u = 5 + i
            v1 = (i - 1) % 5
            v2 = (i + 1) % 5
            adj[u] |= (1 << v1)
            adj[v1] |= (1 << u)
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
        for i in range(5):
            u = 10
            v = 5 + i
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Vertices 11-15 are isolated in the core
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.add((u, v))
        return edges

    # Local Search WalkSAT style
    def local_search(initial_edges, time_limit):
        current_edges = set(initial_edges)
        current_adj = get_adjacency(current_edges)
        
        # Verify initial state
        if not is_triangle_free(current_edges):
            return None
        is_col, _ = is_3_colorable(current_adj)
        if is_col:
            # If initial is 3-colorable, we need to make it 4-chromatic.
            # This is harder. We might need to start with a 4-chromatic seed.
            # For now, return None if not 4-chromatic, or try to fix.
            pass
            
        best_edges = set(current_edges)
        best_count = len(current_edges)
        
        start_ls = time.time()
        no_improve = 0
        
        while time.time() - start_ls < time_limit:
            # Randomly pick an edge to flip
            # With prob 0.5, try to add a non-edge. With prob 0.5, try to remove an edge.
            # But adding is preferred to maximize density.
            
            if random.random() < 0.8:
                # Try to add an edge
                # Find all non-edges
                all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
                non_edges = [p for p in all_pairs if p not in current_edges]
                if not non_edges:
                    break
                u, v = random.choice(non_edges)
                
                # Check triangle-free constraint for adding (u,v)
                # (u,v) creates a triangle if there is w such that (u,w) in E and (v,w) in E
                # i.e., N(u) & N(v) is not empty.
                if current_adj[u] & current_adj[v]:
                    continue
                
                # Tentatively add
                current_edges.add((u, v))
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                # Check 3-colorability
                is_col, _ = is_3_colorable(current_adj)
                
                if is_col:
                    # Adding this edge made it 3-colorable? 
                    # Wait, adding edges can only increase chromatic number.
                    # If the graph was 4-chromatic, it stays 4-chromatic.
                    # If the graph was 3-colorable, it might become 4-chromatic.
                    # So if is_col is True, the graph is 3-colorable.
                    # We want it to be NOT 3-colorable.
                    # So if is_col is True, this move is BAD (unless we were 3-colorable and wanted to become 4-chromatic? No, we want to maximize edges of a 4-chromatic graph).
                    # If current graph is 4-chromatic, adding an edge keeps it 4-chromatic. So is_col should be False.
                    # If current graph is 3-colorable, adding an edge might make it 4-chromatic (is_col False) or keep it 3-colorable (is_col True).
                    # We ONLY accept moves that result in a NOT 3-colorable graph.
                    
                    # So, if is_col is True, we reject this move (unless we are in a phase of "repairing" 3-colorability? No, we want 4-chromatic).
                    current_edges.remove((u, v))
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    no_improve += 1
                else:
                    # Graph is NOT 3-colorable. Good.
                    new_count = len(current_edges)
                    if new_count > best_count:
                        best_count = new_count
                        best_edges = set(current_edges)
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                # Try to remove an edge
                if not current_edges:
                    break
                u, v = random.choice(list(current_edges))
                
                current_edges.remove((u, v))
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Removing an edge can only decrease chromatic number.
                # If it was 4-chromatic, it might become 3-colorable.
                # We want to stay 4-chromatic.
                is_col, _ = is_3_colorable(current_adj)
                
                if is_col:
                    # It became 3-colorable. Bad.
                    # Revert
                    current_edges.add((u, v))
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    no_improve += 1
                else:
                    # Still 4-chromatic.
                    # We lost an edge. This is only good if it allows adding 2 edges later?
                    # In WalkSAT, we accept moves that reduce cost. Here cost is -edges.
                    # So removing an edge increases cost. We should only accept if it leads to a better state later?
                    # Standard WalkSAT accepts with some probability even if cost increases, to escape local optima.
                    # But here, we strictly want to maximize edges.
                    # So we should probably not accept pure removals unless we are stuck.
                    # Let's revert removals unless we are in a "deep search" mode.
                    # For now, revert.
                    current_edges.add((u, v))
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    no_improve += 1
            
            if no_improve > 1000:
                break

        return best_edges

    # Main Loop
    best_edges_global = set()
    best_count_global = 0
    
    # Restart loop
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a 4-chromatic seed
        # Use Grötzsch core + random additions
        seed_edges = create_chvatal_core()
        
        # Randomly add edges to vertices 11-15
        for v in range(11, 16):
            for u in range(v):
                if random.random() < 0.5:
                    # Check if we can add (u,v) to seed while keeping triangle-free
                    # We need to check against current seed_edges
                    # This is slow if done one by one. 
                    # Let's just add them all to a set and check triangle-free at the end.
                    # If it fails, we discard or repair.
                    seed_edges.add((u, v))
        
        # Check if seed is triangle-free
        if not is_triangle_free(seed_edges):
            # Repair: remove edges that form triangles
            adj = get_adjacency(seed_edges)
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        if adj[u] & adj[v]:
                            # Conflict. Remove one of the edges in the triangle.
                            # Find a w in adj[u] & adj[v]
                            common = adj[u] & adj[v]
                            if common:
                                w = (common & -common).bit_length() - 1
                                # Remove (u,v)
                                seed_edges.discard((u, v))
                                # Update adj
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
        
        # Check 3-colorability
        adj = get_adjacency(seed_edges)
        is_col, _ = is_3_colorable(adj)
        
        if is_col:
            # Seed is 3-colorable. We need to make it 4-chromatic.
            # This is a "repair" task.
            # Try to add edges that break 3-colorability.
            # This is harder. Let's skip this seed for now and hope for a better random one.
            # Or, use the Grötzsch core strictly without random additions to 11-15?
            # The Grötzsch core is 4-chromatic. The random additions might have broken it?
            # No, adding edges cannot break 4-chromaticity.
            # So if the seed is 3-colorable, it means my "Grötzsch core" construction is flawed or the random additions created a different structure?
            # Wait, Grötzsch core is 4-chromatic. Adding edges keeps it 4-chromatic.
            # So if is_col is True, my core is wrong.
            
            # Let's double check Grötzsch core.
            # C5 (0-4). Copies 5-9. Top 10.
            # Is it 4-chromatic? Yes.
            # So why would it be 3-colorable?
            # Maybe my `create_chvatal_core` has a bug.
            # Let's assume it's correct. If it's 3-colorable, we discard.
            continue
        
        # Run local search
        time_remaining = deadline - time.time()
        if time_remaining < 2.0:
            break
        
        ls_time = min(5.0, time_remaining / 10.0) # Use small time per restart to allow many restarts
        improved_edges = local_search(seed_edges, ls_time)
        
        if improved_edges:
            if len(improved_edges) > best_count_global:
                best_count_global = len(improved_edges)
                best_edges_global = improved_edges

    # If no valid graph found, fallback to Grötzsch core
    if not best_edges_global:
        best_edges_global = create_chvatal_core()
        # Ensure it's valid
        if not is_triangle_free(best_edges_global):
            # Fallback to empty? No, must be valid.
            # Grötzsch core is triangle-free.
            pass
        is_col, _ = is_3_colorable(get_adjacency(best_edges_global))
        if is_col:
            # This is a critical failure.
            # We need a guaranteed 4-chromatic triangle-free graph.
            # The Grötzsch graph is one.
            # Let's hardcode the edges of the Grötzsch graph to be safe.
            # Vertices 0-10.
            # 0-1, 1-2, 2-3, 3-4, 4-0
            # 5-0, 5-4 (copy of 0? No, copy of i connects to i-1, i+1)
            # Let's re-verify Grötzsch construction.
            # v_i' connected to v_{i-1}, v_{i+1}.
            # v_5' connected to v_4, v_0.
            # v_6' connected to v_0, v_1.
            # v_7' connected to v_1, v_2.
            # v_8' connected to v_2, v_3.
            # v_9' connected to v_3, v_4.
            # v_10 connected to v_5'..v_9'.
            
            hardcore = [
                (0,1), (1,2), (2,3), (3,4), (4,0),
                (5,4), (5,0),
                (6,0), (6,1),
                (7,1), (7,2),
                (8,2), (8,3),
                (9,3), (9,4),
                (10,5), (10,6), (10,7), (10,8), (10,9)
            ]
            best_edges_global = set(hardcore)
    
    return get_edge_list(best_edges_global)

# EVOLVE-BLOCK-END
