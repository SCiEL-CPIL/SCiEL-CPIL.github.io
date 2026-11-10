import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n = 16
    
    # Precompute pairs
    pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Bitmask utilities
    def is_3_colorable(adj):
        """
        Fast 3-colorability check using backtracking with bitsets.
        Returns True if the graph is 3-colorable.
        """
        # Order vertices by degree descending to prune early
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        # Map original vertex to position in order for faster lookup if needed, 
        # but we just iterate through 'order'
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Find forbidden colors
            forbidden = 0
            # Iterate over neighbors of v
            # adj[v] is a bitmask
            neighbors = adj[v]
            while neighbors:
                lsb = neighbors & -neighbors
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    forbidden |= (1 << colors[u])
                neighbors -= lsb
            
            # Try available colors
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def try_add_edge(adj, u, v):
        """Returns True if edge (u,v) can be added while keeping triangle-free."""
        if (adj[u] >> v) & 1:
            return False
        if adj[u] & adj[v]:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True

    # Strategy:
    # 1. Start with K_{8,8} (partition 0-7 and 8-15). This is triangle-free and has 64 edges.
    # 2. K_{8,8} is 2-colorable, so it is 3-colorable. We need to make it not 3-colorable.
    # 3. We delete edges to break 3-colorability.
    # 4. We use a backtracking search to find a set of edges to delete such that the graph is NOT 3-colorable.
    #    To maximize edges, we want to delete the FEWEST edges.
    # 5. However, deleting edges from K_{8,8} might make it 3-colorable again if we delete the "wrong" ones?
    #    No, K_{8,8} is bipartite. Any subgraph is bipartite and thus 2-colorable, hence 3-colorable.
    #    Wait. This is a critical insight.
    #    ANY subgraph of a bipartite graph is bipartite.
    #    Therefore, ANY triangle-free graph that is a subgraph of K_{8,8} is bipartite and 2-colorable.
    #    So we CANNOT get a 4-chromatic graph by just deleting edges from K_{8,8}.
    #    We MUST add edges within the partitions to break bipartiteness, but that creates triangles!
    
    # Correction:
    # We cannot start with K_{8,8} and just delete edges. We must start with a graph that is NOT bipartite.
    # The Grötzsch graph (11 vertices, 20 edges) is 4-chromatic and triangle-free.
    # We can extend it.
    
    # Alternative Strategy:
    # Use the Mycielski construction or known 4-chromatic triangle-free graphs on 16 vertices.
    # The Clebsch graph has 16 vertices, 48 edges, is triangle-free, and is 3-colorable? 
    # No, Clebsch is 4-chromatic? Let's check.
    # Clebsch graph: 16 vertices, 48 edges. It is the folded 5-cube.
    # Chromatic number of Clebsch graph is 3. (It is a perfect graph? No. But it is 3-colorable).
    # Wait, the Clebsch graph is 3-colorable.
    
    # What is the maximum number of edges in a triangle-free, 4-chromatic graph on 16 vertices?
    # Rademacher-Turan theorem bounds triangle-free edges by n^2/4 = 64.
    # We need a graph close to 64 but not bipartite.
    
    # Let's use a "blow-up" or "subdivision" approach.
    # Or simply: Start with a known 4-chromatic triangle-free graph on 16 vertices if it exists.
    # The Grötzsch graph is 11 vertices.
    # Mycielski of C5 is Grötzsch (11 vertices).
    # Mycielski of Grötzsch is 21 vertices. Too big.
    
    # We can take the Grötzsch graph (11 vertices) and add 5 vertices.
    # To maximize edges, we want the 5 new vertices to be connected to as many existing vertices as possible without creating triangles.
    # If we add a vertex v, it can connect to an independent set in the existing graph.
    
    # Let's implement a local search starting from the Grötzsch core + random connections for the extra 5 vertices, 
    # but with a much stronger "repair" mechanism:
    # If the graph becomes 3-colorable, we try to add edges that break 3-colorability.
    # If we can't add edges (triangle-free constraint), we try to swap edges.
    
    # Better Strategy:
    # 1. Generate the Grötzsch graph on vertices 0-10.
    # 2. For vertices 11-15, we decide their neighbors.
    # 3. We use a SAT-like or backtracking approach to assign neighbors for 11-15 to maximize total edges,
    #    subject to:
    #    a) Triangle-free
    #    b) Not 3-colorable
    
    # Actually, the most robust way is to use the "edge addition" local search but start with a VERY dense 
    # 4-chromatic seed.
    
    # Let's try to construct a dense 4-chromatic triangle-free graph on 16 vertices.
    # One known graph is the "Hajnal graph" or similar?
    # Let's stick to the Grötzsch core + greedy fill for the rest, but improve the local search.
    
    def create_groetzsch_core():
        # Vertices 0-4: Cycle C5
        # Vertices 5-9: Copies of 0-4, connected to i-1, i+1
        # Vertex 10: Connected to 5-9
        adj = [0] * n
        
        # Cycle 0-4
        for i in range(5):
            u, v = i, (i+1)%5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Copies 5-9
        for i in range(5):
            u = 5 + i
            v1 = (i-1)%5
            v2 = (i+1)%5
            adj[u] |= (1 << v1)
            adj[v1] |= (1 << u)
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
            
        # Top vertex 10
        for i in range(5):
            u = 10
            v = 5 + i
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        return adj

    def greedy_fill(adj):
        """Add all possible edges that maintain triangle-free property."""
        changed = True
        while changed:
            changed = False
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if not (adj[u] & adj[v]):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
        return adj

    def local_search(adj, time_limit=10.0):
        """
        Local search to maximize edges while maintaining:
        1. Triangle-free
        2. Not 3-colorable
        """
        current_adj = adj[:]
        current_edges = edge_count(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        if is_3_colorable(current_adj):
            return current_adj, current_edges

        start_ls = time.time()
        no_improve_count = 0
        
        while time.time() - start_ls < time_limit:
            improved = False
            
            # Pick a random edge to remove
            edges_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if (current_adj[u] >> v) & 1:
                        edges_list.append((u, v))
                        
            if not edges_list:
                break
                
            u, v = random.choice(edges_list)
            
            # Remove edge
            remove_edge(current_adj, u, v)
            current_edges -= 1
            
            # Try to add 2 new edges
            pot_u = []
            for w in range(n):
                if w != u and w != v and not ((current_adj[u] >> w) & 1):
                    if not (current_adj[u] & current_adj[w]):
                        pot_u.append(w)
                        
            pot_v = []
            for x in range(n):
                if x != v and x != u and not ((current_adj[v] >> x) & 1):
                    if not (current_adj[v] & current_adj[x]):
                        pot_v.append(x)
                        
            if pot_u and pot_v:
                random.shuffle(pot_u)
                random.shuffle(pot_v)
                
                found = False
                for w in pot_u[:15]:
                    for x in pot_v[:15]:
                        if w == x:
                            continue
                            
                        # Check for triangle (u, w, x)
                        if (current_adj[u] >> x) & 1 and (current_adj[w] >> x) & 1:
                            continue
                            
                        # Check for triangle (v, x, w)
                        if (current_adj[v] >> w) & 1 and (current_adj[x] >> w) & 1:
                            continue
                            
                        # Apply changes
                        current_adj[u] |= (1 << w)
                        current_adj[w] |= (1 << u)
                        current_adj[v] |= (1 << x)
                        current_adj[x] |= (1 << v)
                        
                        new_edges = current_edges + 2
                        
                        # Check 3-colorability
                        if not is_3_colorable(current_adj):
                            if new_edges > best_edges:
                                best_edges = new_edges
                                best_adj = current_adj[:]
                                improved = True
                                found = True
                                break
                        else:
                            remove_edge(current_adj, u, w)
                            remove_edge(current_adj, v, x)
                            current_edges -= 2
                            continue
                            
                        if not found:
                            remove_edge(current_adj, u, w)
                            remove_edge(current_adj, v, x)
                            current_edges -= 2
                            
                    if found:
                        break
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                continue
            
            if not improved:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                no_improve_count += 1
                if no_improve_count > 100:
                    break
            else:
                no_improve_count = 0
                current_edges = best_edges

        return best_adj, best_edges

    # Main Loop
    num_restarts = 500
    best_edges_count = 0
    best_edge_list = []
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start with Grötzsch core
        adj = create_groetzsch_core()
        
        # Randomly connect vertices 11-15 to the core and each other
        for v in range(11, 16):
            # Try to connect to random vertices
            candidates = list(range(11)) # Core vertices
            random.shuffle(candidates)
            for u in candidates:
                if random.random() < 0.5:
                    try_add_edge(adj, v, u)
            # Connect to other new vertices
            for w in range(v+1, 16):
                if random.random() < 0.5:
                    try_add_edge(adj, v, w)
                    
        # Greedy fill
        adj = greedy_fill(adj)
        
        # Check if 4-chromatic
        if is_3_colorable(adj):
            # If it became 3-colorable (unlikely if core is 4-chromatic and we only add edges),
            # it's actually impossible for adding edges to make a 4-chromatic graph 3-colorable.
            # Wait, adding edges can only increase chromatic number.
            # So if we start with a 4-chromatic graph, it stays 4-chromatic.
            # So this check is redundant if the core is valid.
            continue
            
        edge_count = edge_count(adj)
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_edge_list = get_edge_list(adj)
            
        # Local search
        time_remaining = deadline - time.time()
        if time_remaining < 5.0:
            break
            
        ls_time = min(5.0, time_remaining / 2.0)
        adj, improved_count = local_search(adj, time_limit=ls_time)
        
        if improved_count > best_edges_count:
            best_edges_count = improved_count
            best_edge_list = get_edge_list(adj)

    if not best_edge_list:
        # Fallback
        adj = create_groetzsch_core()
        adj = greedy_fill(adj)
        best_edge_list = get_edge_list(adj)
        best_edges_count = edge_count(adj)

    return best_edge_list

# EVOLVE-BLOCK-END
