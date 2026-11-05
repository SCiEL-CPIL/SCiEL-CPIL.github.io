import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before the 150s kill
    
    n = 20
    rng = random.Random(42)
    
    def bit_count(x):
        return x.bit_count()
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph adj keeps it valid.
        Assumes (u,v) is not currently an edge.
        """
        for y in range(n):
            if y == u or y == v:
                continue
            ny = adj[y]
            # Check if v is neighbor of y
            if (ny >> v) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    # Check if v is neighbor of z
                    if (nz >> v) & 1:
                        # v is common neighbor of y and z.
                        # Adding (u,v) makes v a common neighbor of u,y,z?
                        # Common neighbors of (u,y,z) are N(u) & N(y) & N(z).
                        # N(u) gains v.
                        # New intersection = (N(u) & N(y) & N(z)) | ({v} & N(y) & N(z)).
                        # Since v is in N(y) and N(z), the second term is {v}.
                        # So count increases by 1.
                        # We need new count < 3 => old count <= 1.
                        if bit_count(adj[u] & ny & nz) > 1:
                            return False
            
            # Check if u is neighbor of y
            if (ny >> u) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    # Check if u is neighbor of z
                    if (nz >> u) & 1:
                        # u is common neighbor of y and z.
                        # Adding (u,v) makes u a common neighbor of v,y,z?
                        # Common neighbors of (v,y,z) are N(v) & N(y) & N(z).
                        # N(v) gains u.
                        # New intersection = (N(v) & N(y) & N(z)) | ({u} & N(y) & N(z)).
                        # Since u is in N(y) and N(z), second term is {u}.
                        # Count increases by 1.
                        # Need old count <= 1.
                        if bit_count(adj[v] & ny & nz) > 1:
                            return False
        
        return True

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_valid_graph(adj):
        """
        Check if the graph is K3,3-free.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    continue
                for w in range(v + 1, n):
                    # Check triple (u, v, w)
                    # Common neighbors: adj[u] & adj[v] & adj[w]
                    # If count >= 3, then there is a K3,3
                    if bit_count(adj[u] & adj[v] & adj[w]) >= 3:
                        return False
        return True

    def repair(adj):
        """
        Remove edges until the graph is valid.
        Returns the number of edges removed.
        """
        removed = 0
        changed = True
        while changed:
            changed = False
            # Find a violation
            for u in range(n):
                if changed:
                    break
                for v in range(u + 1, n):
                    if changed:
                        break
                    if not ((adj[u] >> v) & 1):
                        continue
                    for w in range(v + 1, n):
                        # Check triple (u, v, w)
                        cn = adj[u] & adj[v] & adj[w]
                        if bit_count(cn) >= 3:
                            # Find an edge to remove to break this K3,3
                            # We can remove any edge among the 6 edges of the K3,3
                            # Let's pick one randomly or systematically
                            # The K3,3 has vertices {u,v,w} and {c1,c2,c3} where ci are common neighbors
                            # Edges are (u,ci), (v,ci), (w,ci)
                            # Let's remove (u, c1)
                            cn_list = [i for i in range(n) if (cn >> i) & 1]
                            if cn_list:
                                c1 = cn_list[0]
                                remove_edge(adj, u, c1)
                                removed += 1
                                changed = True
                                break
        return removed

    best_edges = 0
    best_adj = None
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # --- Phase 1: Greedy Construction to get a good starting point ---
    adj = [0] * n
    rng.shuffle(all_edges)
    for (u, v) in all_edges:
        if is_valid_add(adj, u, v):
            add_edge(adj, u, v)
    
    cur_edges = count_edges(adj)
    if cur_edges > best_edges:
        best_edges = cur_edges
        best_adj = [a for a in adj]
        
    # --- Phase 2: Simulated Annealing ---
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    alpha = 0.995
    
    T = T_start
    current_adj = [a for a in adj]
    current_edges = cur_edges
    
    # Precompute edge indices for faster access? 
    # We have 190 edges.
    # We can just iterate over all_edges.
    
    steps_per_temp = 200
    total_steps = 0
    max_steps = 500000 # Safety cap
    
    while T > T_end and time.time() < deadline - 10 and total_steps < max_steps:
        # Try a random move
        # 1. Add an edge
        # 2. Remove an edge
        # 3. Swap an edge (remove one, add another)
        
        move_type = rng.random()
        accepted = False
        
        if move_type < 0.4:
            # Add
            # Pick random non-edge
            attempts = 0
            while attempts < 10:
                u, v = rng.choice(all_edges)
                if not ((current_adj[u] >> v) & 1):
                    break
                attempts += 1
            else:
                continue
            
            if is_valid_add(current_adj, u, v):
                add_edge(current_adj, u, v)
                current_edges += 1
                accepted = True
                # Always accept improvement
            else:
                # Reject (or accept with probability? No, if invalid, we can't add)
                # But in SA, we usually work within a feasible space or with penalties.
                # Here, let's stick to feasible space moves.
                # If we can't add, we don't add.
                pass
                
        elif move_type < 0.8:
            # Remove
            # Pick random edge
            attempts = 0
            while attempts < 10:
                u, v = rng.choice(all_edges)
                if (current_adj[u] >> v) & 1:
                    break
                attempts += 1
            else:
                continue
            
            remove_edge(current_adj, u, v)
            current_edges -= 1
            accepted = True
            # Accept if delta E < 0 or with probability exp(delta E / T)
            delta_e = -1
            if delta_e < 0 or rng.random() < math.exp(delta_e / T):
                pass # Keep removal
            else:
                add_edge(current_adj, u, v)
                current_edges += 1
                accepted = False
                
        else:
            # Swap
            # Remove random edge e1, add random non-edge e2
            attempts = 0
            while attempts < 10:
                u1, v1 = rng.choice(all_edges)
                if (current_adj[u1] >> v1) & 1:
                    break
                attempts += 1
            else:
                continue
            
            attempts = 0
            while attempts < 10:
                u2, v2 = rng.choice(all_edges)
                if not ((current_adj[u2] >> v2) & 1) and (u2, v2) != (u1, v1) and (u2, v2) != (v1, u1):
                    break
                attempts += 1
            else:
                continue
            
            remove_edge(current_adj, u1, v1)
            if is_valid_add(current_adj, u2, v2):
                add_edge(current_adj, u2, v2)
                # Edge count stays same
                delta_e = 0
                accepted = True
                # Always accept same energy (or with prob 1)
            else:
                # Revert
                add_edge(current_adj, u1, v1)
                accepted = False
        
        if accepted and current_edges > best_edges:
            best_edges = current_edges
            best_adj = [a for a in current_adj]
            
        total_steps += 1
        T *= alpha
        
        # Periodically repair? 
        # Since we only add valid edges and remove edges, the graph should always be valid.
        # Wait, does removing an edge ever make it invalid? No, removing edges cannot create a K3,3.
        # Does adding a valid edge keep it valid? Yes, by definition of is_valid_add.
        # So the state is always valid.
        
    # --- Phase 3: Final Local Search Polish ---
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Add-only
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if (current_adj[u] >> v) & 1:
                continue
            if is_valid_add(current_adj, u, v):
                add_edge(current_adj, u, v)
                current_edges += 1
                improved = True
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [a for a in current_adj]

    if best_adj is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
