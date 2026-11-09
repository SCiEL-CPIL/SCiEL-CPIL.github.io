"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Keep 10s margin

    # Fallback: Paley graph on 37 vertices
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    # Best known valid graph so far
    best_n, best_edges = get_paley(37)
    best_adj = np.zeros((best_n, best_n), dtype=np.int8)
    for u, v in best_edges:
        best_adj[u, v] = 1
        best_adj[v, u] = 1

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    def count_bad_sets(n, adj):
        """
        Counts K5 and Independent 5-sets.
        Uses vectorized operations where possible, but relies on pair iteration for precision and speed balance.
        """
        # K5 count
        k5_count = 0
        # Precompute common neighbors for all pairs? 
        # A @ A gives path counts.
        # For each edge (u,v), we need triangles in N(u) & N(v).
        # Let's iterate edges.
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    # Common neighbors
                    # Using numpy for intersection of rows
                    # adj[u] is a vector.
                    # common_mask = (adj[u] == 1) & (adj[v] == 1)
                    # common_indices = np.where(common_mask)[0]
                    # This is O(n) per pair. O(n^3) total.
                    # Then count K3 in induced subgraph of common_indices.
                    # For n=42, this is fast.
                    
                    # Optimization: Use bitsets or just list comprehension?
                    # NumPy is faster for large n, but for n=42, Python list might be competitive?
                    # Let's use NumPy.
                    common_mask = adj[u] & adj[v]
                    # Exclude u and v (they are 0 in common mask usually if no self loops, but adj[u,u]=0)
                    # adj[u,u] is 0.
                    common_indices = np.where(common_mask == 1)[0]
                    m = len(common_indices)
                    if m >= 3:
                        # Extract subgraph
                        # sub_adj = np.take(np.take(adj, common_indices, axis=0), common_indices, axis=1)
                        # Count triangles: trace(sub_adj @ sub_adj @ sub_adj) / 6
                        # Or simpler: sum of sub_adj[i,j] * sub_adj[i,k] * sub_adj[j,k] ...
                        # Actually, number of triangles in a graph with adjacency matrix M is trace(M^3)/6.
                        # For small m, this is fast.
                        sub_adj = np.take(np.take(adj, common_indices, axis=0), common_indices, axis=1)
                        # trace(M^3)
                        m2 = sub_adj @ sub_adj
                        m3 = m2 @ sub_adj
                        # trace is sum of diagonal
                        tr = np.trace(m3)
                        k5_count += tr // 6
        
        # Ind5 count
        # Independent 5-set in G is K5 in complement of G.
        # Comp adj = 1 - adj - I
        # We can just call count_bad_sets on the complement?
        # But we need to exclude self loops.
        comp_adj = 1 - adj
        np.fill_diagonal(comp_adj, 0)
        
        ind5_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if comp_adj[u, v] == 1: # Non-edge in G
                    common_mask = comp_adj[u] & comp_adj[v]
                    common_indices = np.where(common_mask == 1)[0]
                    m = len(common_indices)
                    if m >= 3:
                        sub_adj = np.take(np.take(comp_adj, common_indices, axis=0), common_indices, axis=1)
                        m2 = sub_adj @ sub_adj
                        m3 = m2 @ sub_adj
                        tr = np.trace(m3)
                        ind5_count += tr // 6
                        
        return k5_count + ind5_count

    def search(n, time_limit):
        # Initialize
        # Try starting from Paley(37) extended?
        # If n == 37, use Paley.
        # If n > 37, start with random or extend Paley.
        # Extending Paley: Add random vertices.
        # Let's just use random init for n != 37, or Paley for n=37.
        # Actually, a good heuristic is to start with a random graph with p=0.5.
        
        adj = np.zeros((n, n), dtype=np.int8)
        if n == 37:
            # Use Paley
            residues = {(x * x) % n for x in range(1, n)}
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i) % n in residues:
                        adj[i, j] = 1
                        adj[j, i] = 1
        else:
            # Random init
            # np.random.rand(n, n) < 0.5
            # Symmetrize
            rand_mat = np.random.rand(n, n)
            adj = (rand_mat < 0.5).astype(np.int8)
            adj = (adj + adj.T) // 2 # Ensure symmetry
            np.fill_diagonal(adj, 0)
            
        current_cost = count_bad_sets(n, adj)
        if current_cost == 0:
            return adj_to_edges(n, adj)

        # Local Search
        # We will use a simple hill-climbing with occasional random restarts or "simulated annealing" style acceptance.
        # Given the speed of count_bad_sets (likely <1ms for n=42), we can do many steps.
        
        no_improve_count = 0
        max_no_improve = 5000 # Restart if stuck
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                # Random restart
                adj = np.zeros((n, n), dtype=np.int8)
                rand_mat = np.random.rand(n, n)
                adj = (rand_mat < 0.5).astype(np.int8)
                adj = (adj + adj.T) // 2
                np.fill_diagonal(adj, 0)
                current_cost = count_bad_sets(n, adj)
                if current_cost == 0:
                    return adj_to_edges(n, adj)
                no_improve_count = 0
                max_no_improve *= 1.5 # Increase patience? Or keep same?
                
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            old_cost = current_cost
            # Flip
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
            
            new_cost = count_bad_sets(n, adj)
            
            if new_cost <= old_cost:
                current_cost = new_cost
                no_improve_count = 0
                if current_cost == 0:
                    return adj_to_edges(n, adj)
            else:
                # Revert
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                no_improve_count += 1
                
        return None

    # Try sizes
    # 42 is the target.
    
    time_remaining = deadline - time.time()
    
    # Try 42
    if time_remaining > 30:
        alloc = min(150, time_remaining - 10)
        res = search(42, time.time() + alloc)
        if res:
            return 42, res
    
    # Try 41
    time_remaining = deadline - time.time()
    if time_remaining > 30:
        alloc = min(100, time_remaining - 10)
        res = search(41, time.time() + alloc)
        if res:
            return 41, res
            
    # Try 40
    time_remaining = deadline - time.time()
    if time_remaining > 20:
        alloc = min(60, time_remaining - 10)
        res = search(40, time.time() + alloc)
        if res:
            return 40, res
            
    # Try 39
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        alloc = min(30, time_remaining - 5)
        res = search(39, time.time() + alloc)
        if res:
            return 39, res

    # Fallback
    return best_n, best_edges

# EVOLVE-BLOCK-END
