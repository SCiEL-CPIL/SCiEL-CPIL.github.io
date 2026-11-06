import time
import random
import numpy as np

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy: 
    1. The rainbow condition on 4-cycles is trivially satisfied if the graph is C4-free.
    2. We aim to find a C4-free graph with the maximum number of edges.
    3. Instead of pure random sampling, we use a greedy/heuristic construction:
       - Start with a sparse random graph.
       - Iteratively add random edges.
       - If adding an edge creates a 4-cycle, we do not add it.
       - To escape local optima, we use random restarts and potentially 
         "swap" operations or simply rely on many fast restarts with 
         a very efficient C4 check.
    4. We use numpy bitmasks for extremely fast common-neighbor checks.
    """
    n = 20
    max_time = 145.0
    start_time = time.time()
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_possible = len(all_edges)
    
    # Adjacency matrix as bitmasks for fast intersection
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def is_c4_free_fast():
        """
        Checks if the current graph (defined by adj bitmasks) is C4-free.
        A 4-cycle exists if there are two vertices u, w with at least 2 common neighbors.
        We check all pairs (u, w).
        """
        for u in range(n):
            for w in range(u + 1, n):
                # Common neighbors are bits set in both adj[u] and adj[w]
                common = adj[u] & adj[w]
                # Check if number of bits set is > 1
                # Using bit_count is fast in Python 3.8+
                if common.bit_count() > 1:
                    return False
        return True
        
    def count_edges():
        # Sum of bits in adj[u] for u in range(n) / 2
        total_bits = sum(adj[u].bit_count() for u in range(n))
        return total_bits // 2

    best_edge_count = 0
    best_adj_state = None
    
    # We will try to construct a dense C4-free graph.
    # The theoretical max for C4-free on 20 vertices is around 60-70.
    # We use a greedy "try to add" approach with randomization.
    
    # To ensure we finish, we limit the number of restarts or time per restart.
    # Each restart:
    # 1. Clear graph.
    # 2. Add random edges until no more can be added without creating C4? 
    #    No, that's greedy and gets stuck.
    # Better: 
    # 1. Start with a random set of edges.
    # 2. Check if C4-free. If yes, record.
    # 3. Try to improve by adding edges? 
    #    
    # Let's stick to a robust "Random Restart with Greedy Insertion"
    # 1. Shuffle all edges.
    # 2. Iterate through edges.
    # 3. Try to add edge.
    # 4. If it creates a C4, skip it.
    # This yields a maximal C4-free subgraph of the random ordering.
    # By shuffling, we get different maximal subgraphs.
    
    # Optimization: We can check "does adding (u,v) create a C4?" in O(n) or O(1) with bitmasks?
    # Adding (u,v) creates a C4 if u and v have a common neighbor w, AND that common neighbor 
    # is part of a path of length 2 between u and v? 
    # Actually, adding (u,v) creates a 4-cycle if there is a path u-x-y-v of length 3.
    # This is equivalent to: u and v have a common neighbor? No.
    # A 4-cycle involving new edge (u,v) is u-v-w-x-u.
    # This requires w to be neighbor of v and x to be neighbor of u, and w-x edge exists.
    # So, adding (u,v) creates a C4 if there exists w in N(v) and x in N(u) such that (w,x) is an edge.
    # We can check this fast:
    # Let N_v = adj[v] (before adding u)
    # Let N_u = adj[u] (before adding v)
    # We need to check if any w in N_v is connected to any x in N_u.
    # i.e., is (adj[w] & N_u) non-zero for any w in N_v?
    
    def would_create_c4(u, v):
        """
        Returns True if adding edge (u,v) to the current graph creates a 4-cycle.
        Assumes (u,v) is not currently an edge.
        """
        # Potential cycles: u - v - w - x - u
        # w must be neighbor of v.
        # x must be neighbor of u.
        # w and x must be adjacent.
        
        # Get neighbors of v (excluding u, but u is not neighbor yet)
        # Get neighbors of u (excluding v, but v is not neighbor yet)
        
        # We iterate over neighbors of v. 
        # To make it faster, we can iterate over the smaller set.
        deg_v = adj[v].bit_count()
        deg_u = adj[u].bit_count()
        
        if deg_v < deg_u:
            # Iterate over neighbors of v
            w_mask = adj[v]
            while w_mask:
                # Get lowest set bit
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                # Check if w is connected to any neighbor of u
                # Neighbors of u are in adj[u]
                # If adj[w] & adj[u] is not 0, then there is a common neighbor x
                # which forms u-x-w-v-u? 
                # Wait. The cycle is u-v-w-x-u.
                # Edges: (u,v) new, (v,w) exists, (w,x) exists, (x,u) exists.
                # So yes, if w (neighbor of v) is connected to any x (neighbor of u), we have a C4.
                if adj[w] & adj[u]:
                    return True
                w_mask &= w_mask - 1
        else:
            # Iterate over neighbors of u
            x_mask = adj[u]
            while x_mask:
                lsb = x_mask & -x_mask
                x = lsb.bit_length() - 1
                if adj[x] & adj[v]:
                    return True
                x_mask &= x_mask - 1
                
        return False

    # Main Loop: Random Restarts with Greedy Construction
    # We try many random orderings.
    
    while time.time() - start_time < max_time:
        # Reset
        adj = [0] * n
        
        # Shuffle edges
        random.shuffle(all_edges)
        
        # Greedy add
        for u, v in all_edges:
            # Check if edge already exists (shouldn't in this pass, but safe)
            if (adj[u] >> v) & 1:
                continue
            if not would_create_c4(u, v):
                add_edge(u, v)
        
        # End of greedy pass. We have a maximal C4-free subgraph.
        current_count = count_edges()
        
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_adj_state = adj[:]
            
            # If we hit a very high number, we might stop early?
            # Max possible for C4-free n=20 is ~ 60-70. 
            # If we get > 65, it's likely near optimal.
            if best_edge_count > 65:
                # We can continue a bit to see if we can go higher, 
                # but probability of improvement drops.
                # For now, let's just keep the best and continue if time allows.
                pass

    # If we didn't find anything (shouldn't happen), return star
    if best_adj_state is None:
        return [[0, v, 0] for v in range(1, 20)]
    
    # Reconstruct edge list from best_adj_state
    final_edges = []
    for u in range(n):
        mask = best_adj_state[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                final_edges.append([u, v, 0])
            mask &= mask - 1
            
    return final_edges

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
