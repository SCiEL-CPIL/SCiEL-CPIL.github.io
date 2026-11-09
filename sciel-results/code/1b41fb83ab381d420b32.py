import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a K2,3-free graph on 20 vertices maximizing edges.
    
    Strategy:
    1. Use a different structural starting point: the Paley graph P(19) or similar
       dense structures, combined with a complete graph K4, to break the symmetry
       of the disjoint union of K4s.
    2. Use a fast bit-parallel local search with 1-swap (edge swap) and 2-swap moves
       to escape local optima that greedy addition cannot.
    3. The key insight is that the disjoint union of K4s (30 edges) might not be
       optimal for n=20. By allowing edge swaps (removing one edge, adding another),
       we can restructure the graph into potentially denser configurations.
       
    Why this is different:
    - The current approach only adds edges (monotonic). This approach allows
      removing edges to enable adding more edges later (non-monotonic search).
    - Uses a different initial seed (mix of structures) to avoid the K4-union basin.
    - Employs simulated annealing to accept temporary degradations in edge count.
    """
    n = 20
    start_time = time.time()
    time_limit = 120.0  # Hard limit, with safety margin
    
    # Bitset representation: adj[i] is an integer where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    
    # Initial Seed: Try a different structure.
    # Let's start with a random graph and then optimize.
    # Or, better: start with a known dense graph that is K2,3-free but not 5xK4.
    # For example, the complete bipartite graph K_{10,10} is NOT K2,3-free (it contains many).
    # The Paley graph P(19) on 19 vertices + 1 isolated vertex? P(19) has 19*9/2 = 85.5? No, degree 9.
    # P(19) is K2,3-free? Paley graphs are strongly regular (19, 9, 4, 5).
    # lambda = 4 means adjacent vertices have 4 common neighbors. This violates K2,3-free (needs <= 2).
    # So Paley is not K2,3-free.
    
    # Let's stick to the K4-union but perturb it.
    # Or start with a random graph and let the optimizer find the structure.
    
    # Strategy: Random restarts with Simulated Annealing.
    
    def count_edges(adj_list):
        return sum(bin(x).count('1') for x in adj_list) // 2
    
    def is_k23_free(adj_list):
        """
        Check if graph is K2,3-free.
        K2,3 exists if there are two vertices u, v with at least 3 common neighbors.
        """
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                if bin(common).count('1') >= 3:
                    return False
        return True
    
    def valid_add(adj_list, u, v):
        """
        Check if adding edge (u,v) keeps the graph K2,3-free.
        """
        if adj_list[u] & (1 << v):
            return False
        
        # 1. Common neighbors of u and v must be <= 2
        common_uv = adj_list[u] & adj_list[v]
        if bin(common_uv).count('1') > 2:
            return False
        
        # 2. For any w in N(u), common neighbors of w and v must be <= 1 (since u becomes common)
        #    Wait, if we add (u,v), then u is a neighbor of v.
        #    For any w in N(u), u is a common neighbor of w and v?
        #    No. u is connected to w. If we add (u,v), u is connected to v.
        #    So u is a common neighbor of w and v ONLY IF u is connected to w (true) and u is connected to v (true after add).
        #    So yes, u becomes a common neighbor of w and v.
        #    The existing common neighbors of w and v are N(w) & N(v).
        #    Total common neighbors of w and v will be |N(w) & N(v)| + 1 (for u).
        #    This must be <= 2. So |N(w) & N(v)| must be <= 1.
        
        neighbors_u = adj_list[u]
        temp = neighbors_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            
            common_wv = adj_list[w] & adj_list[v]
            if bin(common_wv).count('1') > 1:
                return False
        
        # 3. Symmetrically, for any w in N(v), common neighbors of w and u must be <= 1.
        neighbors_v = adj_list[v]
        temp = neighbors_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            
            common_wu = adj_list[w] & adj_list[u]
            if bin(common_wu).count('1') > 1:
                return False
        
        return True
    
    def valid_remove(adj_list, u, v):
        """
        Removing an edge always preserves K2,3-freeness if it was valid before.
        """
        return bool(adj_list[u] & (1 << v))
    
    def add_edge(adj_list, u, v):
        adj_list[u] |= (1 << v)
        adj_list[v] |= (1 << u)
        
    def remove_edge(adj_list, u, v):
        adj_list[u] &= ~(1 << v)
        adj_list[v] &= ~(1 << u)
        
    def random_graph():
        """Generate a random K2,3-free graph by greedily adding random edges."""
        adj_rand = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if valid_add(adj_rand, u, v):
                add_edge(adj_rand, u, v)
        return adj_rand
    
    best_edges = 0
    best_adj = [0] * n
    
    # Multiple random restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Start with a random valid graph
        adj = random_graph()
        current_edges = count_edges(adj)
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while T > T_min and (time.time() - start_time < time_limit - 5.0):
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random non-edge
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                if not non_edges:
                    T *= cooling
                    continue
                    
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Reject
                    pass
                    
            elif move_type < 0.6:
                # Try to remove a random edge
                edges = []
                for u in range(n):
                    temp = adj[u]
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append((u, v))
                        temp -= lsb
                            
                if not edges:
                    T *= cooling
                    continue
                    
                u, v = random.choice(edges)
                # Removing an edge is always valid (preserves K2,3-free)
                # But we only accept if it leads to a better state later?
                # In SA, we accept with probability based on energy change.
                # Energy = -edges. Removing edge increases energy (bad).
                # Delta E = 1. Accept with prob exp(-1/T).
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1
                # Else reject
                    
            else:
                # Try a swap: remove (u1, v1), add (u2, v2)
                # This is complex to sample uniformly. Let's stick to add/remove for now.
                # Actually, swaps are crucial for escaping local optima.
                # Let's implement a simple swap: pick a random edge to remove, and a random non-edge to add.
                edges = []
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append((u, v))
                        else:
                            non_edges.append((u, v))
                            
                if not edges or not non_edges:
                    T *= cooling
                    continue
                    
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                
                # Check if swap is valid:
                # 1. Remove (u1, v1)
                # 2. Add (u2, v2)
                
                # We need to check if adding (u2, v2) is valid in the graph where (u1, v1) is removed.
                # This requires a temporary check.
                
                # Save state
                old_adj = adj[:]
                
                # Remove
                remove_edge(adj, u1, v1)
                
                # Check add
                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    # Edge count unchanged
                    # Check if this new state is better? It's the same count.
                    # But it might allow future additions.
                    # In SA, we accept neutral moves with prob 1 or exp(0)=1.
                    # So we accept it.
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Revert
                    adj = old_adj
                    
            T *= cooling
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                
    # Extract edges from the best graph found
    edges = []
    for u in range(n):
        temp = best_adj[u]
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            temp -= lsb
            
    return edges
# EVOLVE-BLOCK-END
