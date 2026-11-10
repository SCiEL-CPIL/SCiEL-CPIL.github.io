import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Build adjacency masks
    # adj_mask[i] is an integer where bit j is 1 if i and j are connected (lcm(i,j) <= n)
    adj_mask = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
        if time.time() > deadline:
            # Fallback
            return [1, 2]

    # Initial best clique: Divisors of 504
    # 504 = 2^3 * 3^2 * 7
    best_mask = 0
    best_size = 0
    for k in range(1, 513):
        if 504 % k == 0:
            best_mask |= (1 << k)
            best_size += 1
            
    def popcount(x):
        return x.bit_count()

    def tomita_max_clique(adj_mask, deadline, initial_best_size, initial_best_mask):
        """
        Implements Tomita's Maximum Clique Algorithm with Greedy Coloring.
        Uses bitmasks for efficiency.
        """
        best_size_local = initial_best_size
        best_mask_local = initial_best_mask
        
        # Helper to get list of vertices from mask
        def get_vertices(mask):
            verts = []
            while mask:
                bit = mask & -mask
                verts.append(bit.bit_length() - 1)
                mask ^= bit
            return verts
        
        def expand(R_mask, P_mask, X_mask):
            nonlocal best_size_local, best_mask_local
            
            if time.time() > deadline:
                return
                
            if P_mask == 0:
                if X_mask == 0:
                    r_size = popcount(R_mask)
                    if r_size > best_size_local:
                        best_size_local = r_size
                        best_mask_local = R_mask
                return

            # Pruning: if max possible size <= current best, return
            if popcount(R_mask) + popcount(P_mask) <= best_size_local:
                return

            # Get vertices in P
            P_verts = get_vertices(P_mask)
            
            # Greedy Coloring on P
            # We assign colors 1..k such that vertices of same color are not adjacent.
            # This gives an upper bound on clique size in P.
            # color[i] is the color assigned to P_verts[i]
            # We also want to order P_verts by color to help branching
            colors = [0] * len(P_verts)
            color_map = {} # color -> list of indices in P_verts
            
            current_max_color = 0
            uncolored = set(range(len(P_verts)))
            
            # Simple greedy coloring:
            # For c = 1, 2, ...
            #   Find independent set in remaining uncolored vertices
            #   Assign color c to them
            #   Remove them from uncolored
            #   If all colored, break
            
            while uncolored:
                current_max_color += 1
                # Try to find a maximal independent set in uncolored
                # Greedy: pick vertices that are not adjacent to each other
                independent = []
                # Iterate through uncolored vertices
                # To make it deterministic and fast, we just check against previously picked in this color
                remaining = list(uncolored)
                for idx in remaining:
                    v = P_verts[idx]
                    is_independent = True
                    # Check if v is adjacent to any vertex already in 'independent'
                    for other_idx in independent:
                        other_v = P_verts[other_idx]
                        # Check adjacency
                        # adj_mask[v] has bit other_v set if connected
                        if (adj_mask[v] >> other_v) & 1:
                            is_independent = False
                            break
                    if is_independent:
                        independent.append(idx)
                
                if not independent:
                     # Should not happen if uncolored is not empty, 
                     # unless graph has no edges, but we pick at least one vertex
                     # If independent is empty, it means every vertex is adjacent to someone picked?
                     # No, the first vertex picked is always independent.
                     # So this case only if uncolored was empty, but loop condition prevents that.
                     pass
                
                for idx in independent:
                    colors[idx] = current_max_color
                    uncolored.remove(idx)
                    
            # Now P_verts are colored 1..current_max_color
            # Reorder P_verts by color (descending) for branching
            # Vertices with higher colors are "better" candidates first?
            # Tomita suggests ordering by color.
            # We will iterate from highest color to lowest.
            
            # Create a list of (color, vertex)
            color_verts = [(colors[i], P_verts[i]) for i in range(len(P_verts))]
            
            # Sort by color descending, then vertex id for stability
            color_verts.sort(key=lambda x: (-x[0], -x[1]))
            
            # Branching loop
            # We maintain the set of available vertices for the next level.
            # Instead of passing X_mask explicitly in a complex way, 
            # Tomita's method often just passes P and R, and relies on the fact 
            # that we are finding maximal cliques. 
            # However, standard BK with pivot/X is safer for completeness if not careful.
            # Let's stick to the structure: 
            # For each vertex v in color_verts (ordered by color):
            #   If popcount(R) + color_of_v <= best_size: break
            #   Expand with v
            
            for color_val, v in color_verts:
                if popcount(R_mask) + color_val <= best_size_local:
                    break
                
                if time.time() > deadline:
                    return

                bit_v = 1 << v
                
                # Check if v is still in P (it should be, as we built list from P)
                # Recurse
                # New R = R U {v}
                # New P = P ∩ N(v)
                # New X = X ∩ N(v)
                
                R_new = R_mask | bit_v
                P_new = P_mask & adj_mask[v]
                X_new = X_mask & adj_mask[v]
                
                expand(R_new, P_new, X_new)
                
                # Update P and X for the next iteration in this loop
                # P = P \ {v}
                P_mask ^= bit_v
                # X = X U {v}
                X_mask |= bit_v
                
            # Note: The standard Tomita implementation usually manages P and X 
            # slightly differently to ensure all cliques are found without 
            # redundant work, but this recursive structure with explicit 
            # P/X updates mimics Bron-Kerbosch with the coloring bound.
            
        # Initial call
        all_nodes_mask = 0
        for i in range(1, n + 1):
            all_nodes_mask |= (1 << i)
            
        expand(0, all_nodes_mask, 0)
        
        return best_size_local, best_mask_local

    best_size, best_mask = tomita_max_clique(adj_mask, deadline, best_size, best_mask)
    
    # Reconstruct list from best_mask
    result = []
    temp_res = best_mask
    while temp_res:
        bit = temp_res & -temp_res
        temp_res ^= bit
        result.append(bit.bit_length() - 1)
        
    if not result:
        return [1]
        
    return result
