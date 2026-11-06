"""Rewritten program using an optimized Tomita Max Clique algorithm."""

def construct():
    n = 256
    # Build compatibility graph adjacency masks
    # comp_mask[i] has bit j set if i != j and (i+1)*(j+1) % ((i+1)+(j+1)) != 0
    comp_mask = [0] * n
    
    for a in range(1, n + 1):
        i = a - 1
        for b in range(a + 1, n + 1):
            j = b - 1
            # If they do NOT conflict, they are compatible
            if (a * b) % (a + b) != 0:
                comp_mask[i] |= (1 << j)
                comp_mask[j] |= (1 << i)
                
    best_clique = []
    best_size = 0
    
    def expand(candidates_mask, clique):
        nonlocal best_clique, best_size
        
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Greedy Coloring of the candidates
        # We want to assign colors such that no two adjacent vertices have the same color.
        # The color of a vertex v is an upper bound on the clique size in the subgraph induced by {v} U {u in candidates | color(u) > color(v)}.
        # Actually, in Tomita's algorithm, we color the candidates and then process them in reverse color order.
        
        colored = []
        U = candidates_mask
        color = 0
        
        while U:
            color += 1
            S = U
            while S:
                # Pick a vertex from S. 
                # Heuristic: Pick the one with smallest degree in S to keep colors low? 
                # Or just pick any. Standard Tomita picks arbitrary or by index.
                # Let's pick the lowest index for determinism and speed (bit trick).
                v_bit = S & (-S)
                v = v_bit.bit_length() - 1
                
                colored.append((v, color))
                
                # Remove v from U (it is now colored)
                U &= ~v_bit
                
                # Remove v and its neighbors from S (they cannot have this color)
                S &= ~comp_mask[v]
                S &= ~v_bit
                
        # Process vertices in reverse order of coloring (highest color first)
        # This ensures that when we check the bound, we are using the tightest possible bound for the remaining candidates.
        # Note: The `colored` list is ordered by color 1, 1, 1, 2, 2, ...
        # We iterate from the end.
        
        # We need to maintain the set of candidates that are still "available" for branching.
        # However, the standard Tomita algorithm modifies the candidate set in-place during the loop.
        
        # To do this correctly with bitmasks, we iterate through the colored list.
        # For each vertex v with color c:
        # 1. If len(clique) + c <= best_size, break (prune).
        # 2. Branch on v: new_candidates = current_candidates & comp_mask[v]
        # 3. Remove v from current_candidates for the next iteration.
        
        current_candidates = candidates_mask
        
        for i in range(len(colored) - 1, -1, -1):
            v, c = colored[i]
            
            # Check if v is still in current_candidates
            # (It might have been removed if it was a neighbor of a previously branched vertex? 
            # No, in Tomita, we branch on v, then remove v from the pool for *subsequent* vertices in this loop.
            # The neighbors of v are handled by the recursive call's candidate set.
            # Here, we just remove v itself.)
            if not (current_candidates & (1 << v)):
                continue
                
            # Upper bound check
            if len(clique) + c <= best_size:
                break
                
            # Branch: include v
            new_clique = clique + [v]
            new_candidates = current_candidates & comp_mask[v]
            expand(new_candidates, new_clique)
            
            # Exclude v: remove v from current_candidates
            current_candidates &= ~(1 << v)
            
    # Start with all vertices
    all_mask = (1 << n) - 1
    expand(all_mask, [])
    
    # Convert vertex indices to values (1-based)
    result = [v + 1 for v in best_clique]
    
    if not result:
        result = [1]
        
    return result
