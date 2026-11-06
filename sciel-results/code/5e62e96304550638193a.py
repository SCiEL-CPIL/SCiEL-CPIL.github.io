"""Rewritten program using Tomita Max Clique algorithm on the complement graph."""
import sys

def construct():
    n = 256
    # Build conflict graph adjacency masks
    # conflict_mask[i] has bit j set if (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    conflict_mask = [0] * n
    
    for a in range(1, n + 1):
        i = a - 1
        for b in range(a + 1, n + 1):
            j = b - 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Build complement graph adjacency masks
    # An edge in complement graph means NO conflict.
    # comp_mask[i] has bit j set if i != j and no conflict between i and j.
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        # All vertices except i, minus those that conflict with i
        comp_mask[i] = all_mask & ~(1 << i) & ~conflict_mask[i]
        
    # Pre-compute vertex ordering for coloring
    # Sort by degree in complement graph (descending)
    vertex_order = sorted(range(n), key=lambda i: comp_mask[i].bit_count(), reverse=True)
    
    best_clique = []
    best_size = 0
    
    def color_sort(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask in the complement graph.
        Returns a list of (vertex, color) sorted by color descending.
        """
        if candidates_mask == 0:
            return []
        
        colored = []
        Q = candidates_mask
        k = 0
        
        while Q:
            k += 1
            S = Q
            while S:
                # Pick a vertex. Heuristic: pick the first one in vertex_order present in S.
                v = -1
                for u in vertex_order:
                    if S & (1 << u):
                        v = u
                        break
                
                if v == -1:
                    break # Should not happen if S != 0
                
                v_bit = 1 << v
                
                colored.append((v, k))
                
                # Remove v from Q
                Q &= ~v_bit
                
                # Remove v and its neighbors in complement graph from S
                # Neighbors in comp graph cannot share the same color
                S &= ~comp_mask[v]
                S &= ~v_bit
                
        # Reverse to process highest color first
        colored.reverse()
        return colored

    def expand(candidates_mask, clique_size, clique_list):
        nonlocal best_clique, best_size
        
        if candidates_mask == 0:
            if clique_size > best_size:
                best_size = clique_size
                best_clique = clique_list[:]
            return
            
        # Get colored vertices
        colored_vertices = color_sort(candidates_mask)
        
        # Process in order (highest color first, as color_sort returns reversed list)
        for v, color in colored_vertices:
            
            # Check if v is still in candidates
            # It should be, but just in case
            if not (candidates_mask & (1 << v)):
                continue
                
            # Upper bound check
            # The maximum size of a clique in the subgraph induced by 
            # vertices with color <= color is at most 'color'.
            # So the max size we can achieve from this branch is clique_size + color.
            if clique_size + color <= best_size:
                break # All remaining vertices have lower or equal color, so they can't improve
                
            # Branch: include v
            new_candidates = candidates_mask & comp_mask[v]
            
            # Add v to clique
            clique_list.append(v)
            expand(new_candidates, clique_size + 1, clique_list)
            
            # Remove v from clique
            clique_list.pop()
            
            # Exclude v: remove v from candidates for subsequent iterations
            candidates_mask &= ~(1 << v)

    # Start with all vertices
    initial_clique = []
    expand(all_mask, 0, initial_clique)
    
    # Convert vertex indices to values (1-based)
    result = [v + 1 for v in best_clique]
    
    if not result:
        result = [1]
        
    return result
