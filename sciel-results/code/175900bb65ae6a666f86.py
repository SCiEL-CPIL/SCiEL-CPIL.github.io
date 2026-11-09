        bound, ordered_verts = coloring_bound_and_order(candidates_mask)
        if current_size + bound <= best_size:
            return
        
        if not ordered_verts:
             # ...
        
        # Pick the vertex with highest degree (first in sorted list)
        # Or maybe pick the one that appears latest in the coloring?
        # Standard Tomita for Max Clique picks the vertex with the highest color.
        # Here we are coloring the conflict graph.
        # Let's stick to highest degree for now, but using the pre-computed list.
        deg, v_val = ordered_verts[0]
        v_idx = v_val - 1
        v_bit = 1 << v_idx
    