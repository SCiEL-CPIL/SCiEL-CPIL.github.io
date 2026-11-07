def construct():
    n = 14
    values = list(range(1, n+1))
    # Generate all edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            u = values[i]
            v = values[j]
            s = u + v
            p = u * v
            all_edges.append((s, p, i, j))
            
    # Calculate frequencies
    from collections import Counter
    sum_counts = Counter(e[0] for e in all_edges)
    prod_counts = Counter(e[1] for e in all_edges)
    
    # Score each edge: higher is better (more "common")
    # We want to pick edges that reuse sums and products.
    # A simple heuristic: sort by (sum_count, prod_count) descending
    # This tends to cluster edges in the middle sums and common products.
    scored_edges = []
    for s, p, i, j in all_edges:
        score = (sum_counts[s], prod_counts[p])
        scored_edges.append((score, i, j))
        
    # Sort
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 35
    selected = scored_edges[:35]
    
    # Refinement: The top 35 might not be optimal for the *union* size.
    # But it's a strong start.
    # Let's do a simple local search: swap an edge in with an edge out if it reduces max(distinct_sums, distinct_prods).
    
    def evaluate(edges):
        sums = set()
        prods = set()
        for _, i, j in edges:
            u = values[i]
            v = values[j]
            sums.add(u+v)
            prods.add(u*v)
        return max(len(sums), len(prods))
    
    current_edges = [(i, j) for _, i, j in selected]
    current_val = evaluate(current_edges)
    
    # Simple local search
    # Try swapping one edge from selected with one from unselected
    selected_set = set(current_edges)
    unselected = [(i, j) for _, i, j in scored_edges[35:]]
    
    improved = True
    while improved:
        improved = False
        for i_in in range(len(current_edges)):
            for i_out in range(len(unselected)):
                e_in = current_edges[i_in]
                e_out = unselected[i_out]
                if e_in == e_out: continue
                # Try swap
                new_edges = current_edges.copy()
                new_edges[i_in] = e_out
                # Check if e_out is already in new_edges? No, e_out was not in current.
                # But e_in is removed.
                # Need to ensure no duplicates? The list is unique.
                val = evaluate(new_edges)
                if val < current_val:
                    current_edges = new_edges
                    current_val = val
                    # Update sets
                    selected_set.remove(e_in)
                    selected_set.add(e_out)
                    # Move e_in to unselected, e_out to selected list in our tracking
                    # For simplicity, just update the list and re-sort unselected? 
                    # Actually, just updating the list and flagging improved is enough for one pass.
                    improved = True
                    # Break out of loops to restart? Or just continue.
                    # To be safe, break and restart the outer loop to ensure consistency.
                    break
            if improved:
                break
                
    # Convert to required format
    edges = [list(e) for e in current_edges]
    return {'values': values, 'edges': edges}
