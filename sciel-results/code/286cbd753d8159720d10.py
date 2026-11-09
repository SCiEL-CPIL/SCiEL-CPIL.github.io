"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 14
    m = 35
    
    # Choose values that encourage shared sums and products
    # Use a set of values with many common factors and clustered magnitudes
    # Powers of 2 give few distinct products (exponents add)
    # But sums are more varied. Let's try a mix.
    
    # Strategy: use values that are multiples of a common base to reduce product distinctness
    # and cluster values to reduce sum distinctness.
    
    # Let's use: 1,2,3,4,5,6,7,8,9,10,11,12,13,14 -> sums 3..27, products many
    # Better: use values with high overlap in sums/products
    
    # Greedy edge selection: start with empty edge set, repeatedly add the edge
    # that adds the fewest new distinct sums and products (weighted).
    
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    from itertools import combinations
    
    all_edges = list(combinations(range(n), 2))
    
    distinct_sums = set()
    distinct_products = set()
    selected_edges = []
    
    # Precompute sum and product for each edge
    edge_info = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_info[(i, j)] = (s, p)
    
    remaining = set(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        best_new_sums = 0
        best_new_prods = 0
        
        for edge in remaining:
            s, p = edge_info[edge]
            new_sums = 0 if s in distinct_sums else 1
            new_prods = 0 if p in distinct_products else 1
            # Score: total distinct after adding
            score = (len(distinct_sums) + new_sums) + (len(distinct_products) + new_prods)
            # We want to minimize max(distinct_sums, distinct_products)
            # Better score: max of the two after addition
            new_max = max(len(distinct_sums) + new_sums, len(distinct_products) + new_prods)
            if new_max < best_score or (new_max == best_score and (new_sums + new_prods) < (best_new_sums + best_new_prods)):
                best_score = new_max
                best_edge = edge
                best_new_sums = new_sums
                best_new_prods = new_prods
        
        if best_edge is not None:
            selected_edges.append([best_edge[0], best_edge[1]])
            s, p = edge_info[best_edge]
            distinct_sums.add(s)
            distinct_products.add(p)
            remaining.discard(best_edge)
    
    return {'values': values, 'edges': selected_edges}

# EVOLVE-BLOCK-END
