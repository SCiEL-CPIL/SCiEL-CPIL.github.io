"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    current_sums = set()
    current_products = set()
    selected_edges = []
    remaining_edges = set(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        best_new_sums = set()
        best_new_products = set()
        
        for edge in remaining_edges:
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            
            new_sums = current_sums | {s}
            new_products = current_products | {p}
            
            # We want to minimize the max of the counts
            score = max(len(new_sums), len(new_products))
            
            # Tie-breaker: prefer edges that add fewer new products, then fewer new sums
            # Or just keep the first one found with min score
            if score < best_score:
                best_score = score
                best_edge = edge
                best_new_sums = new_sums
                best_new_products = new_products
            elif score == best_score:
                # Secondary criteria: minimize number of new products added
                new_p_count = len(new_products - current_products)
                old_p_count = len(best_new_products - current_products)
                if new_p_count < old_p_count:
                    best_edge = edge
                    best_new_sums = new_sums
                    best_new_products = new_products
                elif new_p_count == old_p_count:
                    new_s_count = len(new_sums - current_sums)
                    old_s_count = len(best_new_sums - current_sums)
                    if new_s_count < old_s_count:
                        best_edge = edge
                        best_new_sums = new_sums
                        best_new_products = new_products
                        
        if best_edge is None:
            break
            
        selected_edges.append(list(best_edge))
        current_sums = best_new_sums
        current_products = best_new_products
        remaining_edges.remove(best_edge)
        
    return {'values': values, 'edges': selected_edges}
# EVOLVE-BLOCK-END
