import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, 10s buffer
    
    # Values: 1 to 14
    # This minimizes distinct sums (max 25)
    values = list(range(1, 15))
    n = len(values)
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u, v = values[i], values[j]
            s = u + v
            p = u * v
            all_edges.append((i, j, s, p))
    
    # Greedy selection to minimize max(distinct_sums, distinct_products)
    # We want to pick 35 edges
    target_m = 35
    
    selected_edges = []
    used_sums = set()
    used_prods = set()
    
    # Pre-calculate potential gains? 
    # Simple greedy: pick edge that minimizes (len(used_sums + {s}), len(used_prods + {p}))
    # Tie break: prefer edges that add fewer new products, then fewer new sums.
    
    # To speed up, we can sort edges initially by "popularity" of product
    # But dynamic greedy is safer.
    
    remaining_edges = list(all_edges)
    
    # Heuristic: Start with edges that have very common products
    # Count product frequencies in all edges
    prod_counts = {}
    for _, _, _, p in all_edges:
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    # Sort remaining edges by product frequency descending
    # This biases towards edges that are likely to collide with others
    remaining_edges.sort(key=lambda x: prod_counts.get(x[3], 0), reverse=True)
    
    while len(selected_edges) < target_m:
        if time.time() > deadline:
            break
            
        best_edge = None
        best_score = float('inf')
        best_new_sums = float('inf')
        best_new_prods = float('inf')
        
        # Iterate through candidates
        # Optimization: If we have many candidates, we might want to sample or use a better heuristic.
        # 91 edges is small, so full scan is fine.
        
        for idx, edge in enumerate(remaining_edges):
            i, j, s, p = edge
            
            # Calculate new counts
            new_sums_count = len(used_sums) + (0 if s in used_sums else 1)
            new_prods_count = len(used_prods) + (0 if p in used_prods else 1)
            
            current_score = max(new_sums_count, new_prods_count)
            
            if current_score < best_score:
                best_score = current_score
                best_new_sums = new_sums_count
                best_new_prods = new_prods_count
                best_edge = edge
                best_idx = idx
            elif current_score == best_score:
                # Tie-breaking: prefer lower new products, then lower new sums
                if new_prods_count < best_new_prods:
                    best_new_prods = new_prods_count
                    best_new_sums = new_sums_count
                    best_edge = edge
                    best_idx = idx
                elif new_prods_count == best_new_prods and new_sums_count < best_new_sums:
                    best_new_sums = new_sums_count
                    best_edge = edge
                    best_idx = idx
                    
        if best_edge is None:
            break
            
        # Add best edge
        i, j, s, p = best_edge
        selected_edges.append((i, j))
        used_sums.add(s)
        used_prods.add(p)
        
        # Remove from remaining
        remaining_edges.pop(best_idx)
        
    # If we didn't reach 35 edges (should not happen), fill with any
    if len(selected_edges) < target_m:
        for edge in all_edges:
            if len(selected_edges) >= target_m:
                break
            if edge not in [(e[0], e[1]) for e in selected_edges]:
                # Check if already added by index pair
                if (edge[0], edge[1]) not in selected_edges:
                    selected_edges.append((edge[0], edge[1]))
                    used_sums.add(edge[2])
                    used_prods.add(edge[3])

    # Final check: Ensure we have exactly 35 edges
    # The logic above should guarantee this.
    
    # Convert to required format
    edges_list = [list(e) for e in selected_edges]
    
    # Validate
    if len(values) != 14:
        raise ValueError("Values must be 14 distinct")
    if len(set(values)) != 14:
        raise ValueError("Values must be distinct")
    if any(v <= 0 or v > 10000 for v in values):
        raise ValueError("Values out of range")
    if len(edges_list) != 35:
        raise ValueError("Must have 35 edges")
        
    # Check for duplicates/loops
    seen = set()
    for e in edges_list:
        if len(e) != 2:
            raise ValueError("Edge format")
        i, j = e
        if i == j:
            raise ValueError("Loop")
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            raise ValueError("Duplicate")
        seen.add((i, j))
        if i >= 14 or j >= 14:
            raise ValueError("Index out of bounds")

    return {'values': values, 'edges': edges_list}

if __name__ == '__main__':
    res = construct()
    # Quick local check
    vals = res['values']
    edges = res['edges']
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(vals[u] + vals[v])
        prods.add(vals[u] * vals[v])
    print(f"Distinct Sums: {len(sums)}")
    print(f"Distinct Prods: {len(prods)}")
    print(f"Raw Value: {max(len(sums), len(prods))}")
