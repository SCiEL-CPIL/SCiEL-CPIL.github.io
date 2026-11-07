"""Rewritten program: minimize max(distinct edge sums, distinct edge products) for 14 values, 35 edges."""
# EVOLVE-BLOCK-START

def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Strategy: Use values that are powers of 2 or small integers to create many product collisions.
    # If we use values that are divisors of a highly composite number, we get many factor pairs.
    # Let's try to construct a set of 14 numbers where many pairs have the same product.
    
    # A known approach: use numbers from a multiplicative structure.
    # For example, if we use numbers like 2^a * 3^b, products collide when exponents sum the same.
    # But we only have 14 numbers.
    
    # Let's try a specific construction based on factorization.
    # We want to maximize the number of edges that share the same product.
    
    # Consider the number 60 = 2^2 * 3 * 5. Its divisors are 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60.
    # That's 12 divisors. We need 14 distinct values.
    # Let's add 1 and 2 to the divisors of 60? No, 1 and 2 are already there.
    # Let's use divisors of 120 = 2^3 * 3 * 5. Divisors: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120.
    # That's 16 divisors. We can pick 14 of them.
    
    # Let's pick: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40.
    values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40]
    
    # Now we need to select 35 edges out of C(14,2) = 91 edges.
    # We want to minimize the number of distinct sums and distinct products.
    # Let's compute all possible sums and products for all pairs.
    
    all_pairs = list(combinations(range(n), 2))
    pair_sums = {}
    pair_prods = {}
    
    for i, j in all_pairs:
        s = values[i] + values[j]
        p = values[i] * values[j]
        if s not in pair_sums:
            pair_sums[s] = []
        pair_sums[s].append((i, j))
        if p not in pair_prods:
            pair_prods[p] = []
        pair_prods[p].append((i, j))
    
    # We want to pick 35 edges such that the number of distinct sums used is small and the number of distinct products used is small.
    # This is a combinatorial optimization problem.
    # Greedy approach: pick edges that belong to the most frequent sum/product groups.
    
    # Let's sort sum groups by size descending, and product groups by size descending.
    sum_groups = sorted(pair_sums.items(), key=lambda x: -len(x[1]))
    prod_groups = sorted(pair_prods.items(), key=lambda x: -len(x[1]))
    
    # We'll use a greedy algorithm: at each step, pick the edge that is in the largest available group (by sum or product).
    # We need to track which edges are already picked.
    
    used_edges = set()
    selected_edges = []
    
    # Precompute for each edge its sum group size and product group size.
    edge_sum_group_size = {}
    edge_prod_group_size = {}
    
    for s, edges in pair_sums.items():
        for e in edges:
            edge_sum_group_size[e] = len(edges)
    
    for p, edges in pair_prods.items():
        for e in edges:
            edge_prod_group_size[e] = len(edges)
    
    # Greedy: pick the edge with the maximum (sum_group_size + prod_group_size) among unused edges.
    # This tends to pick edges that are part of large collision groups.
    
    for _ in range(m):
        if time.time() > deadline:
            break
        best_edge = None
        best_score = -1
        for i, j in all_pairs:
            if (i, j) in used_edges:
                continue
            score = edge_sum_group_size[(i, j)] + edge_prod_group_size[(i, j)]
            if score > best_score:
                best_score = score
                best_edge = (i, j)
        if best_edge is None:
            break
        used_edges.add(best_edge)
        selected_edges.append(list(best_edge))
    
    # Check if we have enough edges
    if len(selected_edges) < m:
        # Fill remaining edges arbitrarily
        for i, j in all_pairs:
            if len(selected_edges) >= m:
                break
            if (i, j) not in used_edges:
                selected_edges.append(list((i, j)))
                used_edges.add((i, j))
    
    # Compute the objective
    distinct_sums = set()
    distinct_prods = set()
    for i, j in selected_edges:
        distinct_sums.add(values[i] + values[j])
        distinct_prods.add(values[i] * values[j])
    
    objective = max(len(distinct_sums), len(distinct_prods))
    
    # Try to improve with local search: swap edges to reduce distinct sums/prods.
    # This is complex, so we'll just return the greedy result for now.
    
    # Ensure types are correct
    values = [int(v) for v in values]
    selected_edges = [[int(i), int(j)] for i, j in selected_edges]
    
    return {'values': values, 'edges': selected_edges}

# EVOLVE-BLOCK-END
