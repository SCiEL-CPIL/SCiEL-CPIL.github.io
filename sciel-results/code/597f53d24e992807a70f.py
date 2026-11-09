"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools
import random

def construct():
    n = 14
    m = 35
    
    def get_complete_graph_distincts(values):
        """Calculate distinct sums and products for the complete graph."""
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i+1, n):
                sums.add(values[i] + values[j])
                prods.add(values[i] * values[j])
        return len(sums), len(prods)

    def generate_candidate_sets():
        candidate_sets = []
        
        # Hardcoded sets designed for high factor overlap
        hardcoded = [
            [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
            [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
            [1, 2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 40, 7],
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            [6, 10, 12, 14, 15, 18, 20, 21, 22, 24, 25, 26, 27, 28]
        ]
        for h in hardcoded:
            if len(set(h)) == 14 and all(1 <= v <= 10000 for v in h):
                candidate_sets.append(sorted(h))
                
        # Generate random sets and filter by complete-graph distinctness
        # We want sets with low distinct sums and products in the complete graph
        rng = random.Random(42)
        seen_sets = set(tuple(sorted(s)) for s in candidate_sets)
        
        for _ in range(100):
            # Generate random set with bias towards small composites
            # Sample from a pool of small numbers with high factor counts
            pool = list(range(1, 50))
            rng.shuffle(pool)
            vals = pool[:14]
            if len(set(vals)) != 14:
                continue
            vals = sorted(vals)
            key = tuple(vals)
            if key in seen_sets:
                continue
            
            # Heuristic: Check if this set is "good" by looking at complete graph distincts
            # We don't want to run the full greedy on all 100, so we filter here
            d_sums, d_prods = get_complete_graph_distincts(vals)
            # If the complete graph already has high distinct counts, the subset will likely be high too
            # But a low complete graph count is a strong signal of potential
            # Let's keep sets where max(d_sums, d_prods) is relatively low
            # For 14 nodes, complete graph has 91 edges.
            # Distinct sums range from 1+2=3 to 48+49=97. Max possible ~95.
            # Distinct products range from 2 to 2352.
            # A "good" set might have < 30 distinct sums and < 40 distinct products in complete graph?
            # Let's just keep the top 10 best random sets based on max(d_sums, d_prods)
            
            seen_sets.add(key)
            candidate_sets.append(vals)
            
        # Sort candidate sets by their complete graph score to prioritize good ones
        def score_set(s):
            d_sums, d_prods = get_complete_graph_distincts(s)
            return max(d_sums, d_prods)
            
        candidate_sets.sort(key=score_set)
        
        # Keep top 15 sets to balance diversity and computation time
        return candidate_sets[:15]

    candidate_sets = generate_candidate_sets()

    def run_greedy(values, seed=0):
        rng = random.Random(seed)
        all_edges = list(itertools.combinations(range(n), 2))
        
        distinct_sums = set()
        distinct_products = set()
        selected_edges = []
        
        # Precompute sum and product for each edge
        edge_info = {}
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_info[(i, j)] = (s, p)
            
        # Precompute frequency of sums and products in the FULL set of edges
        sum_counts = {}
        prod_counts = {}
        for i, j in all_edges:
            s, p = edge_info[(i, j)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        remaining = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_tie_breaker = float('inf')
            
            for edge in remaining:
                s, p = edge_info[edge]
                new_sums = 0 if s in distinct_sums else 1
                new_prods = 0 if p in distinct_products else 1
                
                # Primary metric: max of distinct counts after addition
                new_max = max(len(distinct_sums) + new_sums, len(distinct_products) + new_prods)
                
                # Secondary metric: Sum of new distincts
                new_total = new_sums + new_prods
                
                # Tertiary metric: Prefer edges that don't add new distinct values.
                # If they do, prefer those with higher frequency in the remaining pool?
                # Or simply prefer lower frequency? 
                # Actually, if a sum is very common, it's likely to be added soon anyway.
                # If a sum is rare, adding it now "locks" it in, which might be bad if we could have avoided it.
                # But if we MUST add a new sum, which one?
                # Let's stick to: Minimize new_total.
                # Tie-breaker: Random noise to explore different paths.
                
                tie_breaker = new_total
                noise = rng.random() * 0.1
                total_score = (new_max, tie_breaker + noise)
                
                if total_score < (best_score, best_tie_breaker):
                    best_score = new_max
                    best_tie_breaker = tie_breaker + noise
                    best_edge = edge
            
            if best_edge is not None:
                selected_edges.append([best_edge[0], best_edge[1]])
                s, p = edge_info[best_edge]
                distinct_sums.add(s)
                distinct_products.add(p)
                remaining.discard(best_edge)
        
        # Calculate final score
        raw_value = max(len(distinct_sums), len(distinct_products))
        return raw_value, {'values': values, 'edges': selected_edges}

    best_raw_value = float('inf')
    best_result = None
    
    # Try multiple seeds for each value set to escape local minima
    # We have 15 sets. 10 seeds each = 150 runs.
    # Each run is O(m * E) = 35 * 91 ~ 3000 ops. 150 * 3000 = 450k ops. Very fast.
    for vals in candidate_sets:
        for seed in range(10):
            raw_val, result = run_greedy(vals, seed)
            if raw_val < best_raw_value:
                best_raw_value = raw_val
                best_result = result
                
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result

# EVOLVE-BLOCK-END
