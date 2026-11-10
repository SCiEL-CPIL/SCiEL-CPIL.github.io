"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    # Key insight: To minimize distinct sums AND products simultaneously,
    # we need value sets where many pairs share the same sum AND same product.
    # 
    # For sums: If values form an arithmetic progression, many pairs can share sums.
    # For products: If values are chosen so that a*b = c*d for many quadruples.
    #
    # Strategy: Use structured value sets and systematic edge selection.
    # 
    # Approach 1: Values in arithmetic progression (e.g., 1,2,3,...,14)
    #   Sums: i+j has many collisions. E.g., 1+13=2+12=3+11=...=7+8=14
    #   Products: 1*12=2*6=3*4=12, 1*14=2*7=14, etc.
    #
    # Approach 2: Values as divisors of a highly composite number
    #   E.g., divisors of 2^a * 3^b * 5^c * 7^d
    #   This creates many product collisions.
    #
    # Approach 3: Mix - some small values, some structured
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate candidate value sets with good collision properties
    value_sets = []
    
    # Set 1: Small integers 1-14 (good sum collisions)
    value_sets.append(list(range(1, n+1)))
    
    # Set 2: Arithmetic progression with step 2 (1,3,5,...,27)
    value_sets.append(list(range(1, 2*n+1, 2)))
    
    # Set 3: Powers and small numbers mixed
    # 1,2,3,4,5,6,8,9,10,12,15,16,20,24
    value_sets.append([1,2,3,4,5,6,8,9,10,12,15,16,20,24])
    
    # Set 4: Divisors of 360 (highly composite)
    # 1,2,3,4,5,6,8,9,10,12,15,18,20,24
    value_sets.append([1,2,3,4,5,6,8,9,10,12,15,18,20,24])
    
    # Set 5: Divisors of 720
    # 1,2,3,4,5,6,8,10,12,15,16,20,24,30
    value_sets.append([1,2,3,4,5,6,8,10,12,15,16,20,24,30])
    
    # Set 6: Geometric-ish with collisions
    # 1,2,4,8,3,6,12,24,5,10,20,40,7,14
    value_sets.append([1,2,4,8,3,6,12,24,5,10,20,40,7,14])
    
    # Set 7: Focus on product collisions
    # 1,2,3,4,6,8,9,12,16,18,24,27,32,36
    value_sets.append([1,2,3,4,6,8,9,12,16,18,24,27,32,36])
    
    # Set 8: Small range for maximum sum collisions
    # 1,2,3,4,5,6,7,8,9,10,11,12,13,14
    value_sets.append(list(range(1, 15)))
    
    # Set 9: Two groups - small and large
    # 1,2,3,4,5,6,7,8,9,10,11,12,13,14
    value_sets.append(list(range(1, 15)))
    
    # Set 10: Fibonacci-like
    value_sets.append([1,2,3,5,8,13,21,34,55,89,144,233,377,610])
    
    # Validate and add random sets
    valid_value_sets = []
    for vs in value_sets:
        if len(set(vs)) == n and all(isinstance(v, int) and v > 0 and v <= maximum for v in vs):
            valid_value_sets.append(vs)
    
    # Add random value sets with structure
    for _ in range(15):
        if time.time() > deadline - 15:
            break
        # Random but biased toward small values for more collisions
        random_vals = random.sample(range(1, 50), n)
        valid_value_sets.append(random_vals)
    
    # For each value set, use systematic edge selection
    def optimize_edges_systematic(values):
        """Select m edges to minimize max(distinct_sums, distinct_prods)"""
        all_edges = list(combinations(range(n), 2))
        
        # Precompute sum and product for each edge
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i,j)] = values[i] + values[j]
            edge_prod[(i,j)] = values[i] * values[j]
        
        # Group edges by sum
        sum_groups = {}
        for e in all_edges:
            s = edge_sum[e]
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append(e)
        
        # Group edges by product
        prod_groups = {}
        for e in all_edges:
            p = edge_prod[e]
            if p not in prod_groups:
                prod_groups[p] = []
            prod_groups[p].append(e)
        
        # Strategy: Greedy selection favoring edges that create collisions
        # Sort sum groups by size (largest first)
        sorted_sum_groups = sorted(sum_groups.items(), key=lambda x: -len(x[1]))
        sorted_prod_groups = sorted(prod_groups.items(), key=lambda x: -len(x[1]))
        
        # Multiple strategies, pick best
        best_local_score = float('inf')
        best_local_edges = None
        
        # Strategy A: Greedy by sum group size
        def greedy_by_sum():
            selected = []
            selected_set = set()
            sum_counts = {}
            prod_counts = {}
            
            for s, edges in sorted_sum_groups:
                for e in edges:
                    if len(selected) >= m:
                        break
                    if e not in selected_set:
                        selected.append(e)
                        selected_set.add(e)
                        p = edge_prod[e]
                        prod_counts[p] = prod_counts.get(p, 0) + 1
                if len(selected) >= m:
                    break
            
            # If we don't have m edges, fill randomly
            if len(selected) < m:
                remaining = [e for e in all_edges if e not in selected_set]
                random.shuffle(remaining)
                for e in remaining[:m-len(selected)]:
                    selected.append(e)
                    selected_set.add(e)
            
            return selected
        
        # Strategy B: Greedy by product group size
        def greedy_by_prod():
            selected = []
            selected_set = set()
            
            for p, edges in sorted_prod_groups:
                for e in edges:
                    if len(selected) >= m:
                        break
                    if e not in selected_set:
                        selected.append(e)
                        selected_set.add(e)
                if len(selected) >= m:
                    break
            
            if len(selected) < m:
                remaining = [e for e in all_edges if e not in selected_set]
                random.shuffle(remaining)
                for e in remaining[:m-len(selected)]:
                    selected.append(e)
                    selected_set.add(e)
            
            return selected
        
        # Strategy C: Combined - prefer edges that are in both large sum and product groups
        def greedy_combined():
            # Score each edge by how "colliding" it is
            edge_scores = {}
            for e in all_edges:
                s = edge_sum[e]
                p = edge_prod[e]
                sum_size = len(sum_groups.get(s, [e]))
                prod_size = len(prod_groups.get(p, [e]))
                # Higher score = more collisions
                edge_scores[e] = sum_size * prod_size
            
            # Sort by score descending
            sorted_edges = sorted(all_edges, key=lambda e: -edge_scores[e])
            selected = sorted_edges[:m]
            return selected
        
        # Strategy D: Random restart hill climbing
        def random_hill_climb():
            current_edges = set(random.sample(all_edges, m))
            current_score = _raw_value(values, list(current_edges))
            best_local = current_score
            best_edges_local = set(current_edges)
            
            for _ in range(500):
                if time.time() > deadline - 10:
                    break
                
                e_out = random.choice(list(current_edges))
                possible_in = [e for e in all_edges if e not in current_edges]
                if not possible_in:
                    break
                e_in = random.choice(possible_in)
                
                new_edges = set(current_edges)
                new_edges.remove(e_out)
                new_edges.add(e_in)
                
                new_score = _raw_value(values, list(new_edges))
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    if current_score < best_local:
                        best_local = current_score
                        best_edges_local = set(current_edges)
            
            return list(best_edges_local)
        
        # Try all strategies
        strategies = [greedy_by_sum, greedy_by_prod, greedy_combined, random_hill_climb]
        
        for strategy in strategies:
            if time.time() > deadline - 5:
                break
            try:
                edges = strategy()
                if len(edges) == m and len(set(edges)) == m:
                    score = _raw_value(values, edges)
                    if score < best_local_score:
                        best_local_score = score
                        best_local_edges = edges
            except:
                pass
        
        return best_local_edges, best_local_score
    
    # Evaluate all value sets
    for idx, vs in enumerate(valid_value_sets):
        if time.time() > deadline - 5:
            break
        
        edges, score = optimize_edges_systematic(vs)
        if edges is not None and score < best_score:
            best_score = score
            best_values = vs
            best_edges = edges
    
    # If no good solution found, use fallback
    if best_edges is None:
        best_values = list(range(1, n+1))
        best_edges = list(combinations(range(n), 2))[:m]
        best_score = _raw_value(best_values, best_edges)
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and v > 0 and v <= maximum for v in best_values)
    assert len(best_edges) == m
    assert all(len(e) == 2 for e in best_edges)
    assert all(e[0] != e[1] for e in best_edges)
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {
        'values': list(best_values),
        'edges': [list(e) for e in best_edges]
    }
# EVOLVE-BLOCK-END
