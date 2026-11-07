import time
import itertools
import random

DEADLINE = 85.0

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values):
    """
    Given 14 values, find 35 edges minimizing max(distinct_sums, distinct_prods).
    Since sums are usually < 35 and products are the bottleneck, we focus on products.
    However, we must track both.
    """
    n = len(values)
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_props = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_props.append((u, v, a + b, a * b))
        
    # State: set of selected edges, set of sums, set of products
    # We want to select 35 edges.
    
    # Greedy initialization
    selected_edges = []
    selected_set = set()
    current_sums = set()
    current_prods = set()
    
    # Sort edges by product frequency? No, just greedy.
    # To speed up, we can pre-sort edges by product value? 
    # If we pick edges with the same product, we don't increase distinct prods.
    
    # Let's count frequency of each product in all possible edges
    prod_freq = {}
    sum_freq = {}
    for _, _, s, p in edge_props:
        prod_freq[p] = prod_freq.get(p, 0) + 1
        sum_freq[s] = sum_freq.get(s, 0) + 1
        
    # Edges with high frequency products are better candidates
    # But we need to ensure we can pick 35 distinct edges.
    
    # Greedy:
    available_edges = list(range(len(all_edges)))
    for _ in range(35):
        best_idx = -1
        best_score = float('inf')
        for idx in available_edges:
            if idx in selected_set:
                continue
            u, v, s, p = edge_props[idx]
            new_s = 1 if s not in current_sums else 0
            new_p = 1 if p not in current_prods else 0
            score = max(len(current_sums) + new_s, len(current_prods) + new_p)
            # Tie breaker: prefer edges that add 0 new prods, then 0 new sums
            tie_break = new_p * 10 + new_s
            if score < best_score or (score == best_score and tie_break < best_score):
                best_score = score
                best_idx = idx
                # Store tie_break for comparison? 
                # Let's just use score, and if tie, we need a stable order.
                # Actually, if score is same, we should pick the one that is "safer".
                # Let's just pick the first one found with min score.
                
        if best_idx == -1:
            break
            
        u, v, s, p = edge_props[best_idx]
        selected_edges.append((u, v))
        selected_set.add(best_idx)
        current_sums.add(s)
        current_prods.add(p)
        
    # Local Search
    # Try to improve by swapping
    # An edge e1 in selected can be swapped with e2 not in selected
    # if it reduces distinct count.
    
    # To make this fast, we only consider swaps that reduce distinct products.
    # Condition for reduction:
    # 1. Product of e1 is unique to e1 in selected set.
    # 2. Product of e2 is already in current_prods (from other selected edges).
    # 3. Sum condition must not increase max distinct sums.
    
    improved = True
    while improved:
        improved = False
        # Build counts of products and sums in selected set
        prod_counts = {}
        sum_counts = {}
        for u, v in selected_edges:
            # Find props
            # This is slow. Let's maintain dicts.
            pass
            
    # Re-implement local search with maintained state
    selected_edges_list = [edge_props[i] for i in selected_set]
    # Map edge index to props
    # We need to be able to remove and add.
    
    # Let's restart local search with explicit sets
    sel_sums = set()
    sel_prods = set()
    sel_prod_counts = {}
    sel_sum_counts = {}
    
    for u, v in selected_edges:
        # Find props again? No, we have them.
        pass
        
    # This is getting messy. Let's use a simpler local search:
    # Iterate through selected edges. For each, try to replace with any non-selected edge.
    # Calculate delta.
    
    # To speed up, precompute for each edge its (s, p).
    # current state: set of (s, p) pairs? No, just sets.
    
    # Let's just do a few rounds of random swaps if score is high?
    # No, deterministic is better.
    
    # Optimized Local Search:
    # 1. Compute frequency of each product in the selected set.
    # 2. Compute frequency of each sum in the selected set.
    # 3. For each selected edge e1:
    #    If prod(e1) appears only once, removing it removes that product.
    #    Try to find an unselected edge e2 such that:
    #       prod(e2) is already in sel_prods (freq > 0 in selected? No, freq in selected after removing e1? 
    #       If prod(e1) is unique, removing it makes count 0.
    #       We need prod(e2) to have count > 0 in the *remaining* selected set.
    #       OR prod(e2) == prod(e1) is impossible since e2 != e1 and products are same? 
    #       If prod(e2) == prod(e1), and e1 is unique, then e2 must be the only other one? No.
    #       If we swap e1 with e2:
    #       New prods = (sel_prods - {p1 if count==1}) + {p2 if p2 not in (sel_prods - {p1})}
    #       
    #       We want len(new_prods) < len(sel_prods).
    #       This happens if:
    #       1. p1 is removed (count was 1).
    #       2. p2 is NOT added (p2 is already in the remaining set).
    #       
    #       Similarly for sums, but we only care if max(distinct_sums, distinct_prods) decreases.
    
    # Implementation:
    
    # Helper to get props of edge index
    # We stored edge_props indexed by all_edges index.
    
    # Let's maintain:
    # sel_prod_counts: dict product -> count
    # sel_sum_counts: dict sum -> count
    # sel_prods_set: set of products with count > 0
    # sel_sums_set: set of sums with count > 0
    
    sel_prod_counts = {}
    sel_sum_counts = {}
    for u, v in selected_edges:
        # Find index in all_edges?
        # We have selected_set of indices.
        pass
        
    # Re-init state from selected_edges
    sel_prod_counts = {}
    sel_sum_counts = {}
    for u, v in selected_edges:
        # We need to map (u,v) to props.
        # Since n=14, we can just look it up.
        idx = all_edges.index((u,v)) # O(N)
        _, _, s, p = edge_props[idx]
        sel_prod_counts[p] = sel_prod_counts.get(p, 0) + 1
        sel_sum_counts[s] = sel_sum_counts.get(s, 0) + 1
        
    sel_prods_set = set(sel_prod_counts.keys())
    sel_sums_set = set(sel_sum_counts.keys())
    
    for _ in range(100): # Max iterations
        improved = False
        # Try to improve
        for idx1 in list(selected_set):
            u1, v1, s1, p1 = edge_props[idx1]
            # If p1 is unique, removing it reduces distinct prods by 1
            if sel_prod_counts[p1] > 1:
                continue # Removing this edge doesn't reduce distinct prods
            
            # Try to find an edge idx2 not in selected_set
            # such that p2 is in sel_prods_set (and p2 != p1, or if p2==p1, count must be >1? No, if p2==p1, and we remove p1 (unique), then add p2 (same), count goes 1->0->1. Net change 0. We want net change -1.)
            # So we need p2 != p1 AND p2 in sel_prods_set.
            # Wait, if p2 == p1, and we swap, we still have p1. Distinct prods unchanged.
            # We need p2 to be a product that is ALREADY present (count > 0) and p2 != p1.
            # Then we remove p1 (count 1->0, distinct -1) and add p2 (count k->k+1, distinct +0).
            # Net distinct prods -1.
            
            # Check sums:
            # Removing s1: if count(s1)==1, distinct sums -1.
            # Adding s2: if s2 not in sel_sums_set, distinct sums +1.
            # We need max(new_distinct_sums, new_distinct_prods) < max(old_distinct_sums, old_distinct_prods).
            
            old_ds = len(sel_sums_set)
            old_dp = len(sel_prods_set)
            old_score = max(old_ds, old_dp)
            
            # Candidate new state:
            new_dp = old_dp - 1 # p1 removed
            # p2 added. If p2 in sel_prods_set, new_dp stays old_dp - 1.
            # If p2 not in sel_prods_set, new_dp stays old_dp.
            
            # We iterate over unselected edges
            for idx2 in available_edges:
                if idx2 in selected_set:
                    continue
                u2, v2, s2, p2 = edge_props[idx2]
                
                # Check product condition
                if p2 == p1:
                    continue # No gain
                if p2 not in sel_prods_set:
                    continue # No gain in prods (actually loss if we don't remove p1? No, we remove p1. 
                             # If p2 is new, distinct prods: remove p1, add p2. Net 0. We want -1.)
                             # So p2 MUST be in sel_prods_set.
                
                # So p2 is in sel_prods_set and p2 != p1.
                # new_dp = old_dp - 1
                
                # Check sum condition
                # Remove s1, add s2
                # If s1 is unique, distinct sums -1. Else 0.
                # If s2 is new, distinct sums +1. Else 0.
                
                delta_ds = 0
                if sel_sum_counts[s1] == 1:
                    delta_ds -= 1
                if s2 not in sel_sums_set:
                    delta_ds += 1
                    
                new_ds = old_ds + delta_ds
                new_score = max(new_ds, new_dp)
                
                if new_score < old_score:
                    # Perform swap
                    # Remove idx1
                    selected_set.remove(idx1)
                    available_edges.append(idx1)
                    sel_prod_counts[p1] -= 1
                    if sel_prod_counts[p1] == 0:
                        del sel_prod_counts[p1]
                        sel_prods_set.remove(p1)
                    sel_sum_counts[s1] -= 1
                    if sel_sum_counts[s1] == 0:
                        del sel_sum_counts[s1]
                        sel_sums_set.remove(s1)
                        
                    # Add idx2
                    selected_set.add(idx2)
                    available_edges.remove(idx2) # O(N)
                    sel_prod_counts[p2] = sel_prod_counts.get(p2, 0) + 1
                    sel_prods_set.add(p2)
                    sel_sum_counts[s2] = sel_sum_counts.get(s2, 0) + 1
                    sel_sums_set.add(s2)
                    
                    improved = True
                    break # Restart loop? Or continue?
                    # Breaking inner loop, continue outer loop with new state
                    # Actually, we should restart the search for idx1 because state changed.
                    # But simple break and continue outer loop is okay for small n.
                    
            if improved:
                break # Break outer loop too? 
                # If we break outer, we restart the while loop.
                
        if not improved:
            break
            
    # Convert back to list of tuples
    final_edges = [ (edge_props[i][0], edge_props[i][1]) for i in selected_set ]
    return final_edges

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Generate pool of smooth numbers (2, 3, 5)
    pool = set()
    for a in range(14):
        for b in range(9):
            for c in range(7):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000 and val > 0:
                    pool.add(val)
    
    pool = list(pool)
    
    # Random search over subsets of 14
    random.seed(42)
    
    # Also try some structured sets
    # Arithmetic progressions
    for a in range(1, 20):
        for d in range(1, 10):
            vals = [a + i * d for i in range(14)]
            if max(vals) <= 10000:
                edges = _optimize_edges(vals)
                score = _raw_score(vals, edges)
                if score < best_score:
                    best_score = score
                    best_result = (vals, edges)
            if time.time() > DEADLINE:
                return {'values': [int(v) for v in best_result[0]], 'edges': [list(e) for e in best_result[1]]} if best_result else {'values': list(range(1,15)), 'edges': [list(e) for e in itertools.combinations(range(14),2)[:35]]}
                
    # Random subsets from smooth pool
    if time.time() < DEADLINE - 5:
        for _ in range(2000):
            if time.time() > DEADLINE - 5:
                break
            vals = random.sample(pool, 14)
            edges = _optimize_edges(vals)
            score = _raw_score(vals, edges)
            if score < best_score:
                best_score = score
                best_result = (vals, edges)
                
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(14), 2))[:35]
        return {'values': vals, 'edges': [list(e) for e in edges]}
        
    vals, edges = best_result
    vals = [int(v) for v in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}
