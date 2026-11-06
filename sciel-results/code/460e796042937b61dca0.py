import itertools
import random
import math

def construct():
    n = 16
    # Edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Map edge to index
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Subsets of 4 vertices
    subsets = list(itertools.combinations(range(n), 4))
    num_subsets = len(subsets)
    
    # For each subset, list of edge indices
    subset_edges = []
    for sub in subsets:
        es = []
        for u, v in itertools.combinations(sub, 2):
            es.append(edge_to_idx[(u, v)])
        subset_edges.append(es)
        
    # For each edge index, list of subset indices it belongs to
    edge_to_subsets = [[] for _ in range(num_edges)]
    for s_idx, es in enumerate(subset_edges):
        for e_idx in es:
            edge_to_subsets[e_idx].append(s_idx)
            
    # State: colors for each edge
    # Start with a random assignment of 13 colors
    max_colors_initial = 13
    colors = [random.randint(0, max_colors_initial - 1) for _ in range(num_edges)]
    
    # Precompute color counts for each subset
    # sub_color_counts[s_idx] is a dict {color: count}
    sub_color_counts = [{} for _ in range(num_subsets)]
    # sub_distinct[s_idx] is the number of distinct colors
    sub_distinct = [0] * num_subsets
    
    for s_idx, es in enumerate(subset_edges):
        counts = {}
        for e_idx in es:
            c = colors[e_idx]
            counts[c] = counts.get(c, 0) + 1
        sub_color_counts[s_idx] = counts
        sub_distinct[s_idx] = len(counts)
        
    # Global violation count
    violations = sum(1 for d in sub_distinct if d < 5)
    
    # Objective function
    def get_objective():
        distinct_colors = len(set(colors))
        return distinct_colors, violations

    # Simulated Annealing
    # We want to minimize distinct_colors, subject to violations == 0.
    # Cost = distinct_colors + penalty * violations
    
    current_distinct, current_violations = get_objective()
    best_colors = colors[:]
    best_distinct = current_distinct
    
    # If we start valid, record it. But random 13 is likely invalid.
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    penalty = 10.0 # High penalty to enforce validity first
    
    # We will run until time limit or T too low
    # Use a fixed number of iterations to be safe, or check time.
    # 120s limit. Let's do ~100,000 iterations. Each is O(91) ~ 10k ops. 1e9 ops? Too slow in Python.
    # Need to optimize the update.
    
    # Optimization:
    # Instead of updating counts for all 91 subsets, we can just recalculate the distinct count for those 91 subsets?
    # No, maintaining counts is better.
    # But dictionary operations in Python are slow.
    # Let's use a list of size 120 for counts? No, colors are 0..119.
    # We can use a global array for color usage? No, per subset.
    
    # Let's try a simpler move: Swap colors of two edges?
    # Or just change one edge.
    
    # To speed up, let's limit the number of iterations and rely on the structure.
    # Actually, 120s is generous. 100,000 iterations of simple Python loops might take ~10-20s.
    # Let's try 200,000 iterations.
    
    import time
    start_time = time.time()
    
    for step in range(200000):
        if time.time() - start_time > 110:
            break
            
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_c = colors[e_idx]
        
        # Pick a new color
        # Bias towards existing colors to reduce distinct count?
        # Or random 0..12?
        # Let's pick from 0..12.
        new_c = random.randint(0, 12)
        
        if new_c == old_c:
            continue
            
        # Calculate delta violations
        delta_violations = 0
        # We need to update the 91 subsets
        # To avoid modifying state if we reject, we can just calculate the change.
        # But calculating change requires knowing the counts.
        # Let's just apply, check, and revert if bad?
        # Or calculate delta on the fly.
        
        # Let's apply to a copy of the relevant data? No, too slow.
        # Let's apply, calculate new violations, and revert if worse.
        
        # Apply
        colors[e_idx] = new_c
        
        # Update subsets
        affected_subs = edge_to_subsets[e_idx]
        # We need to track which subsets changed validity to update global violation count.
        # But we can just recalculate global violations? No, O(1820) is too slow.
        
        # Let's maintain sub_distinct and sub_color_counts.
        # We need to revert them if we reject.
        # This is getting complex.
        
        # Alternative: 
        # Just recalculate the distinct count for the 91 affected subsets.
        # For each affected subset, count distinct colors in its 6 edges.
        # This is 91 * 6 = 546 operations.
        # 200,000 * 546 = 109,200,000 operations.
        # In Python, this might be ~10-20 seconds. Feasible.
        
        # Let's do this simpler approach.
        
        # Revert colors first to calculate delta?
        # No, let's just calculate the new state's validity for affected subsets.
        
        # We need to know the old distinct counts for these subsets to update the global violation count.
        # So we must maintain `sub_distinct`.
        
        # Plan:
        # 1. Save old distinct counts for affected subsets.
        # 2. Update colors[e_idx].
        # 3. Recalculate distinct counts for affected subsets.
        # 4. Calculate new global violations.
        # 5. Calculate new distinct colors (global).
        # 6. Decide accept/reject.
        # 7. If reject, revert colors[e_idx] and restore old distinct counts.
        
        old_distincts = [sub_distinct[s] for s in affected_subs]
        
        # Recalculate
        for s_idx in affected_subs:
            es = subset_edges[s_idx]
            # Get colors of these edges
            c_set = set()
            for e in es:
                c_set.add(colors[e])
            sub_distinct[s_idx] = len(c_set)
            
        new_violations = sum(1 for d in sub_distinct if d < 5)
        new_distinct_colors = len(set(colors))
        
        # Objective
        old_cost = current_distinct + penalty * current_violations
        new_cost = new_distinct_colors + penalty * new_violations
        
        delta_cost = new_cost - old_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        elif delta_cost == 0:
            # Accept to explore
            accept = True
        else:
            if T > 0:
                # Boltzmann probability
                # Avoid overflow
                if delta_cost / T < 700:
                    prob = math.exp(-delta_cost / T)
                    if random.random() < prob:
                        accept = True
        
        if accept:
            current_distinct = new_distinct_colors
            current_violations = new_violations
            if new_violations == 0 and new_distinct_colors < best_distinct:
                best_distinct = new_distinct_colors
                best_colors = colors[:]
        else:
            # Revert
            colors[e_idx] = old_c
            for i, s_idx in enumerate(affected_subs):
                sub_distinct[s_idx] = old_distincts[i]
                
        # Update temperature
        T *= cooling
        
    # After SA, if best_colors has violations, we need to fix it.
    # But we should have found a valid one.
    # If not, fall back to cyclic 17.
    
    # Check validity of best_colors
    def is_valid(cols):
        for s_idx, es in enumerate(subset_edges):
            c_set = set()
            for e in es:
                c_set.add(cols[e])
            if len(c_set) < 5:
                return False
        return True
        
    if is_valid(best_colors):
        # Remap colors to 0..k-1
        used = set(best_colors)
        mapping = {c: i for i, c in enumerate(sorted(used))}
        final_colors = [mapping[c] for c in best_colors]
        return final_colors
    else:
        # Fallback
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                res.append((i + j) % 17)
        return res
