import time
import random
from itertools import combinations
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin (90s limit)
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    # Value sets that encourage repeated sums/products
    value_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    ]
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    def compute_edge_maps(vals):
        edge_sums = {}
        edge_prods = {}
        for i, j in all_edges:
            edge_sums[(i, j)] = vals[i] + vals[j]
            edge_prods[(i, j)] = vals[i] * vals[j]
        return edge_sums, edge_prods
    
    def greedy_edges(vals, edge_sums, edge_prods):
        """Greedy edge selection: prefer edges that reuse existing sum/product values."""
        sum_freq = Counter(edge_sums.values())
        prod_freq = Counter(edge_prods.values())
        sum_count = Counter()
        prod_count = Counter()
        edges = []
        edge_set = set()
        remaining = set(all_edges)
        
        for _ in range(m):
            best_e = None
            best_key = None
            for e in remaining:
                s = edge_sums[e]
                p = edge_prods[e]
                new_s = 0 if sum_count[s] > 0 else 1
                new_p = 0 if prod_count[p] > 0 else 1
                # Primary: minimize new distinct values; Secondary: prefer high-frequency
                key = (new_s + new_p, -sum_freq[s], -prod_freq[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_e = e
            edges.append(list(best_e))
            edge_set.add(best_e)
            sum_count[edge_sums[best_e]] += 1
            prod_count[edge_prods[best_e]] += 1
            remaining.discard(best_e)
        return edges, edge_set
    
    def local_search(vals, init_edges, max_iters=30000):
        """Local search with both edge swaps and value perturbations, using fast counters."""
        vals = list(vals)
        edges = [list(e) for e in init_edges]
        edge_set = set(tuple(e) for e in edges)
        
        edge_sums, edge_prods = compute_edge_maps(vals)
        
        # Build initial counters
        s_counts = Counter()
        p_counts = Counter()
        for e in edge_set:
            s_counts[edge_sums[e]] += 1
            p_counts[edge_prods[e]] += 1
        
        cur_score = max(len(s_counts), len(p_counts))
        best_local_score = cur_score
        best_local_vals = list(vals)
        best_local_edges = [list(e) for e in edges]
        
        T = 3.0
        cooling = 0.99995
        min_T = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if random.random() < 0.75:
                # Edge swap: remove one edge, add another
                old_edge = random.choice(list(edge_set))
                new_edge = random.choice(all_edges)
                if new_edge in edge_set:
                    continue
                
                old_s = edge_sums[old_edge]
                old_p = edge_prods[old_edge]
                new_s = edge_sums[new_edge]
                new_p = edge_prods[new_edge]
                
                # Apply swap
                s_counts[old_s] -= 1
                if s_counts[old_s] == 0:
                    del s_counts[old_s]
                p_counts[old_p] -= 1
                if p_counts[old_p] == 0:
                    del p_counts[old_p]
                
                s_counts[new_s] = s_counts.get(new_s, 0) + 1
                p_counts[new_p] = p_counts.get(new_p, 0) + 1
                
                new_score = max(len(s_counts), len(p_counts))
                
                # Accept if better or with probability based on temperature
                accept = new_score < cur_score or (new_score == cur_score and random.random() < 0.3) or \
                         (new_score > cur_score and random.random() < min(1.0, (cur_score - new_score) / T))
                
                if accept:
                    edge_set.remove(old_edge)
                    edge_set.add(new_edge)
                    edges[edges.index([old_edge[0], old_edge[1]])] = [new_edge[0], new_edge[1]]
                    cur_score = new_score
                    
                    if cur_score < best_local_score:
                        best_local_score = cur_score
                        best_local_vals = list(vals)
                        best_local_edges = [list(e) for e in edge_set]
                else:
                    # Undo
                    s_counts[new_s] -= 1
                    if s_counts[new_s] == 0:
                        del s_counts[new_s]
                    p_counts[new_p] -= 1
                    if p_counts[new_p] == 0:
                        del p_counts[new_p]
                    
                    s_counts[old_s] = s_counts.get(old_s, 0) + 1
                    p_counts[old_p] = p_counts.get(old_p, 0) + 1
            else:
                # Value perturbation: change one value
                v_idx = random.randint(0, n - 1)
                old_val = vals[v_idx]
                # Try a few candidate values
                candidates = []
                for delta in [-3, -2, -1, 1, 2, 3]:
                    nv = old_val + delta
                    if 1 <= nv <= 10000 and nv != old_val and nv not in vals:
                        candidates.append(nv)
                if not candidates:
                    continue
                nv = random.choice(candidates)
                
                # Recompute edge sums/prods for edges involving v_idx
                affected_edges = [e for e in edge_set if v_idx in e]
                
                # Save old values
                old_sums = {}
                old_prods = {}
                for e in affected_edges:
                    old_sums[e] = edge_sums[e]
                    old_prods[e] = edge_prods[e]
                
                # Apply value change
                vals[v_idx] = nv
                for e in affected_edges:
                    other = e[0] if e[1] == v_idx else e[1]
                    edge_sums[e] = vals[v_idx] + vals[other]
                    edge_prods[e] = vals[v_idx] * vals[other]
                
                # Update counters
                for e in affected_edges:
                    old_s = old_sums[e]
                    old_p = old_prods[e]
                    new_s = edge_sums[e]
                    new_p = edge_prods[e]
                    
                    s_counts[old_s] -= 1
                    if s_counts[old_s] == 0:
                        del s_counts[old_s]
                    p_counts[old_p] -= 1
                    if p_counts[old_p] == 0:
                        del p_counts[old_p]
                    
                    s_counts[new_s] = s_counts.get(new_s, 0) + 1
                    p_counts[new_p] = p_counts.get(new_p, 0) + 1
                
                new_score = max(len(s_counts), len(p_counts))
                
                accept = new_score < cur_score or (new_score == cur_score and random.random() < 0.2)
                
                if accept:
                    cur_score = new_score
                    if cur_score < best_local_score:
                        best_local_score = cur_score
                        best_local_vals = list(vals)
                        best_local_edges = [list(e) for e in edge_set]
                else:
                    # Undo value change
                    vals[v_idx] = old_val
                    for e in affected_edges:
                        edge_sums[e] = old_sums[e]
                        edge_prods[e] = old_prods[e]
                    
                    for e in affected_edges:
                        old_s = old_sums[e]
                        old_p = old_prods[e]
                        new_s = edge_sums[e]
                        new_p = edge_prods[e]
                        
                        s_counts[new_s] -= 1
                        if s_counts[new_s] == 0:
                            del s_counts[new_s]
                        p_counts[new_p] -= 1
                        if p_counts[new_p] == 0:
                            del p_counts[new_p]
                        
                        s_counts[old_s] = s_counts.get(old_s, 0) + 1
                        p_counts[old_p] = p_counts.get(old_p, 0) + 1
            
            T *= cooling
            if T < min_T:
                break
        
        return best_local_vals, list(best_local_edges), best_local_score
    
    # Main loop: try each value set with multiple restarts
    for vals in value_sets:
        if time.time() > deadline - 10:
            break
        
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
        
        edge_sums, edge_prods = compute_edge_maps(vals)
        greedy_eds, _ = greedy_edges(vals, edge_sums, edge_prods)
        
        # Run multiple restarts with local search
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Mix: some restarts use greedy init, others random
            if restart < num_restarts // 2:
                init_eds = greedy_eds
            else:
                init_eds = random.sample(all_edges, m)
            
            found_vals, found_eds, found_score = local_search(vals, init_eds, max_iters=20000)
            
            if found_score < best_score:
                best_score = found_score
                best_values = found_vals
                best_edges = found_eds
            
            # Early termination if we find a very good solution
            if best_score <= 9:
                break
    
    # Fallback
    if best_edges is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in random.sample(all_edges, m)]
    
    # Ensure correct types
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
