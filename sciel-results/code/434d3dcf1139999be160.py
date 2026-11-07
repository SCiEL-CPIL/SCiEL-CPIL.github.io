import itertools
import time
import random
import math

def construct():
    n = 14
    m = 35
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Candidate value sets
    vals1 = [1, 2, 4, 8, 3, 6, 12, 24, 16, 9, 18, 36, 72, 48]
    vals2 = list(range(1, 15))
    vals3 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    vals4 = [1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 30, 32, 36, 48]
    
    candidate_values = [vals1, vals2, vals3, vals4]
    
    # Add some randomized structured sets
    for _ in range(10):
        if time.time() > deadline - 15: break
        base = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96]
        random.shuffle(base)
        v = base[:14]
        # Simple mutation
        v = [x + random.randint(-2, 2) for x in v]
        v = [x if x > 0 else 1 for x in v]
        # Ensure distinct
        seen = set()
        unique_v = []
        for x in v:
            if x not in seen and x <= 10000:
                seen.add(x)
                unique_v.append(x)
        while len(unique_v) < 14:
            x = random.randint(1, 100)
            if x not in seen:
                seen.add(x)
                unique_v.append(x)
        candidate_values.append(unique_v)

    best_raw = float('inf')
    best_vals = None
    best_edges = None

    def optimize_edges(values):
        all_pairs = list(itertools.combinations(range(n), 2))
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            pair_sums[(u, v)] = values[u] + values[v]
            pair_prods[(u, v)] = values[u] * values[v]
        
        # Greedy initialization
        selected = set()
        current_sums = set()
        current_prods = set()
        remaining = set(all_pairs)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for e in remaining:
                s = pair_sums[e]
                p = pair_prods[e]
                new_sums_size = len(current_sums) + (1 if s not in current_sums else 0)
                new_prods_size = len(current_prods) + (1 if p not in current_prods else 0)
                score = max(new_sums_size, new_prods_size)
                if score < best_score:
                    best_score = score
                    best_edge = e
            if best_edge is None:
                break
            selected.add(best_edge)
            remaining.remove(best_edge)
            current_sums.add(pair_sums[best_edge])
            current_prods.add(pair_prods[best_edge])
        
        if len(selected) < m:
            for e in remaining:
                if len(selected) >= m: break
                selected.add(e)
                current_sums.add(pair_sums[e])
                current_prods.add(pair_prods[e])
        
        # Local Search with incremental updates
        current_edges = list(selected)
        
        # Helper to update sets and return score
        def get_score():
            return max(len(current_sums), len(current_prods))
        
        current_score = get_score()
        
        T = 1.0
        cooling = 0.95
        min_T = 0.01
        iterations = 0
        
        while T > min_T:
            if time.time() > deadline:
                break
            
            # Pick random edge to remove
            i = random.randint(0, m - 1)
            e_out = current_edges[i]
            
            # Pick random edge to add
            # Sample from remaining
            remaining_edges = [e for e in all_pairs if e not in selected]
            if not remaining_edges:
                break
            e_in = random.choice(remaining_edges)
            
            # Tentative swap
            # Remove e_out
            s_out = pair_sums[e_out]
            p_out = pair_prods[e_out]
            
            # Count occurrences in current_edges? 
            # We need to know if s_out/p_out are used by other edges.
            # This is expensive to check every time.
            # Alternative: Just recalculate score from scratch? 
            # 35 edges is small. Recalculating is O(35).
            # Let's try recalculating for simplicity and correctness first.
            
            old_edge = current_edges[i]
            current_edges[i] = e_in
            
            # Recalculate
            sums = set()
            prods = set()
            for u, v in current_edges:
                sums.add(values[u] + values[v])
                prods.add(values[u] * values[v])
            new_score = max(len(sums), len(prods))
            
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                # Update selected set
                selected.remove(old_edge)
                selected.add(e_in)
            else:
                current_edges[i] = old_edge
            
            T *= cooling
            iterations += 1
            
            # Limit iterations per value set to avoid spending too much time
            if iterations > 5000:
                break
        
        return current_edges, current_score

    for vals in candidate_values:
        if time.time() > deadline:
            break
        
        if len(set(vals)) != 14:
            continue
        if any(v <= 0 or v > 10000 for v in vals):
            continue
            
        edges, score = optimize_edges(vals)
        
        if score < best_raw:
            best_raw = score
            best_vals = vals
            best_edges = edges
            
    if best_vals is None:
        best_vals = vals1
        best_edges = optimize_edges(vals1)[0]
        
    vals_out = [int(x) for x in best_vals]
    edges_out = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': vals_out, 'edges': edges_out}
