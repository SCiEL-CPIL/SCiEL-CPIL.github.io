import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 14
    M = 35
    MAX_VAL = 10000
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    random.seed(42)
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def optimize_edges_for_values(values):
        """
        Given a set of values, find the best 35 edges to minimize max(distinct sums, distinct prods).
        Uses Greedy + Local Search.
        """
        all_pairs = list(combinations(range(N), 2))
        
        # Precompute sums and products for all pairs
        pair_data = []
        for u, v in all_pairs:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_data.append((u, v, s, p))
        
        # Greedy Selection
        # We want to pick M edges.
        # Score of a state is max(|S|, |P|).
        # We pick edges one by one.
        
        used_sums = set()
        used_prods = set()
        selected_edges = []
        remaining_pairs = pair_data[:]
        
        # Sort remaining pairs initially? 
        # Pairs with smaller sums/products might be "safer" or more likely to collide.
        # But collisions depend on the *set* of chosen edges.
        # Let's just do a standard greedy: pick the edge that minimizes the resulting max(|S|, |P|).
        
        for _ in range(M):
            best_score = float('inf')
            best_idx = -1
            best_s = None
            best_p = None
            
            for idx, (u, v, s, p) in enumerate(remaining_pairs):
                # Calculate new sets sizes if we add this edge
                # If s is in used_sums, len(used_sums) stays same. Else +1.
                new_sums_size = len(used_sums) + (0 if s in used_sums else 1)
                new_prods_size = len(used_prods) + (0 if p in used_prods else 1)
                
                score = max(new_sums_size, new_prods_size)
                
                if score < best_score:
                    best_score = score
                    best_idx = idx
                    best_s = s
                    best_p = p
                    
            if best_idx != -1:
                u, v, s, p = remaining_pairs.pop(best_idx)
                selected_edges.append((u, v))
                used_sums.add(s)
                used_prods.add(p)
                
        # Local Search: Try to improve by swapping edges
        # Current state: selected_edges, used_sums, used_prods
        # We need to be careful with the sets. 
        # It's easier to recalculate from scratch for the local search step if the graph is small, 
        # but 35 edges is small enough that we can be slightly smarter or just recalculate.
        # Recalculating 35 sums/prods is very fast.
        
        current_raw = max(len(used_sums), len(used_prods))
        selected_set = set(selected_edges)
        
        improved = True
        while improved:
            improved = False
            # Try all possible swaps: remove e1 from selected, add e2 from non-selected
            # To make this faster, we can just iterate through all non-selected pairs
            # and check if swapping with any selected pair improves the score.
            
            non_selected = [p for p in pair_data if (p[0], p[1]) not in selected_set]
            
            # To avoid O(M * (Total-M)) full recalculations which might be slow if we do many iterations,
            # we can just do a few passes.
            # 35 * 56 = ~2000 swaps per pass. Recalculation is 35 ops. 2000 * 35 = 70k ops. Very fast.
            
            for e_out_idx, e_out in enumerate(non_selected):
                u_out, v_out, s_out, p_out = e_out
                
                for e_in_idx, e_in in enumerate(list(selected_set)):
                    u_in, v_in = e_in
                    
                    # Simulate swap
                    # Remove e_in, Add e_out
                    # We need to know if s_in, p_in are unique in the current set of edges.
                    # This requires counting frequencies in the current selected edges.
                    
                    # Let's just recalculate the score for the swapped set.
                    # It's simpler and less error-prone.
                    new_edges = (selected_set - {e_in}) | {e_out}
                    
                    # Optimization: Quick check
                    # If we remove e_in, do we lose a unique sum/prod?
                    # If we add e_out, do we gain a new sum/prod?
                    
                    # Let's just compute the raw value for the new set.
                    sums_temp = set()
                    prods_temp = set()
                    for u, v in new_edges:
                        sums_temp.add(values[u] + values[v])
                        prods_temp.add(values[u] * values[v])
                    
                    new_raw = max(len(sums_temp), len(prods_temp))
                    
                    if new_raw < current_raw:
                        selected_set = new_edges
                        current_raw = new_raw
                        improved = True
                        break
                if improved:
                    break
            
            # Update non_selected for next iteration
            # non_selected = [p for p in pair_data if (p[0], p[1]) not in selected_set]
            
        return current_raw, [list(e) for e in selected_set]

    # Value Generation Strategies
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy 1: Small random integers
        if attempts % 5 == 0:
            # Try sets with many small numbers
            # Sample 14 distinct from 1..30
            values = random.sample(range(1, 31), N)
        elif attempts % 5 == 1:
            # Arithmetic Progression
            a = random.randint(1, 10)
            d = random.randint(1, 5)
            values = [a + i*d for i in range(N)]
            if max(values) > MAX_VAL:
                values = random.sample(range(1, 51), N)
        elif attempts % 5 == 2:
            # Mix of very small and slightly larger
            # e.g. 1..7 and then 10..16
            small = random.sample(range(1, 8), 7)
            med = random.sample(range(8, 20), 7)
            values = small + med
            if len(set(values)) < N:
                values = random.sample(range(1, 40), N)
        elif attempts % 5 == 3:
            # Powers of 2 mixed with small numbers? 
            # Products of powers of 2 are distinct. Sums might collide.
            # Maybe just random 1..20
            values = random.sample(range(1, 21), N)
        else:
            # Random 1..50
            values = random.sample(range(1, 51), N)
        
        # Ensure distinct and valid
        values = list(set(values))
        if len(values) < N:
            continue
        values = values[:N]
        
        # Sort values? Doesn't matter for the graph, but might help deterministic behavior in greedy?
        # No, greedy depends on the values.
        
        raw, edges = optimize_edges_for_values(values)
        
        if raw < best_raw:
            best_raw = raw
            best_values = values
            best_edges = edges
            
            # If we hit a very low number, we can stop or continue to find even better
            # Theoretical lower bound:
            # If all sums are same, 1. All prods same, 1. Max 1.
            # But with distinct positive integers, sums $v_i+v_j$ for $i<j$.
            # Min sum $v_1+v_2$. Max sum $v_{13}+v_{14}$.
            # With 35 edges, we can't have all sums same unless the graph is very specific and values are specific.
            # For small N, we can get quite low.
            if best_raw <= 10:
                # This is likely near optimal for small values.
                # We can break if we are confident, but let's run a bit more to be sure.
                pass

    # Fallback if no valid solution found (should not happen)
    if best_values is None:
        best_values = list(range(1, N+1))
        best_edges = [list(e) for e in list(combinations(range(N), 2))[:M]]
        best_raw = evaluate(best_values, best_edges)

    # Final verification and type checking
    # Ensure values are int
    best_values = [int(v) for v in best_values]
    # Ensure edges are list of lists of int
    best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
