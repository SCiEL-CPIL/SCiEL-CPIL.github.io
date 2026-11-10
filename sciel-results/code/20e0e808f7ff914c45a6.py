"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
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

    # Joint Simulated Annealing
    # State: values (list of 14 ints), edges (set of 35 pairs)
    
    # Initialize
    # Strategy: Start with small values to encourage product collisions, 
    # and a dense random graph.
    # Small values -> smaller products -> higher collision chance for products.
    # Close values -> smaller spread in sums -> higher collision chance for sums.
    
    # Initial values: random sample from 1..50
    current_values = random.sample(range(1, 51), n)
    all_edges = list(combinations(range(n), 2))
    current_edges = set(random.sample(all_edges, m))
    
    current_score = _raw_value(current_values, current_edges)
    
    best_values = current_values.copy()
    best_edges = set(current_edges)
    best_score = current_score
    
    # Annealing parameters
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.9995
    
    # Precompute all possible sums and products for current values?
    # No, values change, so we must recompute.
    # But we can speed up evaluation by caching if values don't change in a step?
    # In each step, we either change values OR edges, or both.
    # If we change values, all sums/prods change.
    # If we change edges, only specific sums/prods change.
    
    # To make evaluation fast, we maintain:
    # sum_counts: dict {sum_value: count}
    # prod_counts: dict {prod_value: count}
    # And the set of edges.
    
    # Let's implement a fast evaluation structure.
    
    def build_state(values, edges):
        sum_counts = {}
        prod_counts = {}
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        score = max(distinct_sums, distinct_prods)
        
        return sum_counts, prod_counts, score

    sum_counts, prod_counts, current_score = build_state(current_values, current_edges)
    best_score = current_score
    best_values = current_values.copy()
    best_edges = set(current_edges)
    
    T = T_start
    steps = 0
    max_steps = 500000 # Cap to prevent infinite loop if time check fails
    
    while time.time() < deadline - 2.0 and T > T_end and steps < max_steps:
        steps += 1
        
        # Perturbation
        # 50% chance to move an edge, 50% to move a value
        # Or sometimes both? Let's stick to single moves for simplicity and speed, 
        # but allow value moves to be more frequent if score is high?
        # Actually, joint moves might be needed to escape local minima where 
        # changing values alone hurts score (due to fixed edges) but enables 
        # better edge choices later.
        
        # Let's try:
        # 1. Randomly select a vertex i
        # 2. Randomly select a new value v' for i
        # 3. Calculate delta in score if we just change values[i] = v'
        # 4. Accept/Reject based on Metropolis
        
        # AND
        
        # 1. Randomly select an edge to remove (u, v)
        # 2. Randomly select an edge to add (x, y)
        # 3. Calculate delta in score
        # 4. Accept/Reject
        
        # We will alternate or randomize.
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Value move
            i = random.randint(0, n - 1)
            old_val = current_values[i]
            
            # Generate new value
            # Bias towards small values?
            # Let's try uniform 1..100 first, then if stuck, try local search?
            # No, global random restarts within value space are good for SA.
            # But 1..10000 is huge.
            # Let's sample from 1..500.
            new_val = random.randint(1, 500)
            
            if new_val == old_val:
                continue
            if new_val in current_values:
                continue # Must be distinct
            
            # Calculate new score
            # We need to update sum_counts and prod_counts
            # Edges incident to i are affected.
            # Find neighbors of i in current_edges
            neighbors = []
            for u, v in current_edges:
                if u == i:
                    neighbors.append(v)
                elif v == i:
                    neighbors.append(u)
            
            # Remove contributions of i
            temp_sum_counts = dict(sum_counts)
            temp_prod_counts = dict(prod_counts)
            
            for j in neighbors:
                s = old_val + current_values[j]
                p = old_val * current_values[j]
                
                temp_sum_counts[s] -= 1
                if temp_sum_counts[s] == 0:
                    del temp_sum_counts[s]
                    
                temp_prod_counts[p] -= 1
                if temp_prod_counts[p] == 0:
                    del temp_prod_counts[p]
            
            # Add contributions of i with new_val
            for j in neighbors:
                s = new_val + current_values[j]
                p = new_val * current_values[j]
                
                temp_sum_counts[s] = temp_sum_counts.get(s, 0) + 1
                temp_prod_counts[p] = temp_prod_counts.get(p, 0) + 1
            
            new_distinct_sums = len(temp_sum_counts)
            new_distinct_prods = len(temp_prod_counts)
            new_score = max(new_distinct_sums, new_distinct_prods)
            
            delta = new_score - current_score
            
            # Metropolis acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_values[i] = new_val
                sum_counts = temp_sum_counts
                prod_counts = temp_prod_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = current_values.copy()
                    best_edges = set(current_edges)
            else:
                # Reject, do nothing
                pass
                
        else:
            # Edge move
            # Remove one edge, add one edge
            
            if len(current_edges) == 0:
                continue
                
            e_out = random.choice(list(current_edges))
            
            # Find an edge to add
            # Randomly pick from all_edges not in current_edges
            # To be efficient, pick random from all_edges and check
            # Since m=35 and total edges=91, probability of collision is ~38%
            # So expected ~1.6 tries.
            
            e_in = random.choice(all_edges)
            while e_in in current_edges:
                e_in = random.choice(all_edges)
                
            # Calculate new score
            temp_sum_counts = dict(sum_counts)
            temp_prod_counts = dict(prod_counts)
            
            # Remove e_out
            u, v = e_out
            s_out = current_values[u] + current_values[v]
            p_out = current_values[u] * current_values[v]
            
            temp_sum_counts[s_out] -= 1
            if temp_sum_counts[s_out] == 0:
                del temp_sum_counts[s_out]
                
            temp_prod_counts[p_out] -= 1
            if temp_prod_counts[p_out] == 0:
                del temp_prod_counts[p_out]
                
            # Add e_in
            x, y = e_in
            s_in = current_values[x] + current_values[y]
            p_in = current_values[x] * current_values[y]
            
            temp_sum_counts[s_in] = temp_sum_counts.get(s_in, 0) + 1
            temp_prod_counts[p_in] = temp_prod_counts.get(p_in, 0) + 1
            
            new_distinct_sums = len(temp_sum_counts)
            new_distinct_prods = len(temp_prod_counts)
            new_score = max(new_distinct_sums, new_distinct_prods)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_edges.remove(e_out)
                current_edges.add(e_in)
                sum_counts = temp_sum_counts
                prod_counts = temp_prod_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = current_values.copy()
                    best_edges = set(current_edges)
            else:
                pass
                
        # Cool down
        T *= cooling_rate
        
        # Occasionally, if stuck (no improvement in last N steps), do a bigger kick?
        # SA usually handles this with temperature.
        # But if T is low and we are in a local minimum, we might get stuck.
        # The "Value move" allows changing the landscape itself.
        
    # Final refinement: Local search on best found
    # Try to improve best_edges with best_values using strict hill climbing
    if time.time() < deadline - 1.0:
        vals = best_values
        edges = set(best_edges)
        score = best_score
        
        all_edges_list = list(all_edges)
        # Precompute sums/prods for these values
        sum_map = {}
        prod_map = {}
        for i, j in all_edges_list:
            sum_map[(i,j)] = vals[i] + vals[j]
            prod_map[(i,j)] = vals[i] * vals[j]
            
        def eval_edges(edge_set):
            s_counts = {}
            p_counts = {}
            for e in edge_set:
                s = sum_map[e]
                p = prod_map[e]
                s_counts[s] = s_counts.get(s, 0) + 1
                p_counts[p] = p_counts.get(p, 0) + 1
            return max(len(s_counts), len(p_counts))
            
        # Simple hill climbing
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try all possible swaps? Too slow (91*86 ~ 8000 swaps).
            # Try random swaps.
            for _ in range(1000):
                if time.time() > deadline - 0.5:
                    break
                e_out = random.choice(list(edges))
                e_in = random.choice(all_edges_list)
                if e_in in edges:
                    continue
                    
                new_edges = set(edges)
                new_edges.remove(e_out)
                new_edges.add(e_in)
                
                new_score = eval_edges(new_edges)
                if new_score < score:
                    edges = new_edges
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_edges = set(edges)
                        best_values = vals.copy()
                        
    # Ensure types are correct
    # values: list of ints
    # edges: list of lists of ints
    
    final_values = [int(v) for v in best_values]
    final_edges = [list(e) for e in best_edges]
    # Sort edges for consistency? Not required, but nice.
    final_edges.sort(key=lambda x: (x[0], x[1]))
    
    # Double check validity
    if len(final_values) != 14:
        final_values = list(range(1, 15))
    if len(set(final_values)) != 14:
        final_values = list(range(1, 15))
    if any(v < 1 or v > 10000 for v in final_values):
        final_values = list(range(1, 15))
        
    if len(final_edges) != 35:
        # Fallback to random
        final_edges = [list(e) for e in random.sample(all_edges, 35)]
        
    # Check for duplicates/loops in edges
    seen = set()
    valid_edges = []
    for e in final_edges:
        if e[0] == e[1]:
            continue
        key = tuple(sorted(e))
        if key in seen:
            continue
        seen.add(key)
        valid_edges.append(e)
        
    # If we lost edges due to invalidity, refill
    if len(valid_edges) < 35:
        # This should not happen with our generation, but just in case
        all_possible = [list(e) for e in all_edges if tuple(sorted(e)) not in seen]
        random.shuffle(all_possible)
        while len(valid_edges) < 35 and all_possible:
            valid_edges.append(all_possible.pop())
            
    # If still not 35, something is very wrong.
    if len(valid_edges) != 35:
        valid_edges = [list(e) for e in random.sample(all_edges, 35)]

    return {
        'values': final_values,
        'edges': valid_edges
    }
# EVOLVE-BLOCK-END
