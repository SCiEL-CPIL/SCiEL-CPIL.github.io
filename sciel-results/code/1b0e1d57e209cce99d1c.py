"""Strategy: Joint Simulated Annealing on values with greedy edge selection, leveraging multiplicative orbits for product collisions."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    def get_edges_and_score(values):
        """
        Given a list of 14 distinct values, greedily select 35 edges 
        that minimize the max(distinct_sums, distinct_products).
        Returns (edges, score).
        """
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sums and products for all potential edges
        edge_sums = {}
        edge_prods = {}
        for u, v in all_edges:
            edge_sums[(u, v)] = values[u] + values[v]
            edge_prods[(u, v)] = values[u] * values[v]
        
        # Greedy selection:
        # We want to pick edges that reuse existing sum/product values.
        # Score of an edge = (is_sum_new, is_prod_new, -freq_sum, -freq_prod)
        # We pick the edge with the minimum score tuple.
        
        selected_edges = []
        current_sums = set()
        current_prods = set()
        remaining_edges = set(all_edges)
        
        # To speed up, we can pre-sort or use a priority queue, but for n=14 (91 edges), 
        # a simple scan is fast enough. 35 iterations * 91 edges = ~3000 ops. Very fast.
        
        # However, to make it even better, we can look ahead slightly or use a better heuristic.
        # Let's stick to simple greedy first, but sort candidates by "potential collision".
        
        # Actually, a simple greedy that picks the edge with the smallest 
        # (new_sum, new_prod) indicator is good.
        
        for _ in range(m):
            best_edge = None
            best_key = None
            
            for e in remaining_edges:
                s = edge_sums[e]
                p = edge_prods[e]
                
                is_new_s = 0 if s in current_sums else 1
                is_new_p = 0 if p in current_prods else 1
                
                # We prefer edges that add no new sums and no new products.
                # If tied, prefer those with higher global frequency (more likely to collide later?)
                # Or just deterministic tie-breaking.
                
                # Let's use a key that minimizes (is_new_s, is_new_p)
                # If we have a choice, maybe pick the one that "uses up" common values?
                # For now, simple lexicographic.
                
                key = (is_new_s, is_new_p)
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = e
            
            if best_edge is None:
                break
                
            selected_edges.append(best_edge)
            remaining_edges.remove(best_edge)
            current_sums.add(edge_sums[best_edge])
            current_prods.add(edge_prods[best_edge])
            
        if len(selected_edges) < m:
            return None, float('inf')
            
        score = max(len(current_sums), len(current_prods))
        return selected_edges, score

    def evaluate_full(values):
        """
        Full evaluation for SA: 
        1. Check validity (distinct, 1-10000).
        2. Run greedy edge selection.
        3. Return score.
        """
        if len(set(values)) != n:
            return float('inf')
        if any(v < 1 or v > 10000 for v in values):
            return float('inf')
        
        edges, score = get_edges_and_score(values)
        return score

    # Initial Value Generation:
    # Use "Orbits" of powers. 
    # Base 2 powers: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # These have very few distinct products? No, products of powers of 2 are powers of 2.
    # Sums of powers of 2 are distinct (binary representation). 
    # So sums will be 35 distinct. Products will be powers of 2.
    # Distinct products: max exponent sum is log2(8192*4096) ~ 23. Min is 0.
    # So distinct products are roughly 24.
    # Max(35, 24) = 35.
    
    # Can we do better?
    # If we use a mix, e.g. {1, 2, 3, 4, 6, ...}
    
    # Let's start with a few diverse initial states for SA.
    
    init_values_1 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    init_values_2 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36] # Smooth numbers
    init_values_3 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140] # Arithmetic
    
    best_score = float('inf')
    best_values = None
    
    # Pre-check initial states
    for init in [init_values_1, init_values_2, init_values_3]:
        score = evaluate_full(init)
        if score < best_score:
            best_score = score
            best_values = list(init)
            
    # Simulated Annealing on Values
    # Move: Pick a random index i, change values[i] to a new random valid integer.
    # To keep "orbit" structure, we can bias the new value to be a multiple of a small prime 
    # or a power, but random search over 1..10000 is huge.
    # Better move: Swap values? No, values are distinct.
    # Better move: Pick i, j. Add/subtract a small delta? 
    # Or: Pick i, replace values[i] with a random smooth number?
    
    # Let's use a hybrid move:
    # 1. 50%: Pick i, replace values[i] with a random integer in [1, 10000].
    # 2. 50%: Pick i, replace values[i] with a random "smooth" number (factors 2,3,5,7).
    
    def random_smooth_number(max_val=10000):
        primes = [2, 3, 5, 7]
        v = 1
        while v <= max_val:
            v *= random.choice(primes)
            if random.random() < 0.3:
                break
        return min(v, max_val)

    current_values = best_values
    current_score = best_score
    
    t_start = 50.0
    t_end = 0.1
    max_iters = 20000 # High number, but we break on time
    
    t = t_start
    cooling_rate = (t_end / t_start) ** (1.0 / max_iters)
    
    random.seed(42)
    
    for i in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        # Generate neighbor
        new_values = list(current_values)
        idx = random.randint(0, n - 1)
        
        if random.random() < 0.5:
            new_val = random.randint(1, 10000)
        else:
            new_val = random_smooth_number()
            
        new_values[idx] = new_val
        
        # Check validity
        if len(set(new_values)) != n:
            # Collision! Reject or try to fix? 
            # Simplest: Reject.
            # But this might stall if we keep hitting collisions.
            # Let's try a few retries? Or just skip.
            continue
            
        if any(v < 1 or v > 10000 for v in new_values):
            continue
            
        new_score = evaluate_full(new_values)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-9)):
            current_values = new_values
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = list(current_values)
                
        t *= cooling_rate
        
    # Final Polish: Local Search on Edges for the Best Values
    # The greedy edge selection inside evaluate_full is fast but might not be optimal.
    # We can run a local search on the edges for the best_values found.
    
    if best_values:
        # Get the greedy edges
        greedy_edges, _ = get_edges_and_score(best_values)
        
        # Local Search on Edges
        # Swap one edge with another to improve score.
        all_edges = list(itertools.combinations(range(n), 2))
        
        current_edges = greedy_edges
        current_edge_score = evaluate_full(best_values) # This re-runs greedy, so it's consistent.
        
        # To make local search work, we need a faster evaluation that takes edges as input.
        def eval_edges(values, edges):
            sums = set()
            prods = set()
            for u, v in edges:
                sums.add(values[u] + values[v])
                prods.add(values[u] * values[v])
            return max(len(sums), len(prods))
        
        current_edge_score = eval_edges(best_values, current_edges)
        best_edge_set = set(current_edges)
        
        # Precompute edge props
        edge_props = {}
        for u, v in all_edges:
            edge_props[(u, v)] = (values[u] + values[v], values[u] * values[v])
            
        # Simple local search: try swapping each edge with a random non-edge
        # Limit iterations by time
        ls_deadline = deadline - 1.0
        ls_iter = 0
        while time.time() < ls_deadline and ls_iter < 5000:
            ls_iter += 1
            if not current_edges:
                break
                
            # Pick a random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            rem_edge = current_edges[idx]
            
            # Pick a random edge to add
            add_edge = random.choice(all_edges)
            if add_edge in current_edges:
                continue
                
            # Form new set
            new_edges_list = list(current_edges)
            new_edges_list[idx] = add_edge
            new_edges_set = set(new_edges_list)
            
            if len(new_edges_set) != m:
                continue
                
            new_score = eval_edges(best_values, new_edges_list)
            
            if new_score < current_edge_score:
                current_edges = new_edges_list
                current_edge_score = new_score
                best_edge_set = new_edges_set
                
    # Construct Result
    if best_values is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(n), 2))[:m]
        return {'values': vals, 'edges': [[int(u), int(v)] for u, v in edges]}
        
    # Ensure edges correspond to best_values
    # If we did local search, use best_edge_set. Otherwise use greedy.
    # Note: best_edge_set might be from a different value set if we didn't update it correctly?
    # In the code above, local search only runs on best_values.
    
    if 'best_edge_set' in locals() and best_edge_set:
        edges = list(best_edge_set)
    else:
        edges, _ = get_edges_and_score(best_values)
        if edges is None:
             # Should not happen
             edges = list(itertools.combinations(range(n), 2))[:m]
             
    # Final Validation & Type Conversion
    vals = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    # Sanity check
    if len(set(vals)) != 14 or any(v < 1 or v > 10000 for v in vals):
         # Fallback if something went wrong
         vals = list(range(1, 15))
         edges = list(itertools.combinations(range(n), 2))[:m]
         edges = [[int(u), int(v)] for u, v in edges]
         
    if len(set(tuple(sorted(e)) for e in edges)) != 35:
         # Fallback
         edges = list(itertools.combinations(range(n), 2))[:35]
         edges = [[int(u), int(v)] for u, v in edges]
         
    return {'values': vals, 'edges': edges}

# EVOLVE-BLOCK-END
