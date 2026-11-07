"""Strategy: Deterministic structured value pools and greedy maximum-weight clique approximation for edge selection."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    # 1. Generate candidate value sets.
    # We use "smooth numbers" (numbers with only small prime factors) because 
    # products of smooth numbers are more likely to collide with other products,
    # and sums of smooth numbers in specific ranges can also be controlled.
    
    candidate_sets = []
    
    # Set A: Powers of 2 and 3 (classic multiplicative structure)
    set_a = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    candidate_sets.append(set_a)
    
    # Set B: Powers of 2, 3, and 5 (more collision potential)
    set_b = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidate_sets.append(set_b)
    
    # Set C: Arithmetic progression with small steps (good for sums)
    set_c = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    candidate_sets.append(set_c)
    
    # Set D: Mixed small integers
    set_d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidate_sets.append(set_d)
    
    # Set E: Geometric progression base 2, then perturb
    set_e = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    candidate_sets.append(set_e)
    
    # Set F: Randomized smooth numbers
    smooth_primes = [2, 3, 5, 7]
    random.seed(42) # Deterministic for reproducibility
    set_f = set()
    while len(set_f) < n:
        v = 1
        for _ in range(random.randint(1, 5)):
            v *= random.choice(smooth_primes)
        if v <= 10000:
            set_f.add(v)
    candidate_sets.append(list(set_f))
    
    # Set G: Mixed smooth numbers with more diversity
    random.seed(123)
    set_g = set()
    while len(set_g) < n:
        v = random.randint(1, 50)
        if v <= 10000:
            set_g.add(v)
    candidate_sets.append(list(set_g))
    
    # Set H: Numbers with many divisors (highly composite-like)
    set_h = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidate_sets.append(set_h)
    
    def evaluate(values, edges):
        distinct_sums = set()
        distinct_prods = set()
        for u, v in edges:
            distinct_sums.add(values[u] + values[v])
            distinct_prods.add(values[u] * values[v])
        return max(len(distinct_sums), len(distinct_prods))

    def select_edges_greedy(values):
        """
        Select m edges using a greedy approach that prioritizes edges 
        that create the most collisions (i.e., reuse existing sum/product values).
        """
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sums and products for all edges
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_sum[(u, v)] = s
            edge_prod[(u, v)] = p
            
        # Count frequency of each sum and product across ALL possible edges
        sum_freq = {}
        prod_freq = {}
        for u, v in all_edges:
            s = edge_sum[(u, v)]
            p = edge_prod[(u, v)]
            sum_freq[s] = sum_freq.get(s, 0) + 1
            prod_freq[p] = prod_freq.get(p, 0) + 1
            
        selected_edges = []
        current_sums = set()
        current_prods = set()
        
        best_edges = []
        remaining_edges = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_score = -1
            for e in remaining_edges:
                u, v = e
                s = edge_sum[e]
                p = edge_prod[e]
                score = 0
                if s in current_sums:
                    score += 10
                if p in current_prods:
                    score += 10
                # Tie-breaker: global frequency
                score += sum_freq[s] * 0.1 + prod_freq[p] * 0.1
                if score > best_score:
                    best_score = score
                    best_edge = e
                    
            if best_edge:
                best_edges.append(best_edge)
                remaining_edges.remove(best_edge)
                current_sums.add(edge_sum[best_edge])
                current_prods.add(edge_prod[best_edge])
            else:
                break
                
        return best_edges

    def select_edges_sa(values, initial_edges, max_iters=5000, t_start=1.0, t_end=0.01):
        """
        Simulated Annealing to optimize edge selection.
        """
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sums and products for all edges
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_sum[(u, v)] = s
            edge_prod[(u, v)] = p
        
        # Convert initial edges to a set for O(1) lookup
        current_edges = set(initial_edges)
        current_score = evaluate(values, list(current_edges))
        best_edges = set(current_edges)
        best_score = current_score
        
        # Temperature schedule
        t = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / max_iters)
        
        random.seed(42)
        
        for i in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            # Generate a neighbor: swap one edge
            if len(current_edges) == 0:
                break
                
            # Pick a random edge to remove
            edges_list = list(current_edges)
            remove_idx = random.randint(0, len(edges_list) - 1)
            remove_edge = edges_list[remove_idx]
            
            # Pick a random edge to add (not already in current)
            new_edge = None
            attempts = 0
            while new_edge is None and attempts < 50:
                candidate = random.choice(all_edges)
                if candidate not in current_edges:
                    new_edge = candidate
                attempts += 1
            
            if new_edge is None:
                continue
                
            # Compute delta score
            # Remove remove_edge, add new_edge
            new_edges_set = set(current_edges)
            new_edges_set.discard(remove_edge)
            new_edges_set.add(new_edge)
            
            if len(new_edges_set) != m:
                continue
                
            new_score = evaluate(values, list(new_edges_set))
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/t)
            if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-9)):
                current_edges = new_edges_set
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
            
            t *= cooling_rate
        
        return list(best_edges)

    best_result = None
    best_score = float('inf')
    
    # Evaluate each candidate set with greedy edge selection first
    for i, vals in enumerate(candidate_sets):
        if time.time() > deadline:
            break
            
        # Ensure distinctness and range
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        edges = select_edges_greedy(vals)
        if len(edges) < m:
            continue
            
        score = evaluate(vals, edges)
        
        if score < best_score:
            best_score = score
            best_result = (list(vals), list(edges))
            
    # Now apply SA to improve the best result
    if time.time() < deadline - 5.0 and best_result:
        current_vals, current_edges = best_result
        current_score = best_score
        
        # Run SA with multiple restarts
        sa_restarts = 5
        for restart in range(sa_restarts):
            if time.time() > deadline - 3.0:
                break
                
            # Slightly perturb initial edges for diversity
            perturbed_edges = list(current_edges)
            if restart > 0:
                # Swap a few edges
                for _ in range(restart * 2):
                    idx = random.randint(0, len(perturbed_edges) - 1)
                    all_edges = list(itertools.combinations(range(n), 2))
                    new_edge = random.choice(all_edges)
                    if new_edge not in perturbed_edges:
                        perturbed_edges[idx] = new_edge
                
                if len(set(perturbed_edges)) != m:
                    perturbed_edges = list(current_edges)
            
            sa_edges = select_edges_sa(current_vals, perturbed_edges, max_iters=3000)
            if len(sa_edges) == m:
                sa_score = evaluate(current_vals, sa_edges)
                if sa_score < best_score:
                    best_score = sa_score
                    best_result = (list(current_vals), list(sa_edges))
                    current_edges = sa_edges
                    current_score = sa_score
            
            # Also try SA with different value sets
            if restart == 0:
                # Try SA on other promising value sets
                for j, vals2 in enumerate(candidate_sets):
                    if j == i or time.time() > deadline - 3.0:
                        continue
                    if len(set(vals2)) != n or any(v < 1 or v > 10000 for v in vals2):
                        continue
                    
                    # Get greedy edges for this value set
                    greedy_edges = select_edges_greedy(vals2)
                    if len(greedy_edges) < m:
                        continue
                    
                    sa_edges2 = select_edges_sa(vals2, greedy_edges, max_iters=2000)
                    if len(sa_edges2) == m:
                        sa_score2 = evaluate(vals2, sa_edges2)
                        if sa_score2 < best_score:
                            best_score = sa_score2
                            best_result = (list(vals2), list(sa_edges2))
    
    # Final local search polish
    if time.time() < deadline - 1.0 and best_result:
        current_vals, current_edges = best_result
        current_score = best_score
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Try edge swaps
        for _ in range(500):
            if time.time() > deadline - 0.5:
                break
                
            new_edges = list(current_edges)
            idx = random.randint(0, len(new_edges) - 1)
            old_edge = new_edges[idx]
            
            # Try to find a better edge to swap in
            best_swap = None
            best_swap_score = current_score
            
            # Sample some candidate edges
            for _ in range(20):
                new_edge = random.choice(all_edges)
                if new_edge == old_edge or new_edge in new_edges:
                    continue
                
                candidate_edges = list(new_edges)
                candidate_edges[idx] = new_edge
                
                if len(set(candidate_edges)) == m:
                    cand_score = evaluate(current_vals, candidate_edges)
                    if cand_score < best_swap_score:
                        best_swap_score = cand_score
                        best_swap = (idx, new_edge)
            
            if best_swap is not None:
                idx, new_edge = best_swap
                new_edges[idx] = new_edge
                current_edges = new_edges
                current_score = best_swap_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_result = (list(current_vals), list(current_edges))
    
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(n), 2))[:m]
        return {'values': vals, 'edges': [[int(u), int(v)] for u, v in edges]}
        
    vals, edges = best_result
    return {'values': [int(v) for v in vals], 'edges': [[int(u), int(v)] for u, v in edges]}
# EVOLVE-BLOCK-END
