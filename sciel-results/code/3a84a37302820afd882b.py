"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Strategy:
    # 1. Generate candidate value sets. 
    #    We want values that produce many collisions in sums and products.
    #    - Small integers produce small sums/products, increasing collision chance.
    #    - Arithmetic progressions or geometric progressions can create structure.
    #    - Randomized local search on the values themselves to find a "dense" set.
    # 2. For a given set of values, find the best 35 edges.
    #    - We want to pick edges (u,v) such that v[u]+v[v] and v[u]*v[v] 
    #      are in a small set of distinct values.
    #    - Greedy: Start with empty edge set. At each step, add the edge that 
    #      increases the distinct sum/product count the least (or keeps it minimal).
    #      Actually, we want to minimize max(|S|, |P|). 
    #      Let's try to keep both small. 
    #      A good heuristic: prioritize edges whose sum and product are already 
    #      "covered" by existing edges. If not covered, pick the one that adds 
    #      the least "new" distinct values.
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def find_best_edges(values):
        # Greedy edge selection
        # Precompute all possible sums and products for all pairs
        all_pairs = list(combinations(range(n), 2))
        pair_sums = {p: values[p[0]] + values[p[1]] for p in all_pairs}
        pair_prods = {p: values[p[0]] * values[p[1]] for p in all_pairs}
        
        # We want to select m edges.
        # Let's try a few different greedy strategies or just a robust one.
        # Strategy: Sort edges by "cost". 
        # But cost depends on current state.
        # Let's do a simple greedy:
        # 1. Start with empty set.
        # 2. For each step, evaluate all remaining edges.
        # 3. Score an edge by (1 if sum is new, 0 else) + (1 if prod is new, 0 else).
        #    Tie-breaker: maybe prefer edges with smaller sums/prods? Or just random.
        #    To be more effective, we can look ahead slightly or use a different metric.
        
        best_edges = []
        current_sums = set()
        current_prods = set()
        
        available_edges = set(all_pairs)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            # To speed up, we might not check all if time is tight, but 91 pairs is small.
            for e in available_edges:
                s = pair_sums[e]
                p = pair_prods[e]
                new_sum = 1 if s not in current_sums else 0
                new_prod = 1 if p not in current_prods else 0
                score = new_sum + new_prod
                # Tie-breaking: prefer edges that lead to smaller total distinct count?
                # Or just random tie-breaking to explore.
                if score < best_score:
                    best_score = score
                    best_edge = e
                elif score == best_score and best_edge is not None:
                    # Random tie-break to avoid local minima in greedy
                    if random.random() < 0.5:
                        best_edge = e
            
            if best_edge is None:
                break
                
            u, v = best_edge
            current_sums.add(pair_sums[best_edge])
            current_prods.add(pair_prods[best_edge])
            best_edges.append(list(best_edge))
            available_edges.remove(best_edge)
            
            if time.time() > deadline - 5:
                break
                
        return best_edges, evaluate(values, best_edges)

    # Generate initial values
    # Start with a range or random small numbers
    # Let's try a few different value generation strategies
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Strategy 1: Small integers 1..14
    values = list(range(1, n+1))
    edges, score = find_best_edges(values)
    if score < best_score:
        best_score = score
        best_values = values[:]
        best_edges = edges[:]
        
    # Strategy 2: Random small integers
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        vals = [random.randint(1, 20) for _ in range(n)]
        # Ensure distinct
        if len(set(vals)) != n:
            continue
        edges, score = find_best_edges(vals)
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges[:]
            
    # Strategy 3: Local search on values
    # Take best_values and perturb
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    # Simulated annealing-like or simple hill climbing on values
    # Since n=14, we can swap or increment/decrement values.
    for step in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Perturb one value
        new_values = current_values[:]
        idx = random.randint(0, n-1)
        delta = random.choice([-2, -1, 1, 2, -5, 5])
        new_val = current_values[idx] + delta
        if 1 <= new_val <= max_val and new_val not in [v for i, v in enumerate(new_values) if i != idx]:
            new_values[idx] = new_val
            
            # Re-optimize edges for these values
            edges, score = find_best_edges(new_values)
            
            if score < current_score:
                current_values = new_values
                current_edges = edges
                current_score = score
            elif score == current_score:
                # Accept with some probability? Or just reject to keep it simple and fast.
                # Let's accept to explore the plateau.
                if random.random() < 0.1:
                    current_values = new_values
                    current_edges = edges
                    current_score = score
    
    # Final check
    if len(best_edges) < m:
        # Ensure we have exactly 35 edges. If greedy failed, fill randomly.
        # But greedy should find 35 easily.
        pass
        
    # Ensure types are correct
    final_values = [int(v) for v in current_values]
    final_edges = [list(map(int, e)) for e in current_edges]
    
    # Verify constraints
    if len(final_values) != 14 or len(set(final_values)) != 14:
        # Fallback to seed
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]}
        
    if len(final_edges) != 35:
        # Fallback
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]}
        
    # Check for duplicates in edges
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    if len(edge_set) != 35:
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]}
        
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
