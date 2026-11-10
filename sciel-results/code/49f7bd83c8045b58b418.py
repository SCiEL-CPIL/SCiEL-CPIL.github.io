import time
import random
import math
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def get_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def optimize_edges_for_values(values, time_budget=2.0):
    """
    Optimize edge selection for a fixed set of values.
    Uses Simulated Annealing to minimize max(distinct_sums, distinct_prods).
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    m = P['m']
    
    # Precompute sum and product for each edge
    edge_props = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_props[(i, j)] = (s, p)
        
    # Map sums and products to IDs to make counting faster? 
    # Sets of integers are fast enough for 91 items.
    
    def calc_score(edge_set):
        sums = set()
        prods = set()
        for e in edge_set:
            s, p = edge_props[e]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))
        
    # Initialize with a random set of edges
    current_edges = set(random.sample(all_edges, m))
    current_score = calc_score(current_edges)
    
    best_edges = set(current_edges)
    best_score = current_score
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.99
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_budget:
        # Pick an edge to remove and an edge to add
        e_out = rng.choice(list(current_edges))
        e_in = rng.choice(all_edges)
        
        if e_in == e_out:
            continue
            
        # Calculate new score efficiently?
        # For now, just recalculate. 35 edges is small.
        new_edges = set(current_edges)
        new_edges.remove(e_out)
        new_edges.add(e_in)
        
        new_score = calc_score(new_edges)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        temp *= cooling
        
    return list(best_edges), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    n = P['n']
    m = P['m']
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Candidate value sets
    candidates = []
    
    # 1. Powers of 2
    candidates.append([2**i for i in range(n)])
    
    # 2. Arithmetic Progression
    candidates.append(list(range(1, n+1)))
    candidates.append(list(range(2, 2*n+1)))
    candidates.append(list(range(10, 10*n+1)))
    
    # 3. Geometric with base 3
    candidates.append([3**i for i in range(n)])
    
    # 4. Random dense sets (small values)
    for _ in range(5):
        vals = sorted(random.sample(range(1, 50), n))
        candidates.append(vals)
        
    # 5. Random sparse sets
    for _ in range(5):
        vals = sorted(random.sample(range(1, 1000), n))
        candidates.append(vals)
        
    # 6. Structured sets with potential sum collisions
    # e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15 (skip 13 to create gaps?)
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15])
    
    # 7. Mixed: Small values for sum collisions, large for product separation?
    # Actually, we want product collisions too.
    # Try values like 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # This is powers of 2.
    
    # 8. Try to find sets where v_i * v_j = v_k * v_l often
    # e.g. 1, 2, 3, 6, 4, 8, 12, 24...
    # 1*6 = 2*3 = 6
    # 2*6 = 3*4 = 12
    # 1*12 = 2*6 = 3*4 = 12
    # This looks promising.
    # Let's generate a "multiplicative lattice" subset.
    vals = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    candidates.append(sorted(vals))
    
    # Evaluate each candidate
    for vals in candidates:
        if time.time() > deadline - 5:
            break
            
        # Ensure distinct and valid
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        # Optimize edges
        edges, score = optimize_edges_for_values(vals, time_budget=3.0)
        
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
            
    # If we have time, try to improve the best values found
    if best_values is not None and time.time() < deadline - 10:
        current_vals = list(best_values)
        current_edges = best_edges
        current_score = best_score
        
        rng = random.Random(123)
        start_improve = time.time()
        
        while time.time() - start_improve < (deadline - time.time() - 5):
            # Perturb one value
            i = rng.randrange(n)
            old_val = current_vals[i]
            
            # Try small changes
            delta = rng.randint(-10, 10)
            new_val = old_val + delta
            
            if new_val < 1 or new_val > 10000 or new_val in current_vals:
                continue
                
            current_vals[i] = new_val
            
            # Re-optimize edges with short budget
            new_edges, new_score = optimize_edges_for_values(current_vals, time_budget=1.0)
            
            if new_score < current_score:
                current_score = new_score
                current_edges = new_edges
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(current_vals)
                    best_edges = current_edges
            else:
                current_vals[i] = old_val
                
    # Final validation and return
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, _ = optimize_edges_for_values(best_values, time_budget=5.0)
        
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(a), int(b)] for a, b in best_edges]
    }
