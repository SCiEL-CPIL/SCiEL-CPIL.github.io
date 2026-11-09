"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _evaluate_state(values, edges_set):
    """
    Fast evaluation using sets.
    """
    sums = set()
    prods = set()
    for u, v in edges_set:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _initial_solution():
    """
    Generate a high-quality initial solution.
    We use small integers because they provide the highest density of additive collisions.
    For products, small numbers also help keep the product space relatively compact compared to large numbers.
    """
    # Start with 1..14 which is known to be good for sums.
    values = list(range(1, 15))
    
    # Find a good set of 35 edges for these values using a quick greedy/local search
    # We want edges that collide.
    # For values 1..14, sums range from 3 to 27. Products range from 2 to 182.
    # Ideally we want to reuse sum values and product values.
    
    # Let's just pick a random set of 35 edges and refine it quickly.
    all_edges = list(combinations(range(14), 2))
    random.shuffle(all_edges)
    edges = set(all_edges[:35])
    
    # Quick local search to improve this specific edge set for fixed values
    current_score = _evaluate_state(values, edges)
    best_edges = set(edges)
    best_score = current_score
    
    start = time.time()
    while time.time() - start < 1.0:
        if not edges:
            break
        remove_edge = random.choice(tuple(edges))
        add_edge = random.choice(all_edges)
        if add_edge in edges:
            continue
            
        edges.remove(remove_edge)
        edges.add(add_edge)
        
        new_score = _evaluate_state(values, edges)
        if new_score <= best_score:
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edges)
        else:
            # Revert
            edges.remove(add_edge)
            edges.add(remove_edge)
            
    return values, [list(e) for e in best_edges], best_score

def _simulated_annealing_joint(start_values, start_edges, time_limit):
    """
    Joint Simulated Annealing on values and edges.
    
    Moves:
    1. Perturb a value: v[i] += delta (small random integer), ensuring distinctness and bounds.
    2. Swap an edge: remove one, add one.
    
    Objective: Minimize max(distinct_sums, distinct_prods).
    """
    n = len(start_values)
    m = len(start_edges)
    
    values = list(start_values)
    edges = set(tuple(sorted(e)) for e in start_edges)
    
    current_score = _evaluate_state(values, edges)
    best_score = current_score
    best_values = list(values)
    best_edges = set(edges)
    
    start_time = time.time()
    temp = 10.0
    cooling = 0.9999
    
    # Pre-compute all possible edges for faster random selection
    all_edges = list(combinations(range(n), 2))
    
    while time.time() - start_time < time_limit:
        # Choose move type
        if random.random() < 0.5:
            # Move 1: Perturb Value
            idx = random.randint(0, n - 1)
            old_val = values[idx]
            # Small perturbation to maintain density
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Constraints: 1 <= v <= 10000, distinct
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in values and new_val != old_val:
                continue
                
            values[idx] = new_val
            new_score = _evaluate_state(values, edges)
            
            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < pow(2.71828, -delta_score / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(values)
                    best_edges = set(edges)
            else:
                values[idx] = old_val
        else:
            # Move 2: Swap Edge
            if not edges:
                continue
            remove_edge = random.choice(tuple(edges))
            add_edge = random.choice(all_edges)
            if add_edge in edges:
                continue
                
            edges.remove(remove_edge)
            edges.add(add_edge)
            
            new_score = _evaluate_state(values, edges)
            
            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < pow(2.71828, -delta_score / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(values)
                    best_edges = set(edges)
            else:
                edges.remove(add_edge)
                edges.add(remove_edge)
                
        temp *= cooling
        if temp < 0.01:
            break
            
    return best_values, [list(e) for e in best_edges], best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    # We will run multiple restarts with different initial value structures
    # and different initial edge sets.
    
    # Strategy:
    # 1. Start with small integers (1-14) as they offer the best additive collision potential.
    # 2. Occasionally try other structures like multiples or primes, but small ints are usually best for max(sums, prods).
    
    initial_structures = [
        list(range(1, 15)),
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
        list(range(2, 16)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2 (bad for sums, good for prods structure)
        [60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840] # Multiples
    ]
    
    random.shuffle(initial_structures)
    
    for init_vals in initial_structures:
        if time.time() > deadline:
            break
            
        # Ensure validity of initial values
        init_vals = [int(v) for v in init_vals]
        if len(set(init_vals)) != 14 or any(v < 1 or v > 10000 for v in init_vals):
            continue
            
        # Get a decent starting point
        vals, edges, score = _initial_solution()
        
        # If this specific structure is bad, skip heavy optimization
        if score > 40: 
            continue
            
        # Run joint SA
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        sa_time = min(5.0, remaining / 2.0)
        
        if sa_time > 0.5:
            best_vals, best_edges, best_score = _simulated_annealing_joint(vals, edges, sa_time)
            
            if best_score < best_global_score:
                best_global_score = best_score
                best_global_result = {
                    'values': best_vals,
                    'edges': best_edges
                }
                
                # Early stop if we get very good
                if best_global_score <= 10:
                    break

    if best_global_result is None:
        # Fallback to a known good-ish result
        vals, edges, score = _initial_solution()
        best_global_result = {
            'values': vals,
            'edges': edges
        }
        
    # Final type checking and conversion
    res_values = [int(v) for v in best_global_result['values']]
    res_edges = [[int(u), int(v)] for u, v in best_global_result['edges']]
    
    return {
        'values': res_values,
        'edges': res_edges
    }
# EVOLVE-BLOCK-END
