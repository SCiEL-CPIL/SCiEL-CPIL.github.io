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
    Generate a diverse initial solution.
    We start with a highly structured value set (powers of 2 or multiples) 
    and a random valid edge set.
    """
    # Try a few deterministic value structures to find a good starting point
    candidates = [
        [2**i for i in range(14)],
        [60 * i for i in range(1, 15)],
        list(range(1, 15)),
        [10 * i for i in range(1, 15)],
        [4 * i for i in range(1, 15)],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    ]
    
    best_vals = None
    best_score = float('inf')
    
    # Pre-compute all possible edges
    all_edges = list(combinations(range(14), 2))
    
    for vals in candidates:
        # Ensure validity
        if len(set(vals)) != 14 or any(v < 1 or v > 10000 for v in vals):
            continue
            
        # Greedy edge selection for this specific value set
        # This is fast enough for initialization
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining_edges = list(all_edges)
        random.shuffle(remaining_edges)
        
        # Simple greedy: pick edge that minimizes marginal cost
        # To make it faster for init, we just pick random and evaluate? 
        # No, let's do a quick greedy.
        for _ in range(35):
            best_e = None
            best_marginal = float('inf')
            # Sample a subset to speed up init if needed, but 91 is small
            for e in remaining_edges:
                u, v = e
                s_val = vals[u] + vals[v]
                p_val = vals[u] * vals[v]
                
                ns = len(current_sums) + (0 if s_val in current_sums else 1)
                np_ = len(current_prods) + (0 if p_val in current_prods else 1)
                marginal = max(ns, np_)
                
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_e = e
            
            if best_e is None:
                break
                
            selected_edges.append(best_e)
            remaining_edges.remove(best_e)
            u, v = best_e
            current_sums.add(vals[u] + vals[v])
            current_prods.add(vals[u] * vals[v])
            
        score = _evaluate_state(vals, set(selected_edges))
        if score < best_score:
            best_score = score
            best_vals = vals
            best_edges = selected_edges
            
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(combinations(range(14), 2))[:35]
        
    return best_vals, set(best_edges)

def _simulated_annealing_joint(start_time, deadline):
    """
    Joint optimization of values and edges.
    Moves:
    1. Perturb a value (small integer shift).
    2. Swap an edge (remove one, add another).
    3. Add/Remove edge (if size allows, but we must keep exactly 35).
       Actually, swapping is the only valid move for fixed size 35.
    """
    values, edges_set = _initial_solution()
    
    # Ensure types are int
    values = [int(v) for v in values]
    
    current_score = _evaluate_state(values, edges_set)
    best_score = current_score
    best_values = values.copy()
    best_edges = set(edges_set)
    
    # Pre-compute all possible edges for random selection
    all_edges = list(combinations(range(14), 2))
    edges_list = list(edges_set)
    
    # Temperature schedule
    # Start high enough to allow exploration, cool down
    temp = 10.0
    cooling_rate = 0.995
    
    last_improvement_time = start_time
    no_improve_count = 0
    
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline:
            break
            
        # Decide move type
        # 60% edge swap, 40% value perturbation
        if random.random() < 0.6:
            # Edge Swap
            if len(edges_list) < 2:
                continue
            e1_idx = random.randint(0, len(edges_list) - 1)
            e2_idx = random.randint(0, len(edges_list) - 1)
            if e1_idx == e2_idx:
                continue
                
            e1 = edges_list[e1_idx]
            e2 = edges_list[e2_idx]
            
            # Remove e1, add e2
            edges_set.discard(e1)
            if e2 in edges_set:
                # e2 is already there? No, e2 is from edges_list which mirrors edges_set.
                # But wait, if we pick two different indices, they are different edges.
                # So e2 is NOT in edges_set after removing e1? 
                # No, e2 was in edges_list, so it was in edges_set.
                # We removed e1. e2 is still in edges_set.
                # So we can't just add e2. We need to swap with a NON-edge.
                # Let's pick a random non-edge.
                non_edges = [e for e in all_edges if e not in edges_set]
                if not non_edges:
                    edges_set.add(e1) # revert
                    continue
                e_new = random.choice(non_edges)
                
                # Perform swap: remove e1, add e_new
                edges_set.discard(e1)
                edges_set.add(e_new)
                
                # Update edges_list to reflect change
                edges_list[e1_idx] = e_new
                
                new_score = _evaluate_state(values, edges_set)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = values.copy()
                        best_edges = set(edges_set)
                        last_improvement_time = time.time()
                else:
                    # Revert
                    edges_set.discard(e_new)
                    edges_set.add(e1)
                    edges_list[e1_idx] = e1
            else:
                # This case shouldn't happen if logic is correct
                pass
        else:
            # Value Perturbation
            idx = random.randint(0, 13)
            old_val = values[idx]
            # Small perturbation: +/- 1 to 5
            step = random.randint(-5, 5)
            new_val = old_val + step
            
            # Check constraints: 1 <= new_val <= 10000, distinct
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in values and values.index(new_val) != idx:
                continue
                
            values[idx] = new_val
            new_score = _evaluate_state(values, edges_set)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = values.copy()
                    best_edges = set(edges_set)
                    last_improvement_time = time.time()
            else:
                values[idx] = old_val
                
        # Cooling
        temp *= cooling_rate
        if temp < 0.01:
            temp = 0.01
            
        # Stagnation check: if no improvement for a long time, restart or cool faster?
        # For now, just keep going. The joint nature helps escape local minima.
        
    return best_values, best_edges

def construct():
    start_time = time.time()
    # Safety margin: 85s out of 90s
    deadline = start_time + 85.0
    
    best_global_score = float('inf')
    best_global_result = None
    
    # Run joint SA
    # We might want to run multiple restarts if time permits, but joint SA is expensive.
    # Let's do one long run first.
    
    values, edges_set = _simulated_annealing_joint(start_time, deadline)
    
    score = _evaluate_state(values, edges_set)
    
    if score < best_global_score:
        best_global_score = score
        best_global_result = {
            'values': [int(v) for v in values],
            'edges': [list(e) for e in edges_set]
        }
        
    # If we have time left, do a second restart with different initial values?
    if time.time() < deadline - 5.0:
        # Restart with a different random seed or structure
        random.seed() # Reseed
        values2, edges_set2 = _simulated_annealing_joint(start_time, deadline)
        score2 = _evaluate_state(values2, edges_set2)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_result = {
                'values': [int(v) for v in values2],
                'edges': [list(e) for e in edges_set2]
            }

    if best_global_result is None:
        # Fallback
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    # Final type checking and conversion
    res_values = [int(v) for v in best_global_result['values']]
    res_edges = [[int(u), int(v)] for u, v in best_global_result['edges']]
    
    return {
        'values': res_values,
        'edges': res_edges
    }
# EVOLVE-BLOCK-END
