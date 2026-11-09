"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _evaluate(values, edges):
    """Evaluate a configuration, returning score and detailed stats."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products)), len(sums), len(products)

def construct():
    """Construct values and edges minimizing max(distinct sums, distinct products)."""
    random.seed(42)
    
    n = 14
    num_edges = 35
    
    # Strategy: Use values that are divisors of a highly composite number
    # to encourage product collisions, then optimize edge selection.
    
    candidate_value_sets = []
    
    # Set 1: Small divisors with many factorization pairs
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18])
    
    # Set 2: More composite-heavy
    candidate_value_sets.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # Set 3: Powers and composites
    candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 24])
    
    # Set 4: All even numbers
    candidate_value_sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Set 5: Multiples of 6
    candidate_value_sets.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # Set 6: Mixed small composites
    candidate_value_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18])
    
    # Set 7: Focus on numbers with many divisors
    candidate_value_sets.append([2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24])
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    all_possible_edges = list(combinations(range(n), 2))
    
    for values in candidate_value_sets:
        if len(set(values)) != n:
            continue
        if any(v <= 0 or v > 10000 for v in values):
            continue
        
        edge_data = {}
        for i, j in all_possible_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data[(i, j)] = (s, p)
        
        selected_edges = []
        selected_sums = set()
        selected_products = set()
        remaining = set(all_possible_edges)
        
        for _ in range(num_edges):
            best_edge = None
            best_combined = float('inf')
            
            for e in remaining:
                s, p = edge_data[e]
                new_s = 0 if s in selected_sums else 1
                new_p = 0 if p in selected_products else 1
                combined = new_s + new_p
                if combined < best_combined:
                    best_combined = combined
                    best_edge = e
                elif combined == best_combined:
                    if s in selected_sums and p in selected_products:
                        best_edge = e
                        break
            
            if best_edge is None:
                break
                
            selected_edges.append(best_edge)
            s, p = edge_data[best_edge]
            selected_sums.add(s)
            selected_products.add(p)
            remaining.discard(best_edge)
        
        current_edges = selected_edges[:]
        current_score, _, _ = _evaluate(values, current_edges)
        
        for iteration in range(2000):
            out_idx = random.randint(0, len(current_edges) - 1)
            e_out = current_edges[out_idx]
            
            current_set = set(current_edges)
            available = [e for e in all_possible_edges if e not in current_set]
            if not available:
                break
            e_in = random.choice(available)
            
            new_edges = current_edges[:]
            new_edges[out_idx] = e_in
            
            new_score, _, _ = _evaluate(values, new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
            elif new_score == current_score and random.random() < 0.15:
                current_edges = new_edges
        
        if current_score < best_score:
            best_score = current_score
            best_values = values[:]
            best_edges = [list(e) for e in current_edges]
            
            if best_score <= 25:
                break
    
    if best_values is not None and best_edges is not None:
        values = best_values
        current_edges = [tuple(e) for e in best_edges]
        current_score, _, _ = _evaluate(values, current_edges)
        
        for iteration in range(5000):
            out_idx = random.randint(0, len(current_edges) - 1)
            e_out = current_edges[out_idx]
            
            current_set = set(current_edges)
            available = [e for e in all_possible_edges if e not in current_set]
            if not available:
                break
            e_in = random.choice(available)
            
            new_edges = current_edges[:]
            new_edges[out_idx] = e_in
            
            new_score, _, _ = _evaluate(values, new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
            elif new_score == current_score and random.random() < 0.1:
                current_edges = new_edges
        
        best_edges = [list(e) for e in current_edges]
        best_score = current_score
    
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in all_possible_edges[:35]]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
