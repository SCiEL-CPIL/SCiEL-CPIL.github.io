import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n = len(all_triples)
    
    # Precompute pair sets for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
    
    # Build conflict graph: two triples conflict if they share a pair
    # We want to select a maximum set of non-conflicting triples
    # This is equivalent to finding a maximum independent set in the conflict graph
    # Or equivalently, a maximum matching in the hypergraph
    
    # For MILP: x[i] = 1 if we select triple i
    # Maximize sum(x[i])
    # Subject to: for each pair p, sum(x[i] for i where p in triple[i]) <= 1
    
    # Build constraint matrix: rows = pairs, cols = triples
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    pair_to_idx = {p: idx for idx, p in enumerate(all_pairs)}
    n_pairs = len(all_pairs)
    
    # A[i, j] = 1 if pair j is in triple i
    A = np.zeros((n_pairs, n), dtype=np.int8)
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            A[pair_to_idx[p], i] = 1
    
    # Objective: maximize sum(x)
    c = -np.ones(n)  # minimize -sum(x)
    
    # Constraints: A @ x <= 1
    constraints = [LinearConstraint(A, 0, 1)]
    
    # Bounds: 0 <= x[i] <= 1
    bounds = Bounds(0, 1)
    
    # Integrate variables
    integrality = np.ones(n)
    
    # Solve MILP
    best_alpha = 16
    best_triples = []
    
    def has_independent_set_size(triple_list, k):
        masks = []
        for t in triple_list:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(mask)
            
        for subset in combinations(points, k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
                
            is_ind = True
            for m in masks:
                if (m & mask) == m:
                    is_ind = False
                    break
            if is_ind:
                return True
        return False

    def get_alpha(triple_list, max_k=16):
        for k in range(max_k, 0, -1):
            if has_independent_set_size(triple_list, k):
                return k
        return 0
    
    # First, try to solve the full MILP
    try:
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if res.status == 0 and res.x is not None:
            selected = [i for i in range(n) if res.x[i] > 0.5]
            current_triples = [list(all_triples[i]) for i in selected]
            
            if current_triples:
                alpha = get_alpha(current_triples)
                best_alpha = alpha
                best_triples = current_triples
    except Exception:
        pass
    
    # If MILP didn't find a good solution, fall back to random search
    if best_alpha == 16:
        seed_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        best_alpha = get_alpha(seed_triples)
        best_triples = seed_triples
    
    # Try random restarts to find better solutions
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        
        current_triples = []
        used_pairs = set()
        
        for idx in indices:
            t = all_triples[idx]
            pairs = triple_pairs[idx]
            
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
                
            if conflict:
                continue
                
            current_triples.append(t)
            for p in pairs:
                used_pairs.add(p)
                
        if not current_triples:
            continue
            
        # Check if this is better than best_alpha
        if has_independent_set_size(current_triples, best_alpha):
            continue
            
        # It might be better. Find the actual alpha.
        if has_independent_set_size(current_triples, best_alpha - 1):
            new_alpha = best_alpha - 1
        else:
            new_alpha = best_alpha - 1
            while new_alpha > 0 and not has_independent_set_size(current_triples, new_alpha):
                new_alpha -= 1
        
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_triples = current_triples
    
    return best_triples
