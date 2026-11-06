import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start = time.time()
    deadline = start + 115

    n = 9
    vertices = list(range(n))
    all_edges = list(itertools.combinations(vertices, 5))
    num_edges = len(all_edges)
    
    # Map edge tuple to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Generate all 2^(n-1) colorings (fix vertex 0 to 0)
    colorings = []
    for mask in range(1 << (n - 1)):
        colors = [0] * n
        colors[0] = 0
        for i in range(1, n):
            colors[i] = (mask >> (i - 1)) & 1
        colorings.append(colors)
        
    num_colorings = len(colorings)
    
    # Build the constraint matrix A
    # A[c, e] = 1 if edge e is monochromatic under coloring c
    A = np.zeros((num_colorings, num_edges), dtype=np.int8)
    
    for c_idx, colors in enumerate(colorings):
        # Find sets of vertices with color 0 and color 1
        V0 = [v for v in vertices if colors[v] == 0]
        V1 = [v for v in vertices if colors[v] == 1]
        
        # Edges in V0
        if len(V0) >= 5:
            for e in itertools.combinations(V0, 5):
                A[c_idx, edge_to_idx[e]] = 1
                
        # Edges in V1
        if len(V1) >= 5:
            for e in itertools.combinations(V1, 5):
                A[c_idx, edge_to_idx[e]] = 1
                
    # Objective: minimize sum of x_e
    c_obj = np.ones(num_edges)
    
    # Constraints: A x >= 1
    # milp expects constraints in the form lb <= A x <= ub
    # So we set lb = 1, ub = inf
    constraints = LinearConstraint(A, lb=1, ub=np.inf)
    
    # Bounds: 0 <= x_e <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(num_edges)
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # x might be float, round it
            x = np.round(x).astype(int)
            
            selected_edges = []
            for i, xi in enumerate(x):
                if xi == 1:
                    selected_edges.append(list(all_edges[i]))
            
            return {'n': n, 'edges': selected_edges}
        else:
            # Fallback to greedy if ILP fails
            pass
            
    except Exception as e:
        pass
        
    # Fallback: Greedy removal from complete hypergraph
    # This is the logic from the original program, just in case
    def is_not_2colorable_masks(masks):
        if time.time() > deadline:
            return True
        if not masks:
            return False
        for color_mask in range(256):
            ok = True
            for em in masks:
                if (color_mask & em) == 0 or (color_mask & em) == em:
                    ok = False
                    break
            if ok:
                return False
        return True

    edge_masks = []
    for e in all_edges:
        em = 0
        for v in e:
            em |= (1 << v)
        edge_masks.append(em)
        
    mask_to_edge = {em: e for e, em in zip(all_edges, edge_masks)}
    
    best_masks = edge_masks[:]
    best_count = len(best_masks)
    
    for r in range(500):
        if time.time() > deadline - 5:
            break
        current_masks = best_masks[:]
        import random
        random.shuffle(current_masks)
        i = 0
        while i < len(current_masks):
            if time.time() > deadline - 2:
                break
            trial_masks = current_masks[:i] + current_masks[i+1:]
            if is_not_2colorable_masks(trial_masks):
                current_masks = trial_masks
            else:
                i += 1
        if len(current_masks) < best_count:
            best_count = len(current_masks)
            best_masks = current_masks[:]
            
    best_edges = [mask_to_edge[em] for em in best_masks]
    return {'n': n, 'edges': [list(e) for e in best_edges]}
