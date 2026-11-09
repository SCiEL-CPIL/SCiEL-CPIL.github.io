import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try n=11 and n=12
    for n in (11, 12):
        if time.time() > deadline:
            break
            
        try:
            all_edges = list(combinations(range(n), 5))
            num_edges = len(all_edges)
            num_colorings = 1 << n
            
            # Build the constraint matrix A more efficiently
            # A[i, j] = 1 if coloring i is monochromatic on edge j
            # For each edge j, the monochromatic colorings are those where all 5 vertices have same color
            # We can precompute masks for each edge
            
            rows = []
            cols = []
            data = []
            
            # Precompute complement masks for efficiency
            for j, e in enumerate(all_edges):
                e_set = set(e)
                comp = [v for v in range(n) if v not in e_set]
                num_comp = len(comp)
                
                # Generate all colorings where e is monochromatic 0
                # and where e is monochromatic 1
                for mask_comp in range(1 << num_comp):
                    # Coloring where e vertices are 0
                    c0 = 0
                    for k, v in enumerate(comp):
                        if (mask_comp >> k) & 1:
                            c0 |= (1 << v)
                    rows.append(c0)
                    cols.append(j)
                    data.append(1)
                    
                    # Coloring where e vertices are 1
                    c1 = c0
                    for v in e:
                        c1 |= (1 << v)
                    rows.append(c1)
                    cols.append(j)
                    data.append(1)
            
            # Create sparse matrix A
            A = csr_matrix((data, (rows, cols)), shape=(num_colorings, num_edges))
            
            # Objective: minimize sum x_j
            c = np.ones(num_edges)
            
            # Constraints: A x >= 1 for each coloring
            lb = np.ones(num_colorings)
            ub = np.full(num_colorings, np.inf)
            
            # Variable bounds: 0 <= x_j <= 1, binary
            integrality = np.ones(num_edges)
            
            try:
                res = milp(
                    c=c,
                    constraints=LinearConstraint(A, lb, ub),
                    integrality=integrality,
                    bounds=Bounds(0, 1),
                    options={"time_limit": max(0.5, deadline - time.time())}
                )
                
                if res.status == 0:
                    x = res.x
                    x_int = np.round(x).astype(int)
                    
                    edges_count = int(sum(x_int))
                    if edges_count < best_count:
                        best_count = edges_count
                        best_n = n
                        best_edges = [list(all_edges[j]) for j in range(num_edges) if x_int[j] == 1]
                        
            except Exception:
                pass

        except Exception:
            pass

    # Fallback to greedy if ILP failed or didn't run
    if best_edges is None:
        for n in (12, 11, 10):
            if time.time() > deadline:
                break
            try:
                all_edges = list(combinations(range(n), 5))
                num_colorings = 1 << n
                full_mask = (1 << num_colorings) - 1
                masks = []
                
                for e in all_edges:
                    m = 0
                    # More efficient: generate monochromatic colorings directly
                    e_set = set(e)
                    comp = [v for v in range(n) if v not in e_set]
                    num_comp = len(comp)
                    
                    for mask_comp in range(1 << num_comp):
                        c0 = 0
                        for k, v in enumerate(comp):
                            if (mask_comp >> k) & 1:
                                c0 |= (1 << v)
                        m |= (1 << c0)
                        
                        c1 = c0
                        for v in e:
                            c1 |= (1 << v)
                        m |= (1 << c1)
                    
                    masks.append(m)
                
                # Greedy set cover
                uncovered = full_mask
                chosen = []
                remaining = list(range(len(masks)))
                
                while uncovered and remaining:
                    best_idx = -1
                    best_count = 0
                    for idx in remaining:
                        count = (uncovered & masks[idx]).bit_count()
                        if count > best_count:
                            best_count = count
                            best_idx = idx
                    if best_idx == -1:
                        break
                    chosen.append(best_idx)
                    uncovered &= ~masks[best_idx]
                    remaining.remove(best_idx)
                    if time.time() > deadline:
                        break
                
                if uncovered == 0:
                    if best_edges is None or len(chosen) < len(best_edges):
                        best_edges = [list(all_edges[idx]) for idx in chosen]
                        best_n = n
            except:
                pass

    if best_edges is None:
        # Absolute fallback: Complete 5-graph on 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9

    return {'n': best_n, 'edges': best_edges}

if __name__ == '__main__':
    construct()
