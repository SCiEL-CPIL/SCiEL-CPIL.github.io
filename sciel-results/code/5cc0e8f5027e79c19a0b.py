import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Perfect squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges
    # Edge (i, j) exists if valid_nodes[i] + valid_nodes[j] is a perfect square
    edges = []
    for i in range(n):
        vi = valid_nodes[i]
        for j in range(i + 1, n):
            vj = valid_nodes[j]
            if vi + vj in squares:
                edges.append((i, j))
    
    # If no edges, return all valid nodes
    if not edges:
        return valid_nodes

    # Formulate ILP
    # Maximize sum(x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all (i, j) in edges
    #   0 <= x_i <= 1
    #   x_i integer
    
    c = np.ones(n)
    
    # Build constraint matrix A
    # A is sparse, shape (len(edges), n)
    # Each row corresponds to an edge (i, j): A[row, i] = 1, A[row, j] = 1
    rows = []
    cols = []
    data = []
    for k, (i, j) in enumerate(edges):
        rows.append(k)
        cols.append(i)
        data.append(1)
        rows.append(k)
        cols.append(j)
        data.append(1)
        
    A = csr_matrix((data, (rows, cols)), shape=(len(edges), n))
    b = np.ones(len(edges))
    
    constraints = LinearConstraint(A, ub=b)
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 80, "mip_rel_gap": 1e-9})
        
        if res.status == 0:
            x = res.x
            # Select nodes where x_i is close to 1
            selected_indices = [i for i in range(n) if x[i] > 0.5]
            result = sorted(valid_nodes[i] for i in selected_indices)
            
            # Verify result
            if result:
                result_set = set(result)
                valid = True
                for a in result:
                    if 2 * a in squares:
                        valid = False
                        break
                    for b_val in result:
                        if a <= b_val and a + b_val in squares:
                            if a != b_val or 2*a in squares: # a+b square check
                                # Actually just check a+b in squares
                                pass
                # Simpler verification
                for a in result:
                    for b_val in result:
                        if a <= b_val:
                            if a + b_val in squares:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    return result
            
            # If ILP result is invalid or empty, fallback
            pass
        else:
            # Fallback to heuristic if ILP fails
            pass
            
    except Exception:
        # Fallback to heuristic
        pass

    # Heuristic Fallback
    # Greedy + Local Search
    def is_valid_set(selected):
        sel_set = set(selected)
        # Precompute conflicts for speed? No, just check
        for a in selected:
            if 2 * a in squares:
                return False
            for b_val in selected:
                if a <= b_val and a + b_val in squares:
                    return False
        return True

    def greedy_mis(order):
        selected = set()
        # We need to check conflicts against selected
        # Using valid_nodes list
        for idx in order:
            v = valid_nodes[idx]
            conflict = False
            for u in selected:
                if v + u in squares:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
        return selected

    def local_search(selected):
        selected = set(selected)
        improved = True
        while improved and (time.time() - start_time) < deadline - 2:
            improved = False
            # Try adding
            for v in valid_nodes:
                if v not in selected:
                    conflict = False
                    for u in selected:
                        if v + u in squares:
                            conflict = True
                            break
                    if not conflict:
                        selected.add(v)
                        improved = True
                        break
            if improved:
                continue
            # Try removing one and adding multiple
            for rem in list(selected):
                new_sel = selected - {rem}
                added = 0
                for v in valid_nodes:
                    if v not in new_sel and v != rem:
                        conflict = False
                        for u in new_sel:
                            if v + u in squares:
                                conflict = True
                                break
                        if not conflict:
                            new_sel.add(v)
                            added += 1
                if len(new_sel) > len(selected):
                    selected = new_sel
                    improved = True
                    break
        return selected

    import random
    random.seed(42)
    best = set()
    best_size = 0
    
    # Natural order
    order = list(range(n))
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
        
    # Random orders
    for _ in range(10):
        if (time.time() - start_time) > deadline - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_mis(order)
        sel = local_search(sel)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
            
    return sorted(best) if best else [1]
