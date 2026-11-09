import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import math
import time

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    deadline = 60.0  # 60s limit, keep margin
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # 1. Identify vertices that cannot be in the set due to self-conflict (a+a is square)
    # Condition: 2*a is NOT a perfect square
    valid_vertices = [v for v in vertices if not is_square(2 * v)]
    
    # If no valid vertices, return empty? Task says "nonempty subset".
    # But if all 2*v are squares, we can't pick any.
    # 2*v is square => v = k^2/2. Since v is integer, k must be even. k=2m => v=2m^2.
    # Squares of form 2*m^2: 2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242.
    # There are plenty of other numbers. So valid_vertices is not empty.
    
    # Map vertex value to index in the ILP
    # We only include valid_vertices in the ILP to reduce size
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    n_vars = len(valid_vertices)
    
    if n_vars == 0:
        return [] # Should not happen
    
    # 2. Build edges (conflicts)
    # Edge (u, v) exists if u+v is a perfect square and u != v
    # We only care about pairs where both are in valid_vertices
    edges = []
    # To make it faster, iterate through valid vertices
    for i in range(n_vars):
        u = valid_vertices[i]
        for j in range(i + 1, n_vars):
            v = valid_vertices[j]
            if is_square(u + v):
                edges.append((i, j))
    
    # 3. Formulate ILP
    # Maximize sum(x)
    # Subject to x_u + x_v <= 1 for all edges
    # x in {0, 1}
    
    if not edges:
        # No conflicts, take all valid vertices
        return valid_vertices
    
    # Objective
    c = np.ones(n_vars)
    
    # Constraints
    # A_ub * x <= b_ub
    # Each constraint is x_u + x_v <= 1
    m_constraints = len(edges)
    A_ub = np.zeros((m_constraints, n_vars))
    b_ub = np.ones(m_constraints)
    
    for k, (u_idx, v_idx) in enumerate(edges):
        A_ub[k, u_idx] = 1.0
        A_ub[k, v_idx] = 1.0
    
    constraints = LinearConstraint(A_ub, ub=b_ub)
    integrality = np.ones(n_vars)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        # Check time
        if time.time() - start_time > deadline - 5:
            # Fallback to greedy if taking too long (unlikely for this size)
            pass 
        else:
            if res.status == 0:
                # Optimal solution found
                x = res.x
                selected_indices = [i for i in range(n_vars) if x[i] > 0.5]
                selected_vertices = [valid_vertices[i] for i in selected_indices]
                
                # Verify solution
                is_valid = True
                for a in selected_vertices:
                    if is_square(2 * a):
                        is_valid = False
                        break
                for i in range(len(selected_vertices)):
                    a = selected_vertices[i]
                    for j in range(i, len(selected_vertices)):
                        b = selected_vertices[j]
                        if is_square(a + b):
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    return selected_vertices
                else:
                    # Verification failed, fall through to greedy
                    pass
            
    except Exception:
        pass
    
    # Fallback: Greedy Heuristic
    # Sort by degree in the conflict graph
    # Build adjacency list
    adj = {v: set() for v in valid_vertices}
    for u_idx, v_idx in edges:
        u = valid_vertices[u_idx]
        v = valid_vertices[v_idx]
        adj[u].add(v)
        adj[v].add(u)
    
    candidates = set(valid_vertices)
    selected = []
    
    while candidates:
        # Find vertex with minimum degree
        min_deg = float('inf')
        best_v = None
        for v in candidates:
            deg = len([n for n in adj[v] if n in candidates])
            if deg < min_deg:
                min_deg = deg
                best_v = v
        
        if best_v is None:
            break
            
        selected.append(best_v)
        # Remove best_v and its neighbors from candidates
        candidates.discard(best_v)
        for n in list(adj[best_v]):
            if n in candidates:
                candidates.discard(n)
        
        if time.time() - start_time > deadline - 2:
            break
    
    if not selected:
        # Last resort: pick a single valid vertex
        if valid_vertices:
            return [valid_vertices[0]]
        else:
            return [1] # 1+1=2 not square. 1 is safe.
            
    return selected
