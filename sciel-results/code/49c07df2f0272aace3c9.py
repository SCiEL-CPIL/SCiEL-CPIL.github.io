import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

_N = 256
_DEADLINE = 85.0

def _construct_edges():
    """
    Precompute edges for the conflict graph.
    i and j are connected if (i*j) % (i+j) == 0.
    Returns a list of tuples (i, j) with i < j.
    """
    edges = []
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                edges.append((i, j))
    return edges

def _greedy_fallback(edges):
    """
    Fast heuristic to ensure a valid non-empty subset is always returned.
    """
    neighbors = [set() for _ in range(_N + 1)]
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    
    S = set()
    # Greedy: pick nodes with fewest conflicts first
    nodes = list(range(1, _N + 1))
    nodes.sort(key=lambda x: len(neighbors[x]))
    
    for x in nodes:
        if not (neighbors[x] & S):
            S.add(x)
            
    return list(S)

def construct():
    t0 = time.time()
    
    # 1. Precompute graph structure
    edges = _construct_edges()
    
    # If no edges, all nodes are independent
    if not edges:
        return list(range(1, _N + 1))
        
    # 2. Prepare ILP formulation
    # Maximize sum(x_i) subject to x_i + x_j <= 1 for all (i,j) in E
    # scipy.optimize.milp minimizes, so we minimize -sum(x_i)
    
    # Number of variables
    n_vars = _N
    
    # Objective coefficients
    c = np.ones(n_vars)
    
    # Constraint matrix A
    # We have len(edges) constraints: x_i + x_j <= 1
    # Indices for rows (constraints) and columns (variables)
    rows = []
    cols = []
    data = []
    
    for k, (i, j) in enumerate(edges):
        # Variable indices are 0-based: i-1, j-1
        rows.append(k)
        cols.append(i - 1)
        data.append(1)
        
        rows.append(k)
        cols.append(j - 1)
        data.append(1)
        
    # Create sparse matrix
    A = csr_matrix((data, (rows, cols)), shape=(len(edges), n_vars))
    
    # RHS
    b = np.ones(len(edges))
    
    # Constraints: A @ x <= b
    # LinearConstraint expects (A, lb, ub)
    # Here, 0 <= x <= 1 is handled by integrality and bounds?
    # Actually, milp handles integrality separately.
    # The linear constraint is A @ x <= b.
    # lb is -inf, ub is b.
    constraint = LinearConstraint(A, ub=b)
    
    # Bounds for variables: 0 <= x_i <= 1
    # Since we want binary, we use integrality=1.
    # Bounds can be (0, 1) to help the solver.
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: 1 means integer
    integrality = np.ones(n_vars)
    
    # 3. Solve
    # Calculate remaining time
    elapsed = time.time() - t0
    time_left = _DEADLINE - elapsed - 2.0 # 2s safety margin
    
    if time_left <= 1.0:
        # Not enough time for ILP, use fallback
        return _greedy_fallback(edges)
    
    try:
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_left, "presolve": True}
        )
    except Exception as e:
        # If solver fails, use fallback
        return _greedy_fallback(edges)
    
    # 4. Extract solution
    if res.x is not None:
        # res.x is a numpy array of length _N
        # x[i] is 1 if node i+1 is in the set
        solution_indices = np.where(res.x > 0.5)[0]
        solution = [int(i) + 1 for i in solution_indices]
        
        # Validate solution
        if len(solution) > 0:
            S_set = set(solution)
            valid = True
            for i, j in edges:
                if i in S_set and j in S_set:
                    valid = False
                    break
            
            if valid:
                return solution
        
    # Fallback if ILP returned empty or invalid
    return _greedy_fallback(edges)
