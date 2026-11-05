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

def _build_neighbors(edges):
    neighbors = [set() for _ in range(_N + 1)]
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors

def _greedy_fallback(edges, neighbors=None):
    """
    Fast heuristic to ensure a valid non-empty subset is always returned.
    """
    if neighbors is None:
        neighbors = _build_neighbors(edges)
    
    S = set()
    nodes = list(range(1, _N + 1))
    nodes.sort(key=lambda x: len(neighbors[x]))
    
    for x in nodes:
        if not (neighbors[x] & S):
            S.add(x)
            
    return list(S)

def _local_search_improve(S, neighbors, time_limit=5.0):
    """
    Try to improve the independent set using local search.
    """
    t0 = time.time()
    S_set = set(S)
    best_size = len(S_set)
    
    # Try adding nodes
    improved = True
    while improved and time.time() - t0 < time_limit:
        improved = False
        candidates = [x for x in range(1, _N + 1) if x not in S_set]
        candidates.sort(key=lambda x: len(neighbors[x] & S_set))
        
        for x in candidates:
            if not (neighbors[x] & S_set):
                S_set.add(x)
                improved = True
                best_size = len(S_set)
    
    # Try swaps: remove one, add two
    if time.time() - t0 < time_limit:
        for _ in range(100):
            if time.time() - t0 > time_limit:
                break
            if not S_set:
                break
            x = random.choice(list(S_set))
            S_set.remove(x)
            # Try to add back
            added = 0
            for y in range(1, _N + 1):
                if y not in S_set and not (neighbors[y] & S_set):
                    S_set.add(y)
                    added += 1
                    if added >= 2:
                        break
            if len(S_set) > best_size:
                best_size = len(S_set)
    
    return list(S_set)

def construct():
    t0 = time.time()
    
    # 1. Precompute graph structure
    edges = _construct_edges()
    neighbors = _build_neighbors(edges)
    
    # If no edges, all nodes are independent
    if not edges:
        return list(range(1, _N + 1))
        
    # 2. Prepare ILP formulation
    n_vars = _N
    
    # Objective coefficients
    c = np.ones(n_vars)
    
    # Constraint matrix A
    rows = []
    cols = []
    data = []
    
    for k, (i, j) in enumerate(edges):
        rows.append(k)
        cols.append(i - 1)
        data.append(1)
        
        rows.append(k)
        cols.append(j - 1)
        data.append(1)
        
    A = csr_matrix((data, (rows, cols)), shape=(len(edges), n_vars))
    b = np.ones(len(edges))
    constraint = LinearConstraint(A, ub=b)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)
    
    # 3. Solve with time management
    elapsed = time.time() - t0
    time_left = _DEADLINE - elapsed - 10.0  # 10s safety margin for post-processing
    
    best_solution = _greedy_fallback(edges, neighbors)
    
    if time_left > 5.0:
        try:
            res = milp(
                c=c,
                constraints=constraint,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_left, "presolve": True}
            )
            
            if res.x is not None:
                solution_indices = np.where(res.x > 0.5)[0]
                solution = [int(i) + 1 for i in solution_indices]
                
                if len(solution) > 0:
                    S_set = set(solution)
                    valid = True
                    for i, j in edges:
                        if i in S_set and j in S_set:
                            valid = False
                            break
                    
                    if valid and len(solution) > len(best_solution):
                        best_solution = solution
        except Exception:
            pass
    
    # 4. Local search improvement
    elapsed = time.time() - t0
    time_left = _DEADLINE - elapsed - 2.0
    if time_left > 1.0:
        improved = _local_search_improve(best_solution, neighbors, time_limit=time_left)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    return best_solution
