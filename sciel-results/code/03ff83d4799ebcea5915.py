import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[frozenset((u, v))] = i

# Precompute edge list sorted by index (it is already)
# We will use bitmasks for adjacency

def find_hamilton_cycle_ilp(K, deadline):
    """
    Finds a Hamilton cycle using edges with index <= K using ILP.
    """
    allowed = np.zeros((N, N), dtype=int)
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        allowed[u][v] = 1
        allowed[v][u] = 1
        
    for v in range(N):
        if allowed[v].sum() < 2:
            return None

    n_x = N * N
    n_u = N - 1
    total_vars = n_x + n_u
    
    c = np.zeros(total_vars)
    A = []
    b_lb = []
    b_ub = []
    
    for i in range(N):
        row = np.zeros(total_vars)
        for j in range(N):
            if i != j:
                row[i * N + j] = 1
        A.append(row)
        b_lb.append(1)
        b_ub.append(1)
        
    for i in range(N):
        row = np.zeros(total_vars)
        for j in range(N):
            if i != j:
                row[j * N + i] = 1
        A.append(row)
        b_lb.append(1)
        b_ub.append(1)
        
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                continue
            row = np.zeros(total_vars)
            row[n_x + i - 1] = 1
            row[n_x + j - 1] = -1
            row[i * N + j] = N
            A.append(row)
            b_lb.append(-np.inf)
            b_ub.append(N - 1)
            
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    for i in range(N):
        for j in range(N):
            if i != j and allowed[i][j] == 0:
                ub[i * N + j] = 0
            elif i == j:
                ub[i * N + j] = 0
        
    for i in range(1, N):
        idx = n_x + i - 1
        lb[idx] = 1
        ub[idx] = N - 1
        
    integrality = np.zeros(total_vars)
    integrality[:n_x] = 1
    
    constraints = LinearConstraint(np.array(A), b_lb, b_ub)
    bounds = Bounds(lb, ub)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        return None
        
    x = res.x[:n_x].reshape(N, N)
    cycle = [0]
    current = 0
    for _ in range(N - 1):
        next_node = -1
        for j in range(N):
            if j != current and x[current][j] > 0.5:
                next_node = j
                break
        if next_node == -1:
            return None
        cycle.append(next_node)
        current = next_node
        
    if x[current][0] < 0.5:
        return None
        
    return cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    best_K = None
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        cycle = find_hamilton_cycle_ilp(mid, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        cycle = find_hamilton_cycle_ilp(len(EDGE_ORDER), deadline)
        if cycle:
            return cycle
        return list(range(N))
        
    return best_cycle
