import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 24
TIME_LIMIT = 85.0

# Precompute edge indices
EDGE_ORDER = [
    [3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23],
    [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19],
    [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19],
    [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8],
    [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17],
    [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23],
    [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4],
    [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12],
    [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20],
    [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16],
    [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18],
    [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21],
    [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14],
    [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6],
    [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8],
    [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18],
    [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12],
    [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23],
    [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20],
    [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5],
    [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11],
    [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7],
    [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9],
    [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4],
    [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14],
    [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23],
    [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16],
    [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]
]

EDGE_INDEX = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(_u, _v)] = _i
    EDGE_INDEX[(_v, _u)] = _i

def get_edge_index(u, v):
    return EDGE_INDEX[(u, v)]

def solve_ilp():
    """
    Solves the Hamiltonian Cycle problem using ILP to minimize the max edge index.
    Returns the list of vertices in the cycle or None.
    """
    n = N
    # Variables:
    # x[i, j] for i != j, indexed as i * n + j
    # u[i] for i in 0..n-1, indexed as n * n + i
    # M, indexed as n * n + n
    
    num_x = n * (n - 1)
    num_u = n
    num_M = 1
    total_vars = num_x + num_u + num_M
    
    # Objective: Minimize M
    c = np.zeros(total_vars)
    c[-1] = 1.0  # M is the last variable
    
    # Constraints
    A_rows = []
    lb = []
    ub = []
    
    # 1. Flow conservation (Out-degree 1)
    for i in range(n):
        row = np.zeros(total_vars)
        for j in range(n):
            if i != j:
                row[i * n + j] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
        
    # 2. Flow conservation (In-degree 1)
    for i in range(n):
        row = np.zeros(total_vars)
        for j in range(n):
            if i != j:
                row[j * n + i] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
        
    # 3. MTZ Subtour Elimination
    # u_i - u_j + n * x_ij <= n - 1
    for i in range(n):
        for j in range(n):
            if i != j:
                row = np.zeros(total_vars)
                row[i * n + j] = n  # coefficient for x_ij
                row[num_x + i] = 1.0  # coefficient for u_i
                row[num_x + j] = -1.0 # coefficient for u_j
                A_rows.append(row)
                lb.append(-np.inf)
                ub.append(n - 1)
                
    # 4. Link M to edge indices
    # M >= index(i,j) * x_ij  =>  index(i,j) * x_ij - M <= 0
    # Since x_ij is directed, we need to consider both directions for an undirected edge.
    # If edge (i,j) is used, either x_ij=1 or x_ji=1.
    # So M >= index(i,j) * x_ij AND M >= index(i,j) * x_ji.
    for i in range(n):
        for j in range(i + 1, n):
            idx = get_edge_index(i, j)
            # Constraint 1: idx * x_ij - M <= 0
            row1 = np.zeros(total_vars)
            row1[i * n + j] = idx
            row1[-1] = -1.0
            A_rows.append(row1)
            lb.append(-np.inf)
            ub.append(0.0)
            
            # Constraint 2: idx * x_ji - M <= 0
            row2 = np.zeros(total_vars)
            row2[j * n + i] = idx
            row2[-1] = -1.0
            A_rows.append(row2)
            lb.append(-np.inf)
            ub.append(0.0)
            
    A = np.array(A_rows)
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Integrality
    integrality = np.ones(total_vars)
    integrality[-1] = 0  # M can be continuous, but since indices are integers, it will be integer anyway
    # Actually, u_i should be integers too.
    
    # Bounds
    # x_ij in {0, 1}
    # u_i in [1, n]
    # M in [0, inf]
    lb_vars = np.zeros(total_vars)
    ub_vars = np.ones(total_vars)
    # u_i bounds
    lb_vars[num_x : num_x + num_u] = 1.0
    ub_vars[num_x : num_x + num_u] = n
    # M bounds
    lb_vars[-1] = 0.0
    ub_vars[-1] = np.inf
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=Bounds(lb_vars, ub_vars),
            options={"time_limit": 60}
        )
        
        if res.status == 0:  # Optimal
            # Extract x variables
            x_vals = res.x[:num_x]
            # Reconstruct cycle
            # Start from 0
            current = 0
            cycle = [0]
            for _ in range(n - 1):
                # Find next node
                for j in range(n):
                    if current != j and x_vals[current * n + j] > 0.5:
                        cycle.append(j)
                        current = j
                        break
                else:
                    return None # Should not happen if valid
            
            # Check if it closes
            if x_vals[current * n + 0] > 0.5:
                return cycle
            else:
                return None
                
        elif res.status == 1: # Feasible but not optimal (time limit)
             # Use the best feasible solution found
             if res.x is not None:
                 x_vals = res.x[:num_x]
                 current = 0
                 cycle = [0]
                 for _ in range(n - 1):
                     for j in range(n):
                         if current != j and x_vals[current * n + j] > 0.5:
                             cycle.append(j)
                             current = j
                             break
                     else:
                         return None
                 if x_vals[current * n + 0] > 0.5:
                     return cycle
             return None
        else:
            return None
            
    except Exception as e:
        print(f"ILP Error: {e}")
        return None

def heuristic_fallback():
    """
    Fallback heuristic if ILP fails.
    """
    # Simple random restart with local search
    best_cycle = None
    best_score = float('inf')
    
    for _ in range(100):
        cycle = list(range(N))
        random.shuffle(cycle)
        # Local search: 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(i+1, N):
                    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                    score = max(EDGE_INDEX[(new_cycle[k], new_cycle[(k+1)%N])] for k in range(N))
                    if score < best_score:
                        best_score = score
                        best_cycle = new_cycle
                        cycle = new_cycle
                        improved = True
                        break
                if improved:
                    break
                    
    return best_cycle if best_cycle else list(range(N))

def construct():
    start_time = time.time()
    
    # Try ILP first
    cycle = solve_ilp()
    
    if cycle is None:
        # Fallback
        if time.time() - start_time < TIME_LIMIT - 5:
            cycle = heuristic_fallback()
        else:
            cycle = list(range(N))
            
    return cycle
