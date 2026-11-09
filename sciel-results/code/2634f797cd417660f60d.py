import time
import math
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
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
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict adjacency matrix
    conflict_matrix = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflict_matrix[i, j] = 1
                conflict_matrix[j, i] = 1
    
    # Greedy initial solution
    def greedy_mis():
        available = [True] * n
        chosen = []
        while True:
            # Pick vertex with lowest degree among available
            best_v = -1
            best_deg = float('inf')
            for v in range(n):
                if available[v]:
                    deg = sum(conflict_matrix[v, u] for u in range(n) if available[u])
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            chosen.append(best_v)
            available[best_v] = False
            for u in range(n):
                if conflict_matrix[best_v, u]:
                    available[u] = False
        return chosen
    
    greedy_result = greedy_mis()
    greedy_size = len(greedy_result)
    
    # ILP formulation using SciPy
    try:
        # Variables: x_i = 1 if vertex i is in the independent set
        # Objective: maximize sum(x_i)
        # Constraints: x_i + x_j <= 1 for each conflict edge (i,j)
        
        # Objective coefficients
        c = -np.ones(n)  # Minimize -sum(x_i) = maximize sum(x_i)
        
        # Build constraint matrix
        constraint_rows = []
        constraint_rhs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if conflict_matrix[i, j]:
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    constraint_rows.append(row)
                    constraint_rhs.append(1)
        
        if constraint_rows:
            A = np.array(constraint_rows)
            b = np.array(constraint_rhs)
            
            # Set up MILP
            constraints = LinearConstraint(A, None, b)
            integrality = np.ones(n)  # All variables are integer
            bounds = Bounds(0, 1)  # x_i in {0, 1}
            
            # Add initial solution as hint (not directly supported, but we can use it for validation)
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )
            
            if result.success and result.x is not None:
                ilp_solution = np.round(result.x).astype(int)
                ilp_size = int(ilp_solution.sum())
                
                # Use the better of greedy and ILP
                if ilp_size >= greedy_size:
                    best_indices = list(np.where(ilp_solution == 1)[0])
                else:
                    best_indices = greedy_result
            else:
                best_indices = greedy_result
        else:
            best_indices = list(range(n))
        
    except Exception as e:
        # Fallback to greedy if ILP fails
        best_indices = greedy_result
    
    # Final local search polish
    if time.time() < deadline - 3:
        current_set = set(best_indices)
        
        # Try adding vertices
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                if i not in current_set:
                    # Check if i conflicts with any vertex in current_set
                    conflict = any(conflict_matrix[i, j] for j in current_set)
                    if not conflict:
                        current_set.add(i)
                        improved = True
        
        # Try 1-swap moves
        if time.time() < deadline - 1:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                current_list = list(current_set)
                for rem in current_list:
                    if time.time() > deadline - 1:
                        break
                    new_set = current_set - {rem}
                    # Try to add any vertex not in new_set that doesn't conflict
                    for add in range(n):
                        if add not in new_set:
                            conflict = any(conflict_matrix[add, j] for j in new_set)
                            if not conflict:
                                new_set.add(add)
                                if len(new_set) > len(current_set):
                                    current_set = new_set
                                    improved = True
                                break
                    if improved:
                        break
    
    # Extract the result
    result_nodes = [valid_nodes[i] for i in sorted(current_set)]
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
    
    return result_nodes
