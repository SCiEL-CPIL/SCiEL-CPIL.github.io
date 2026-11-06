import numpy as np
from math import lcm
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Nodes are 2 to 256
    nodes = list(range(2, 257))
    n = len(nodes)
    
    # Map node value to index
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Precompute lcm matrix for fast lookup
    lcm_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            val = lcm(nodes[i], nodes[j])
            lcm_matrix[i, j] = val
            lcm_matrix[j, i] = val
    
    # Find non-edges (pairs where lcm <= 256)
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if lcm_matrix[i, j] <= 256:
                non_edges.append((i, j))
                
    # Objective: Maximize sum(1/x). 
    # scipy milp minimizes c @ x. So c = -1/x.
    c = np.zeros(n)
    for i, v in enumerate(nodes):
        c[i] = -1.0 / v
        
    # Constraints: 
    # 1. x_i + x_j <= 1 for all non-edges
    # 2. Clique constraints: sum(x_i for i in D_N) <= 1 for all N in [2, 256]
    #    where D_N = {d in [2, 256] : d divides N}
    
    A_list = []
    b_list = []
    
    # Add non-edge constraints
    for i, j in non_edges:
        row = np.zeros(n)
        row[i] = 1
        row[j] = 1
        A_list.append(row)
        b_list.append(1)
        
    # Add clique constraints for divisor sets
    for N in range(2, 257):
        row = np.zeros(n)
        # Find divisors of N in nodes
        for d in range(2, N + 1):
            if N % d == 0:
                idx = node_to_idx[d]
                row[idx] = 1
        A_list.append(row)
        b_list.append(1)
        
    A = np.array(A_list)
    b = np.array(b_list)
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    
    constraint = LinearConstraint(A, lb=-np.inf, ub=b)
    
    # Check time before running MILP
    if time.time() > deadline - 5:
        # Not enough time for MILP, use greedy
        selected = greedy_select(nodes, lcm_matrix)
        return selected
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback: Greedy
        selected = greedy_select(nodes, lcm_matrix)
        return selected
        
    x = res.x
    selected = [nodes[i] for i in range(n) if x[i] > 0.5]
    
    if not selected:
        return [2]
    
    # Local search polish: try to add elements or swap to improve
    selected_set = set(selected)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try adding elements
        for v in nodes:
            if v not in selected_set:
                if all(lcm_matrix[node_to_idx[v], node_to_idx[y]] > 256 for y in selected_set):
                    selected_set.add(v)
                    improved = True
                    break
        # Try swapping: remove one, add one or two
        if not improved:
            for v_remove in list(selected_set):
                for v_add in nodes:
                    if v_add in selected_set:
                        continue
                    # Check if v_add is compatible with all except v_remove
                    if all(lcm_matrix[node_to_idx[v_add], node_to_idx[y]] > 256 for y in selected_set if y != v_remove):
                        # Check if the swap improves the sum
                        old_sum = sum(1.0 / y for y in selected_set)
                        new_sum = old_sum - 1.0 / v_remove + 1.0 / v_add
                        if new_sum > old_sum + 1e-12:
                            selected_set.remove(v_remove)
                            selected_set.add(v_add)
                            improved = True
                            break
                if improved:
                    break
    
    return list(selected_set)

def greedy_select(nodes, lcm_matrix):
    """Greedy selection by weight (1/v) descending, i.e., v ascending."""
    selected = []
    selected_set = set()
    # Sort by value ascending (higher weight first)
    for x in sorted(nodes):
        if all(lcm_matrix[nodes.index(x), nodes.index(y)] > 256 for y in selected):
            selected.append(x)
            selected_set.add(x)
    return selected
