"""Improved program using Mixed Integer Linear Programming (MIP) with sparse matrices for Capacitated Facility Location."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Mixed Integer Linear Programming (MIP).
    Uses sparse matrices for efficiency.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    if n_customers == 0:
        return np.array([], dtype=int)

    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)

    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_customers * n_facilities] = assign_cost.flatten()
    c[n_customers * n_facilities:] = open_cost

    # Bounds: All variables are binary [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality: All variables are integer (binary)
    integrality = np.ones(n_vars)
    
    # --- Construct A_eq (Customer assignment constraints) ---
    # Shape: (n_customers, n_vars)
    # For each customer i, sum_j x[i,j] = 1
    # x[i,j] is at index i*n_facilities + j
    
    A_eq_rows = []
    A_eq_cols = []
    A_eq_vals = []
    
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq_rows.append(i)
            A_eq_cols.append(i * n_facilities + j)
            A_eq_vals.append(1.0)
            
    A_eq = sparse.csr_matrix((A_eq_vals, (A_eq_rows, A_eq_cols)), shape=(n_customers, n_vars))
    b_eq = np.ones(n_customers)
    
    # --- Construct A_ub (Capacity and Linking constraints) ---
    # Total rows: n_facilities (capacity) + n_customers * n_facilities (linking)
    n_cap_rows = n_facilities
    n_link_rows = n_customers * n_facilities
    n_ub_rows = n_cap_rows + n_link_rows
    
    A_ub_rows = []
    A_ub_cols = []
    A_ub_vals = []
    
    # 1. Capacity constraints: sum_i d_i x[i,j] - C_j y_j <= 0
    for j in range(n_facilities):
        # Coeff for x[i,j] is d_i
        for i in range(n_customers):
            A_ub_rows.append(j)
            A_ub_cols.append(i * n_facilities + j)
            A_ub_vals.append(demands[i])
        # Coeff for y_j is -C_j
        A_ub_rows.append(j)
        A_ub_cols.append(n_customers * n_facilities + j)
        A_ub_vals.append(-capacities[j])
        
    # 2. Linking constraints: x[i,j] - y_j <= 0
    row_offset = n_cap_rows
    for i in range(n_customers):
        for j in range(n_facilities):
            r = row_offset + i * n_facilities + j
            # Coeff for x[i,j] is 1
            A_ub_rows.append(r)
            A_ub_cols.append(i * n_facilities + j)
            A_ub_vals.append(1.0)
            # Coeff for y_j is -1
            A_ub_rows.append(r)
            A_ub_cols.append(n_customers * n_facilities + j)
            A_ub_vals.append(-1.0)
            
    A_ub = sparse.csr_matrix((A_ub_vals, (A_ub_rows, A_ub_cols)), shape=(n_ub_rows, n_vars))
    b_ub = np.zeros(n_ub_rows)
    
    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    ]
    
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x_flat = res.x[:n_customers * n_facilities]
            x = x_flat.reshape((n_customers, n_facilities))
            assignment = np.argmax(x, axis=1)
            return assignment.astype(int)
        else:
            return _fallback_greedy(demands, capacities, open_cost, assign_cost)
            
    except Exception:
        return _fallback_greedy(demands, capacities, open_cost, assign_cost)

def _fallback_greedy(demands, capacities, open_cost, assign_cost):
    """Improved greedy fallback: Sort by demand descending, assign to best feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending to handle large demands first
    order = np.argsort(-demands)
    
    for i in order:
        best_f = -1
        best_cost = np.inf
        
        # Find feasible facilities
        feasible = np.where(load + demands[i] <= capacities)[0]
        
        if len(feasible) == 0:
            # If no feasible facility (should not happen if problem is feasible),
            # pick the one with most remaining capacity
            best_f = np.argmax(capacities - load)
        else:
            # Among feasible, pick the one with lowest assign cost
            # Note: We don't include open cost in this greedy step as it's hard to estimate
            # but we can prioritize opening new facilities if cost is low? 
            # For now, just min assign cost.
            costs = assign_cost[i, feasible]
            min_idx = np.argmin(costs)
            best_f = feasible[min_idx]
            best_cost = costs[min_idx]
            
        assignment[i] = best_f
        load[best_f] += demands[i]
        
    return assignment
# EVOLVE-BLOCK-END
