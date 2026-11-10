import time
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square (diagonal constraint a=a => 2a not square)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    # Build conflict graph: adjacency list
    # conflict[i] = set of j != i such that valid_vertices[i] + valid_vertices[j] is a square
    conflict = [set() for _ in range(n_vars)]
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # If no conflicts, return all
    if all(len(c) == 0 for c in conflict):
        return valid_vertices
    
    # Build A_ub matrix for LP relaxation
    # Each constraint: x_i + x_j <= 1 for each edge (i,j)
    edges = []
    for i in range(n_vars):
        for j in conflict[i]:
            if i < j:
                edges.append((i, j))
    
    m_constraints = len(edges)
    if m_constraints == 0:
        return valid_vertices
    
    A_ub = np.zeros((m_constraints, n_vars), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
    b_ub = np.ones(m_constraints, dtype=np.float64)
    
    # Objective: maximize sum(x) => minimize -sum(x)
    c_obj = -np.ones(n_vars, dtype=np.float64)
    
    # LP relaxation bounds: 0 <= x_i <= 1
    bounds_lp = [(0, 1)] * n_vars
    
    def solve_lp(x_fixed):
        """
        x_fixed: dict of {var_index: value} where value is 0 or 1 (fixed variables)
        Returns (lp_value, lp_solution) or (None, None) if infeasible
        lp_value is the optimal value of -sum(x) (so actual max is -lp_value)
        """
        if time.time() > deadline - 5:
            return None, None
        
        # Remove fixed variables from LP
        free_vars = [i for i in range(n_vars) if i not in x_fixed]
        if not free_vars:
            # All fixed, check feasibility
            for i, j in edges:
                if i in x_fixed and j in x_fixed:
                    if x_fixed[i] + x_fixed[j] > 1:
                        return None, None
            val = -sum(x_fixed.values())
            return val, dict(x_fixed)
        
        # Build reduced problem
        c_red = c_obj[free_vars]
        
        # Filter constraints
        A_red_rows = []
        b_red = []
        for k, (i, j) in enumerate(edges):
            # Skip if both fixed
            if i in x_fixed and j in x_fixed:
                if x_fixed[i] + x_fixed[j] > 1:
                    return None, None
                continue
            # If one fixed, adjust RHS
            if i in x_fixed:
                rhs = 1.0 - x_fixed[i]
                if rhs < -1e-9:
                    return None, None
                # Only include j in this constraint
                if j in free_vars:
                    idx = free_vars.index(j)
                    row = np.zeros(len(free_vars), dtype=np.float64)
                    row[idx] = 1.0
                    A_red_rows.append(row)
                    b_red.append(max(0, rhs))
            elif j in x_fixed:
                rhs = 1.0 - x_fixed[j]
                if rhs < -1e-9:
                    return None, None
                # Only include i in this constraint
                if i in free_vars:
                    idx = free_vars.index(i)
                    row = np.zeros(len(free_vars), dtype=np.float64)
                    row[idx] = 1.0
                    A_red_rows.append(row)
                    b_red.append(max(0, rhs))
            else:
                # Both free
                idx_i = free_vars.index(i)
                idx_j = free_vars.index(j)
                row = np.zeros(len(free_vars), dtype=np.float64)
                row[idx_i] = 1.0
                row[idx_j] = 1.0
                A_red_rows.append(row)
                b_red.append(1.0)
        
        if not A_red_rows:
            # No constraints, all free vars can be 1
            lp_val = -len(free_vars) - sum(x_fixed.values())
            sol = dict(x_fixed)
            for v in free_vars:
                sol[v] = 1.0
            return lp_val, sol
        
        A_red = np.array(A_red_rows, dtype=np.float64)
        b_red_arr = np.array(b_red, dtype=np.float64)
        bounds_red = [(0, 1)] * len(free_vars)
        
        try:
            res = linprog(c_red, A_ub=A_red, b_ub=b_red_arr, bounds=bounds_red, method='highs')
            if res.status == 0:
                lp_val = -res.fun - sum(x_fixed.values())
                sol = dict(x_fixed)
                for idx, v in enumerate(free_vars):
                    sol[v] = res.x[idx]
                return lp_val, sol
            else:
                return None, None
        except Exception:
            return None, None
    
    # Branch and bound
    best_solution = None
    best_value = -np.inf
    
    # Greedy initial solution for lower bound
    def greedy_solution():
        S = set()
        # Sort by degree (conflict count) descending - pick high-degree vertices first
        vertices_sorted = sorted(range(n_vars), key=lambda i: len(conflict[i]), reverse=True)
        for v in vertices_sorted:
            can_add = True
            for u in S:
                if u in conflict[v]:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        return S
    
    initial_greedy = greedy_solution()
    if initial_greedy:
        best_value = len(initial_greedy)
        best_solution = initial_greedy.copy()
    
    # Branch and bound
    # State: (fixed_dict, upper_bound)
    # Use a priority queue: higher upper bound first (best-first search)
    import heapq
    
    # Initial LP relaxation
    lp_val, lp_sol = solve_lp({})
    if lp_val is None:
        # Infeasible? Shouldn't happen
        return list(best_solution) if best_solution else [valid_vertices[0]]
    
    # Upper bound is -lp_val (since lp_val is -sum(x))
    initial_ub = -lp_val
    
    # Priority queue: (-upper_bound, counter, fixed_dict)
    # We want to explore highest upper bound first
    counter = 0
    pq = []
    heapq.heappush(pq, (-initial_ub, counter, {}))
    counter += 1
    
    nodes_explored = 0
    max_nodes = 50000  # Limit to prevent excessive time
    
    while pq and nodes_explored < max_nodes:
        if time.time() > deadline - 10:
            break
        
        neg_ub, _, fixed = heapq.heappop(pq)
        ub = -neg_ub
        nodes_explored += 1
        
        # Prune if upper bound <= best known
        if ub <= best_value + 1e-9:
            continue
        
        # Solve LP for this node
        lp_val_node, lp_sol_node = solve_lp(fixed)
        if lp_val_node is None:
            continue
        
        node_ub = -lp_val_node
        if node_ub <= best_value + 1e-9:
            continue
        
        # Check if LP solution is integral
        integral = True
        for v in range(n_vars):
            if v not in fixed:
                if abs(lp_sol_node[v] - round(lp_sol_node[v])) > 1e-6:
                    integral = False
                    break
        
        if integral:
            # Feasible integer solution
            sol_set = set()
            for v in range(n_vars):
                if lp_sol_node[v] > 0.5:
                    sol_set.add(v)
            if len(sol_set) > best_value:
                best_value = len(sol_set)
                best_solution = sol_set.copy()
            continue
        
        # Branch: find fractional variable with highest conflict density
        # Score = degree / (fractionality) to prioritize high-impact variables
        best_var = -1
        best_score = -1
        for v in range(n_vars):
            if v not in fixed:
                frac = lp_sol_node[v]
                if 1e-6 < frac < 1 - 1e-6:
                    # Score by degree and fractionality
                    score = len(conflict[v]) / (min(frac, 1 - frac) + 1e-9)
                    if score > best_score:
                        best_score = score
                        best_var = v
        
        if best_var == -1:
            continue
        
        # Branch on best_var: try 1 first (more likely to be in optimal solution)
        for val in [1, 0]:
            if time.time() > deadline - 10:
                break
            new_fixed = dict(fixed)
            new_fixed[best_var] = val
            
            # Quick feasibility check
            feasible = True
            for u in conflict[best_var]:
                if u in new_fixed:
                    if new_fixed[u] == 1 and val == 1:
                        feasible = False
                        break
            if not feasible:
                continue
            
            # Get LP bound for child
            lp_val_child, _ = solve_lp(new_fixed)
            if lp_val_child is None:
                continue
            
            child_ub = -lp_val_child
            if child_ub <= best_value + 1e-9:
                continue
            
            heapq.heappush(pq, (-child_ub, counter, new_fixed))
            counter += 1
    
    # If we have a solution, use it
    if best_solution:
        result = [valid_vertices[i] for i in best_solution]
        if result:
            return result
    
    # Fallback to greedy
    S = set()
    for v in valid_vertices:
        can_add = True
        for u in S:
            if u + v in squares:
                can_add = False
                break
        if can_add:
            S.add(v)
    return list(S)
