import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    num_possible_edges = len(all_edges)
    
    def evaluate(values, edges):
        if not edges:
            return 0
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        u_vals = v_arr[e_arr[:, 0]]
        v_vals = v_arr[e_arr[:, 1]]
        
        sums = u_vals + v_vals
        prods = u_vals * v_vals
        
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    def random_valid_values():
        return random.sample(range(1, max_val + 1), n)

    # Strategy:
    # 1. Partition vertices into cliques (triangles are good for small distinct counts).
    # 2. For a fixed set of values, solve an ILP to pick 35 edges minimizing distinct sums/prods.
    # 3. Optimize values using local search, but re-solve the ILP for edges at each step.
    
    # ILP Formulation:
    # Variables:
    #   x_e for each edge e in all_edges (binary: is edge selected?)
    #   y_s for each possible sum s (binary: is sum s used by any selected edge?)
    #   z_p for each possible product p (binary: is product p used by any selected edge?)
    #
    # Objective: Minimize max(sum(y_s), sum(z_p))
    # This is non-linear. We can linearize by introducing a variable T and constraints:
    #   sum(y_s) <= T
    #   sum(z_p) <= T
    #   Minimize T
    #
    # Constraints:
    #   sum(x_e) == 35
    #   For each edge e with sum s_e: x_e <= y_{s_e}
    #   For each edge e with product p_e: x_e <= z_{p_e}
    
    def solve_edges_ilp(values):
        # This is a small ILP. We can use a simple branch and bound or even brute force 
        # if the number of possible sums/prods is small.
        # However, scipy.optimize.milp is available.
        
        v_arr = np.array(values, dtype=np.int64)
        
        # Precompute sum and prod for all edges
        edge_sums = []
        edge_prods = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_sums.append(s)
            edge_prods.append(p)
            
        unique_sums = list(set(edge_sums))
        unique_prods = list(set(edge_prods))
        
        num_edges = len(all_edges)
        num_sums = len(unique_sums)
        num_prods = len(unique_prods)
        
        # Map values to indices
        sum_to_idx = {s: i for i, s in enumerate(unique_sums)}
        prod_to_idx = {p: i for i, p in enumerate(unique_prods)}
        
        # Variables:
        # 0..num_edges-1: x_e
        # num_edges..num_edges+num_sums-1: y_s
        # num_edges+num_sums..num_edges+num_sums+num_prods-1: z_p
        # num_edges+num_sums+num_prods: T
        
        total_vars = num_edges + num_sums + num_prods + 1
        
        # Objective: Minimize T
        c = np.zeros(total_vars)
        c[-1] = 1.0
        
        # Constraints:
        # 1. sum(x_e) == 35
        # 2. sum(y_s) <= T
        # 3. sum(z_p) <= T
        # 4. x_e <= y_{s(e)} for all e
        # 5. x_e <= z_{p(e)} for all e
        
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        # 1. sum(x_e) == 35
        row = np.zeros(total_vars)
        for i in range(num_edges):
            row[i] = 1.0
        A_eq.append(row)
        b_eq.append(35.0)
        
        # 2. sum(y_s) - T <= 0
        row = np.zeros(total_vars)
        for i in range(num_edges, num_edges + num_sums):
            row[i] = 1.0
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # 3. sum(z_p) - T <= 0
        row = np.zeros(total_vars)
        for i in range(num_edges + num_sums, num_edges + num_sums + num_prods):
            row[i] = 1.0
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # 4 & 5. x_e - y_{s(e)} <= 0 and x_e - z_{p(e)} <= 0
        for i in range(num_edges):
            s_idx = sum_to_idx[edge_sums[i]]
            p_idx = prod_to_idx[edge_prods[i]]
            
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[num_edges + s_idx] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[num_edges + num_sums + p_idx] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Bounds
        integrality = np.zeros(total_vars)
        integrality[:total_vars-1] = 1 # x, y, z are integer (binary)
        # T is continuous (or integer, doesn't matter much for max)
        
        # Actually, T can be integer since sums of integers are integers.
        integrality[-1] = 1
        
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        
        # T can be larger than 1
        ub[-1] = float('inf')
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            
            constraints = [
                LinearConstraint(A_ub, -np.inf, b_ub),
                LinearConstraint(A_eq, b_eq, b_eq)
            ]
            bounds = Bounds(lb, ub)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                x_edges = res.x[:num_edges]
                selected_edges = []
                for i in range(num_edges):
                    if x_edges[i] > 0.5:
                        selected_edges.append(list(all_edges[i]))
                return selected_edges
            else:
                # Fallback to greedy if ILP fails
                return select_edges_greedy(values)
        except Exception:
            return select_edges_greedy(values)

    def select_edges_greedy(values):
        used_sums = set()
        used_prods = set()
        selected_edges = []
        
        edge_stats = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_stats.append((u, v, s, p))
        
        available_indices = list(range(len(edge_stats)))
        
        for _ in range(m):
            best_idx = -1
            best_cost = float('inf')
            
            for idx in available_indices:
                u, v, s, p = edge_stats[idx]
                new_s = 0 if s in used_sums else 1
                new_p = 0 if p in used_prods else 1
                cost = new_s + new_p
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
                    if cost == 0:
                        break
            
            if best_idx == -1:
                break
            
            u, v, s, p = edge_stats[best_idx]
            selected_edges.append([u, v])
            used_sums.add(s)
            used_prods.add(p)
            available_indices.remove(best_idx)
            
        return selected_edges

    def optimize_values_local(current_values, current_edges, time_limit):
        best_vals = current_values.copy()
        best_edges = current_edges
        best_score = evaluate(best_vals, best_edges)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        
        while time.time() < time_limit and T > T_min:
            new_vals = best_vals.copy()
            num_changes = random.randint(1, 2)
            for _ in range(num_changes):
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 200)
                while new_val in new_vals:
                    new_val = random.randint(1, 200)
                new_vals[idx] = new_val
            
            # Re-solve edges ILP for new values
            new_edges = solve_edges_ilp(new_vals)
            new_score = evaluate(new_vals, new_edges)
            
            delta = new_score - best_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                best_vals = new_vals
                best_edges = new_edges
                best_score = new_score
            
            T *= cooling
        
        return best_vals, best_edges, best_score

    best_global_score = float('inf')
    best_global_vals = None
    best_global_edges = None
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if r == 0:
            init_vals = list(range(1, 15))
        else:
            init_vals = random.sample(range(1, 100), n)
        
        curr_edges = solve_edges_ilp(init_vals)
        curr_vals = init_vals.copy()
        curr_score = evaluate(curr_vals, curr_edges)
        
        if curr_score < best_global_score:
            best_global_score = curr_score
            best_global_vals = curr_vals.copy()
            best_global_edges = [list(e) for e in curr_edges]
            
        t_start = time.time()
        t_limit = min(deadline - 5.0, t_start + 10.0)
        curr_vals, curr_edges, curr_score = optimize_values_local(curr_vals, curr_edges, t_limit)
        
        if curr_score < best_global_score:
            best_global_score = curr_score
            best_global_vals = curr_vals.copy()
            best_global_edges = [list(e) for e in curr_edges]

    if best_global_vals is None:
        best_global_vals = list(range(1, 15))
        best_global_edges = select_edges_greedy(best_global_vals)
    
    res = {
        'values': [int(v) for v in best_global_vals],
        'edges': [[int(u), int(v)] for u, v in best_global_edges]
    }
    
    return res
