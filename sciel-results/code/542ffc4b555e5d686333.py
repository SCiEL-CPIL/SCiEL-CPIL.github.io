import time
import math
import random
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
        
    # 2. Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    if not valid_vertices:
        return [1]

    # 3. Build conflict graph (adjacency list)
    # We only need conflicts between distinct valid vertices.
    # If u+v is a square, they cannot both be in the set.
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    M = len(valid_vertices)
    
    # List of constraints (i, j) where i < j and valid_vertices[i] + valid_vertices[j] is square
    constraints = []
    conflict_sets = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                constraints.append((i, j))
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
                
    if len(constraints) == 0:
        return list(valid_vertices)

    # 4. Try MILP first
    try:
        c = -np.ones(M)
        A = np.zeros((len(constraints), M))
        b = np.ones(len(constraints))
        
        for k, (i, j) in enumerate(constraints):
            A[k, i] = 1
            A[k, j] = 1
            
        lincons = LinearConstraint(A, ub=b)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(c=c, constraints=lincons, integrality=np.ones(M), bounds=bounds)
        
        if res.success and res.x is not None:
            selected_indices = [i for i, val in enumerate(res.x) if val > 0.5]
            if selected_indices:
                return [valid_vertices[i] for i in selected_indices]
    except Exception:
        pass

    # 5. Fallback: Branch and Bound with LP relaxation
    # Precompute conflict sets as numpy arrays for faster operations
    conflict_arrays = [np.array(list(cs)) if cs else np.array([], dtype=int) for cs in conflict_sets]
    
    def solve_lp(remaining, fixed_true, fixed_false):
        """Solve LP relaxation for the subproblem."""
        # remaining: set of indices not yet decided
        # fixed_true: set of indices forced to 1
        # fixed_false: set of indices forced to 0
        
        # Variables are for indices in 'remaining'
        var_indices = sorted(remaining)
        var_map = {idx: i for i, idx in enumerate(var_indices)}
        n_vars = len(var_indices)
        
        if n_vars == 0:
            return len(fixed_true), {}
        
        # Objective: maximize sum of x
        c = -np.ones(n_vars)
        
        # Constraints:
        # 1. For each conflict (i, j) where both i, j are in remaining:
        #    x_i + x_j <= 1
        # 2. For each conflict (i, j) where i in remaining, j in fixed_true:
        #    x_i <= 0 (i.e., x_i must be 0)
        # 3. For each conflict (i, j) where i in fixed_true, j in remaining:
        #    x_j <= 0
        
        A_list = []
        b_list = []
        
        # Add constraints from conflicts among remaining
        for i in var_indices:
            for j in conflict_arrays[i]:
                if j > i and j in var_map:
                    # Both in remaining, add x_i + x_j <= 1
                    row = np.zeros(n_vars)
                    row[var_map[i]] = 1
                    row[var_map[j]] = 1
                    A_list.append(row)
                    b_list.append(1)
        
        # Add constraints from conflicts with fixed_true
        for i in var_indices:
            for j in conflict_arrays[i]:
                if j in fixed_true:
                    # x_i <= 0
                    row = np.zeros(n_vars)
                    row[var_map[i]] = 1
                    A_list.append(row)
                    b_list.append(0)
        
        if not A_list:
            # No constraints, all can be 1
            return len(fixed_true) + n_vars, {idx: 1.0 for idx in var_indices}
        
        A = np.array(A_list)
        b = np.array(b_list)
        
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n_vars, method='highs')
        
        if res.success:
            # LP objective is -sum(x), so sum(x) = -res.fun
            lp_sum = -res.fun
            total = len(fixed_true) + lp_sum
            solution = {idx: res.x[var_map[idx]] for idx in var_indices}
            return total, solution
        else:
            return -np.inf, {}
    
    def branch_and_bound():
        """Branch and bound for maximum independent set."""
        best_solution = set()
        best_value = -np.inf
        
        # Initial LP relaxation
        remaining = set(range(M))
        fixed_true = set()
        fixed_false = set()
        
        lp_value, lp_solution = solve_lp(remaining, fixed_true, fixed_false)
        
        if lp_value < -np.inf + 1:
            return best_solution
        
        # Branch and bound
        # Use a stack for DFS
        stack = [(remaining, fixed_true, fixed_false, lp_value, lp_solution)]
        
        while stack:
            if time.time() > deadline - 5.0:
                break
                
            rem, ft, ff, bound, sol = stack.pop()
            
            if bound <= best_value + 1e-9:
                continue
            
            # Check if we can add any more
            if not rem:
                if len(ft) > len(best_solution):
                    best_solution = set(ft)
                    best_value = len(ft)
                continue
            
            # Try to find a variable with fractional value
            frac_var = None
            for idx, val in sol.items():
                if idx in rem and 0.01 < val < 0.99:
                    frac_var = idx
                    break
            
            if frac_var is None:
                # All variables are integral in LP solution
                # Construct a solution
                candidate = set(ft)
                for idx, val in sol.items():
                    if idx in rem and val > 0.5:
                        candidate.add(idx)
                if len(candidate) > len(best_solution):
                    best_solution = set(candidate)
                    best_value = len(candidate)
                continue
            
            # Branch on frac_var
            # Branch 1: set frac_var = 1
            new_ft_1 = ft | {frac_var}
            new_ff_1 = ff | conflict_arrays[frac_var].set() if hasattr(conflict_arrays[frac_var], 'set') else ff | set(conflict_arrays[frac_var])
            new_rem_1 = rem - {frac_var} - set(conflict_arrays[frac_var])
            
            # Remove any variables that conflict with newly fixed true
            to_remove = set()
            for v in new_ft_1:
                for u in conflict_arrays[v]:
                    if u in new_rem_1:
                        to_remove.add(u)
            new_rem_1 = new_rem_1 - to_remove
            new_ff_1 = new_ff_1 | to_remove
            
            lp_val_1, sol_1 = solve_lp(new_rem_1, new_ft_1, new_ff_1)
            if lp_val_1 > best_value:
                stack.append((new_rem_1, new_ft_1, new_ff_1, lp_val_1, sol_1))
            
            # Branch 2: set frac_var = 0
            new_ft_2 = ft
            new_ff_2 = ff | {frac_var}
            new_rem_2 = rem - {frac_var}
            
            lp_val_2, sol_2 = solve_lp(new_rem_2, new_ft_2, new_ff_2)
            if lp_val_2 > best_value:
                stack.append((new_rem_2, new_ft_2, new_ff_2, lp_val_2, sol_2))
        
        return best_solution
    
    # Run branch and bound
    best_indices = branch_and_bound()
    
    if best_indices:
        return [valid_vertices[i] for i in sorted(best_indices)]
    
    # 6. Final fallback: Simple Greedy with Local Search
    conflict = {v: set() for v in valid_vertices}
    for i in range(len(valid_vertices)):
        vi = valid_vertices[i]
        for j in range(i + 1, len(valid_vertices)):
            vj = valid_vertices[j]
            if vi + vj in squares:
                conflict[vi].add(vj)
                conflict[vj].add(vi)
                
    def greedy_init():
        sorted_v = sorted(valid_vertices, key=lambda x: len(conflict[x]))
        S = set()
        for v in sorted_v:
            can_add = True
            for u in S:
                if u in conflict[v]:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        return S

    best_S = greedy_init()
    
    def is_compatible(v, S):
        for u in S:
            if u in conflict[v]:
                return False
        return True

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for v in valid_vertices:
                if v not in S:
                    if is_compatible(v, S):
                        S.add(v)
                        improved = True
            if not improved:
                best_move = None
                best_gain = 0
                for u in list(S):
                    S_temp = S - {u}
                    gain = 0
                    changed = True
                    while changed:
                        changed = False
                        for v in valid_vertices:
                            if v not in S_temp:
                                if is_compatible(v, S_temp):
                                    S_temp.add(v)
                                    changed = True
                                    gain += 1
                    if gain - 1 > best_gain:
                        best_gain = gain - 1
                        best_move = S_temp
                if best_move is not None and best_gain > 0:
                    S = best_move
                    improved = True
        return S

    best_S = local_search(best_S)
    
    # Quick randomized local search if time permits
    while time.time() < deadline - 2.0:
        if len(best_S) > 3:
            remove_count = random.randint(3, 5)
            removals = random.sample(list(best_S), min(remove_count, len(best_S)))
            perturbed_S = set(best_S)
            for r in removals:
                perturbed_S.remove(r)
            improved_S = local_search(perturbed_S)
            if len(improved_S) > len(best_S):
                best_S = improved_S

    return list(best_S)
