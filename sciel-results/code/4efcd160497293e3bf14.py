"""Synthetic planted 3-SAT solver with spectral component decomposition and optimized WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 285.0  # 45s safety margin before 330s kill
    
    # Convert clauses to 0-based (var, sign) representation
    # sign: +1 means variable should be True, -1 means False
    clause_vars = []
    for c in clauses:
        cv = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((var_idx, sign))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, _ in cv:
            var_to_clauses[var_idx].append(ci)
    
    # Build clause incidence matrix for spectral clustering
    # A[i,j] = 1 if variables i and j appear together in some clause
    A = np.zeros((n, n), dtype=np.float64)
    for cv in clause_vars:
        for i in range(len(cv)):
            for j in range(i+1, len(cv)):
                A[cv[i][0], cv[j][0]] += 1
                A[cv[j][0], cv[i][0]] += 1
    
    # Find connected components using the incidence structure
    # Use BFS/DFS on the variable co-occurrence graph
    visited = [False] * n
    components = []
    
    # Build adjacency list for variable co-occurrence
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] > 0:
                adj[i].append(j)
                adj[j].append(i)
    
    for v in range(n):
        if not visited[v]:
            # BFS
            component = []
            queue = [v]
            visited[v] = True
            while queue:
                u = queue.pop(0)
                component.append(u)
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        queue.append(w)
            components.append(component)
    
    # Sort components by size (largest first for time budget allocation)
    components.sort(key=len, reverse=True)
    
    # For each component, collect the relevant clauses
    # A clause belongs to a component if all its variables are in that component
    comp_clauses = {ci: [] for ci in range(len(components))}
    unassigned_clauses = []
    
    for ci, cv in enumerate(clause_vars):
        # Check which component all variables belong to
        comp_set = set()
        for var_idx, _ in cv:
            # Find which component this variable is in
            found = False
            for ci_idx, comp in enumerate(components):
                if var_idx in comp:
                    comp_set.add(ci_idx)
                    found = True
                    break
            if not found:
                break
        if len(comp_set) == 1:
            comp_idx = list(comp_set)[0]
            comp_clauses[comp_idx].append(ci)
        else:
            # Clause spans multiple components - handle in global solve
            unassigned_clauses.append(ci)
    
    # Solve each component independently
    # Use WalkSAT with optimized gain heuristic for each component
    def walk_sat_component(vars_in_comp, clause_indices, time_budget, seed):
        """Solve a subset of clauses involving only vars_in_comp."""
        rng = random.Random(seed)
        
        # Map global var indices to local
        var_to_local = {}
        for li, gv in enumerate(vars_in_comp):
            var_to_local[gv] = li
        local_n = len(vars_in_comp)
        
        # Build local clause representation
        local_clauses = []
        for ci in clause_indices:
            cv = clause_vars[ci]
            lc = []
            for var_idx, sign in cv:
                if var_idx in var_to_local:
                    lc.append((var_to_local[var_idx], sign))
            if len(lc) > 0:
                local_clauses.append(lc)
        
        local_m = len(local_clauses)
        if local_m == 0:
            return {gv: False for gv in vars_in_comp}
        
        # Initialize random assignment
        local_assign = [rng.random() < 0.5 for _ in range(local_n)]
        
        # Precompute for fast evaluation
        # For each variable, list of (clause_idx, sign) where it appears
        var_to_local_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for var_idx, sign in lc:
                var_to_local_clauses[var_idx].append((ci, sign))
        
        # Track satisfaction status of each clause
        # clause_sat[ci] = number of literals satisfied in clause ci
        clause_sat = [0] * local_m
        for ci, lc in enumerate(local_clauses):
            for var_idx, sign in lc:
                if local_assign[var_idx] == (sign == 1):
                    clause_sat[ci] += 1
        
        unsat_count = sum(1 for s in clause_sat if s == 0)
        
        if unsat_count == 0:
            return {gv: local_assign[li] for li, gv in enumerate(vars_in_comp)}
        
        # WalkSAT with gain heuristic
        # gain[v] = (number of unsat clauses containing v where flipping helps) - 
        #           (number of sat clauses containing v where flipping hurts)
        
        comp_start = time.time()
        max_steps = 50000
        step = 0
        
        while step < max_steps:
            if time.time() - comp_start > time_budget:
                break
            
            if unsat_count == 0:
                break
            
            # Pick a random unsat clause
            unsat_clauses = [ci for ci in range(local_m) if clause_sat[ci] == 0]
            if not unsat_clauses:
                break
            ci = rng.choice(unsat_clauses)
            lc = local_clauses[ci]
            
            # Compute gain for each variable in this clause
            best_var = -1
            best_gain = -float('inf')
            
            for var_idx, sign in lc:
                # Flipping var_idx: 
                # - In this clause (currently unsat), this literal becomes satisfied
                #   if the new value matches sign
                #   new_value = not local_assign[var_idx]
                #   literal satisfied if new_value == (sign == 1)
                new_val = not local_assign[var_idx]
                this_clause_gain = 1 if new_val == (sign == 1) else 0
                
                # For other clauses containing var_idx:
                other_gain = 0
                for (oci, osign) in var_to_local_clauses[var_idx]:
                    if oci == ci:
                        continue
                    # Current literal value
                    cur_lit_sat = local_assign[var_idx] == (osign == 1)
                    new_lit_sat = new_val == (osign == 1)
                    if cur_lit_sat and not new_lit_sat:
                        # Was satisfied, now unsat - check if clause becomes unsat
                        if clause_sat[oci] == 1:
                            other_gain -= 1
                    elif not cur_lit_sat and new_lit_sat:
                        # Was unsat, now sat
                        other_gain += 1
                
                gain = this_clause_gain + other_gain
                if gain > best_gain:
                    best_gain = gain
                    best_var = var_idx
            
            if best_var < 0:
                break
            
            # Flip best_var
            old_val = local_assign[best_var]
            new_val = not old_val
            local_assign[best_var] = new_val
            
            # Update clause_sat for all clauses containing best_var
            for (oci, osign) in var_to_local_clauses[best_var]:
                cur_lit_sat_old = old_val == (osign == 1)
                cur_lit_sat_new = new_val == (osign == 1)
                if cur_lit_sat_old and not cur_lit_sat_new:
                    clause_sat[oci] -= 1
                    if clause_sat[oci] == 0:
                        unsat_count += 1
                elif not cur_lit_sat_old and cur_lit_sat_new:
                    clause_sat[oci] += 1
                    if clause_sat[oci] == 1:
                        unsat_count -= 1
            
            step += 1
        
        return {gv: local_assign[li] for li, gv in enumerate(vars_in_comp)}
    
    # Allocate time budget to components
    # Larger components get more time
    total_time_budget = deadline - time.time() - 5.0  # reserve 5s for finalization
    if total_time_budget < 1.0:
        total_time_budget = 1.0
    
    # Weight by number of clauses in each component
    comp_clause_counts = [len(comp_clauses[ci]) for ci in range(len(components))]
    total_clauses_in_comps = sum(comp_clause_counts)
    if total_clauses_in_comps == 0:
        comp_time_budgets = [total_time_budget / len(components)] * len(components)
    else:
        comp_time_budgets = []
        for ci in range(len(components)):
            if comp_clause_counts[ci] > 0:
                comp_time_budgets.append(total_time_budget * comp_clause_counts[ci] / total_clauses_in_comps)
            else:
                comp_time_budgets.append(0.1)
    
    # Solve each component
    solution = {}
    for ci, comp in enumerate(components):
        if not comp:
            continue
        if comp_time_budgets[ci] < 0.1:
            comp_time_budgets[ci] = 0.1
        comp_solution = walk_sat_component(comp, comp_clauses[ci], comp_time_budgets[ci], seed=42 + ci)
        solution.update(comp_solution)
    
    # Handle any unassigned clauses (clauses spanning multiple components)
    # For now, just make sure all variables have an assignment
    for v in range(n):
        if v not in solution:
            solution[v] = False
    
    # Final check: verify if the solution satisfies all clauses
    # If not, do a global WalkSAT pass with remaining time
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 1.0:
        # Check current satisfaction
        assignment = [solution.get(v, False) for v in range(n)]
        
        def is_satisfied(clause):
            return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
        
        unsat = [c for c in clauses if not is_satisfied(c)]
        
        if unsat:
            # Global WalkSAT to fix remaining unsat clauses
            rng = random.Random(123)
            global_start = time.time()
            max_global_steps = 100000
            
            # Build global var_to_clauses for gain computation
            global_var_to_clauses = [[] for _ in range(n)]
            for ci, cv in enumerate(clause_vars):
                for var_idx, sign in cv:
                    global_var_to_clauses[var_idx].append((ci, sign))
            
            # Track clause satisfaction
            clause_sat_global = [0] * m
            for ci, cv in enumerate(clause_vars):
                for var_idx, sign in cv:
                    if assignment[var_idx] == (sign == 1):
                        clause_sat_global[ci] += 1
            
            unsat_count_global = sum(1 for s in clause_sat_global if s == 0)
            
            step = 0
            while step < max_global_steps:
                if time.time() - global_start > remaining_time:
                    break
                if unsat_count_global == 0:
                    break
                
                unsat_clauses = [ci for ci in range(m) if clause_sat_global[ci] == 0]
                if not unsat_clauses:
                    break
                ci = rng.choice(unsat_clauses)
                cv = clause_vars[ci]
                
                best_var = -1
                best_gain = -float('inf')
                
                for var_idx, sign in cv:
                    new_val = not assignment[var_idx]
                    this_clause_gain = 1 if new_val == (sign == 1) else 0
                    
                    other_gain = 0
                    for (oci, osign) in global_var_to_clauses[var_idx]:
                        if oci == ci:
                            continue
                        cur_lit_sat = assignment[var_idx] == (osign == 1)
                        new_lit_sat = new_val == (osign == 1)
                        if cur_lit_sat and not new_lit_sat:
                            if clause_sat_global[oci] == 1:
                                other_gain -= 1
                        elif not cur_lit_sat and new_lit_sat:
                            other_gain += 1
                    
                    gain = this_clause_gain + other_gain
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var_idx
                
                if best_var < 0:
                    break
                
                old_val = assignment[best_var]
                new_val = not old_val
                assignment[best_var] = new_val
                solution[best_var] = new_val
                
                for (oci, osign) in global_var_to_clauses[best_var]:
                    cur_lit_sat_old = old_val == (osign == 1)
                    cur_lit_sat_new = new_val == (osign == 1)
                    if cur_lit_sat_old and not cur_lit_sat_new:
                        clause_sat_global[oci] -= 1
                        if clause_sat_global[oci] == 0:
                            unsat_count_global += 1
                    elif not cur_lit_sat_old and cur_lit_sat_new:
                        clause_sat_global[oci] += 1
                        if clause_sat_global[oci] == 1:
                            unsat_count_global -= 1
                
                step += 1
    
    # Return final assignment
    return [solution.get(v, False) for v in range(n)]
# EVOLVE-BLOCK-END
