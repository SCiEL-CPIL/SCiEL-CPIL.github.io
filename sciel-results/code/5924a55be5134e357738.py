"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from collections import deque


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    # clause_vars: (m, 3) int array of variable indices
    # clause_signs: (m, 3) int array of signs (1 or -1)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
            
    # Build variable-to-clause index for efficient delta calculation
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
            
    # Build variable adjacency for connected components
    var_adj = [[] for _ in range(n)]
    for i in range(m):
        v0, v1, v2 = clause_vars[i, 0], clause_vars[i, 1], clause_vars[i, 2]
        var_adj[v0].append(v1)
        var_adj[v0].append(v2)
        var_adj[v1].append(v0)
        var_adj[v1].append(v2)
        var_adj[v2].append(v0)
        var_adj[v2].append(v1)
        
    # Find connected components
    visited = [False] * n
    components = []
    for start_var in range(n):
        if not visited[start_var]:
            comp = []
            queue = deque([start_var])
            visited[start_var] = True
            while queue:
                v = queue.popleft()
                comp.append(v)
                for neighbor in var_adj[v]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
            
    # Map each variable to its component index
    var_to_comp = [-1] * n
    for ci, comp in enumerate(components):
        for v in comp:
            var_to_comp[v] = ci
            
    # Group clauses by component
    comp_clauses = [[] for _ in components]
    for ci in range(m):
        comp_idx = var_to_comp[clause_vars[ci, 0]]
        comp_clauses[comp_idx].append(ci)
        
    # Helper: Evaluate clauses given an assignment array
    def evaluate_clauses(assign_np):
        """
        assign_np: (n,) int8 array (0 or 1)
        Returns:
            clause_sat: (m,) bool array
            unsat_lits: (m,) int array (number of unsatisfied literals per clause)
        """
        # actual: (m, 3) where actual[i,j] is the value of the variable in clause i, literal j
        actual = assign_np[clause_vars]
        # sat: (m, 3) bool array, True if literal is satisfied
        # If sign is 1, satisfied if actual is 1. If sign is -1, satisfied if actual is 0.
        sat = (actual * clause_signs) == 1
        clause_sat = sat.any(axis=1)
        unsat_lits = 3 - sat.sum(axis=1)
        return clause_sat, unsat_lits

    # Helper: WalkSAT solver for a subset of variables and clauses
    def walksat_solve(comp_vars, comp_clause_indices, time_budget, seed):
        if not comp_clause_indices:
            return {v: False for v in comp_vars}
            
        local_n = len(comp_vars)
        if local_n == 0:
            return {}
            
        # Map global vars to local indices
        var_map = {v: i for i, v in enumerate(comp_vars)}
        
        # Build local clause arrays
        local_m = len(comp_clause_indices)
        local_clause_vars = np.zeros((local_m, 3), dtype=np.int32)
        local_clause_signs = np.zeros((local_m, 3), dtype=np.int8)
        
        for i, ci in enumerate(comp_clause_indices):
            for j in range(3):
                gv = clause_vars[ci, j]
                local_clause_vars[i, j] = var_map[gv]
                local_clause_signs[i, j] = clause_signs[ci, j]
                
        # Local variable to clause index
        local_var_to_clauses = [[] for _ in range(local_n)]
        for i in range(local_m):
            for j in range(3):
                v = local_clause_vars[i, j]
                local_var_to_clauses[v].append(i)
                
        def local_evaluate(assign_np):
            actual = assign_np[local_clause_vars]
            sat = (actual * local_clause_signs) == 1
            clause_sat = sat.any(axis=1)
            unsat_lits = 3 - sat.sum(axis=1)
            return clause_sat, unsat_lits

        rng = random.Random(seed)
        best_assign = None
        best_unsat = float('inf')
        
        # Multiple restarts
        # Allocate time per restart dynamically
        max_restarts = 10
        time_per_restart = time_budget / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Random initialization
            w_assignment = np.array([1 if rng.random() < 0.5 else 0 for _ in range(local_n)], dtype=np.int8)
            clause_sat, unsat_lits = local_evaluate(w_assignment)
            total_unsat = (~clause_sat).sum()
            
            if total_unsat == 0:
                return {v: bool(w_assignment[var_map[v]]) for v in comp_vars}
                
            run_best_unsat = total_unsat
            run_best_assign = w_assignment.copy()
            no_improve = 0
            patience = 10000 # Steps without improvement
            
            while time.time() < deadline - 2.0:
                if no_improve > patience:
                    break
                    
                if total_unsat < run_best_unsat:
                    run_best_unsat = total_unsat
                    run_best_assign = w_assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if total_unsat == 0:
                    break
                
                # Select a random unsatisfied clause
                unsat_indices = np.where(~clause_sat)[0]
                chosen_clause_idx = int(rng.choice(unsat_indices))
                
                # Select a literal in that clause to flip
                # Weighted by number of other satisfied literals in clauses containing this var? 
                # Simple WalkSAT: pick random literal from the clause
                # Better: pick literal that minimizes delta (breaks least other clauses)
                # Let's use a simplified min-delta heuristic which is faster than full VSIDS
                best_var = -1
                best_delta = float('inf')
                
                for j in range(3):
                    v = int(local_clause_vars[chosen_clause_idx, j])
                    sign = int(local_clause_signs[chosen_clause_idx, j])
                    current_val = w_assignment[v]
                    
                    # If literal is already satisfied, flipping it would definitely break this clause
                    # So we only consider unsatisfied literals
                    if current_val * sign == 1:
                        continue
                        
                    # Calculate delta: how many other clauses will be broken/satisfied
                    delta = 0
                    for ci in local_var_to_clauses[v]:
                        if ci == chosen_clause_idx:
                            continue
                        # Check if clause ci is currently satisfied
                        # If it is, and flipping v breaks it, delta++
                        # If it is not, and flipping v satisfies it, delta--
                        
                        # Quick check: is v the only satisfied literal in ci?
                        # Or is v the only unsatisfied literal in ci?
                        
                        # We can compute this by checking the clause ci
                        # But doing this for every v in the clause is O(deg(v))
                        # For 3-SAT, degree is usually manageable.
                        
                        # Let's just count satisfied literals in ci
                        # If satisfied count == 1 and it's v, then flipping breaks it (delta +1)
                        # If unsatisfied count == 1 and it's v, then flipping satisfies it (delta -1)
                        
                        # To avoid O(3) check per clause, we can precompute or just check.
                        # Since m is 150*? and n=150, this is fine.
                        
                        is_sat = False
                        num_sat = 0
                        for k in range(3):
                            vv = int(local_clause_vars[ci, k])
                            ss = int(local_clause_signs[ci, k])
                            if w_assignment[vv] * ss == 1:
                                num_sat += 1
                                if vv == v:
                                    is_sat = True
                                    
                        if num_sat == 1 and is_sat:
                            delta += 1
                        elif num_sat == 0:
                            # Check if v is the only unsatisfied one
                            # If num_sat == 0, all 3 are unsatisfied.
                            # Flipping v makes num_sat 1. Clause becomes satisfied.
                            # So delta -= 1
                            delta -= 1
                        elif num_sat >= 2:
                            # Clause remains satisfied regardless (unless v was the only one? No, num_sat>=2 means others keep it sat)
                            # Wait, if num_sat >= 2, and we flip v (which is unsat), num_sat stays >=2.
                            # So delta 0.
                            pass
                            
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                        
                if best_var != -1:
                    w_assignment[best_var] = 1 - w_assignment[best_var]
                    clause_sat, unsat_lits = local_evaluate(w_assignment)
                    total_unsat = (~clause_sat).sum()
                else:
                    # Fallback: random flip from the clause
                    j = rng.randint(0, 2)
                    v = int(local_clause_vars[chosen_clause_idx, j])
                    w_assignment[v] = 1 - w_assignment[v]
                    clause_sat, unsat_lits = local_evaluate(w_assignment)
                    total_unsat = (~clause_sat).sum()
                    
            if run_best_unsat < best_unsat:
                best_unsat = run_best_unsat
                best_assign = run_best_assign
                
        # Map back to global
        if best_assign is not None:
            sol = {}
            for v in comp_vars:
                sol[v] = bool(best_assign[var_map[v]])
            return sol
        else:
            # Fallback random
            sol = {}
            for v in comp_vars:
                sol[v] = rng.random() < 0.5
            return sol

    # Solve each connected component independently
    final_assignment = np.zeros(n, dtype=np.int8)
    
    # Sort components by size (largest first)
    comp_order = sorted(range(len(components)), key=lambda i: -len(components[i]))
    
    # Allocate time budget
    # Reserve 10% for global polish
    polish_time = 20.0
    search_time = deadline - time.time() - polish_time - 5.0
    
    if search_time < 5.0:
        # Not enough time, just random
        final_assignment = np.array([1 if random.random() < 0.5 else 0 for _ in range(n)], dtype=np.int8)
        return final_assignment.tolist()
        
    # Distribute time proportionally to clause count (harder components get more time)
    total_clauses_in_comps = sum(len(comp_clauses[i]) for i in comp_order)
    if total_clauses_in_comps == 0:
        total_clauses_in_comps = 1
        
    for comp_idx in comp_order:
        if time.time() > deadline - polish_time - 5.0:
            break
            
        comp_vars = components[comp_idx]
        comp_clause_indices = comp_clauses[comp_idx]
        
        # Time proportional to clauses
        frac = len(comp_clause_indices) / total_clauses_in_comps
        # Ensure minimum 1s if possible
        t_budget = max(1.0, search_time * frac)
        t_budget = min(t_budget, deadline - time.time() - polish_time - 1.0)
        
        if t_budget < 0.5:
            # Too little time, random
            for v in comp_vars:
                final_assignment[v] = 1 if random.random() < 0.5 else 0
            continue
            
        sol = walksat_solve(comp_vars, comp_clause_indices, t_budget, seed=comp_idx)
        for v, val in sol.items():
            final_assignment[v] = 1 if val else 0
            
    # Global Polish
    # If any clauses are still unsatisfied, run a global WalkSAT on the whole instance
    # This handles cases where component solving failed or left conflicts (though components are independent)
    # Actually, if components are independent, solving each separately solves the whole thing.
    # But WalkSAT might fail to find a solution in a component.
    # So we run a global pass to fix any remaining unsatisfied clauses.
    
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 1.0:
        clause_sat, unsat_lits = evaluate_clauses(final_assignment)
        total_unsat = (~clause_sat).sum()
        
        if total_unsat > 0:
            rng = random.Random(42)
            best_assign = final_assignment.copy()
            best_unsat = total_unsat
            w_assignment = final_assignment.copy()
            no_improve = 0
            patience = 5000
            
            while time.time() < deadline - 2.0:
                if no_improve > patience:
                    break
                    
                if total_unsat < best_unsat:
                    best_unsat = total_unsat
                    best_assign = w_assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if total_unsat == 0:
                    break
                    
                unsat_indices = np.where(~clause_sat)[0]
                chosen_clause_idx = int(rng.choice(unsat_indices))
                
                # Min-delta flip
                best_var = -1
                best_delta = float('inf')
                
                for j in range(3):
                    v = int(clause_vars[chosen_clause_idx, j])
                    sign = int(clause_signs[chosen_clause_idx, j])
                    current_val = w_assignment[v]
                    
                    if current_val * sign == 1:
                        continue
                        
                    delta = 0
                    for ci in var_to_clauses[v]:
                        if ci == chosen_clause_idx:
                            continue
                        num_sat = 0
                        is_sat_v = False
                        for k in range(3):
                            vv = int(clause_vars[ci, k])
                            ss = int(clause_signs[ci, k])
                            if w_assignment[vv] * ss == 1:
                                num_sat += 1
                                if vv == v:
                                    is_sat_v = True
                                    
                        if num_sat == 1 and is_sat_v:
                            delta += 1
                        elif num_sat == 0:
                            delta -= 1
                            
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                        
                if best_var != -1:
                    w_assignment[best_var] = 1 - w_assignment[best_var]
                else:
                    j = rng.randint(0, 2)
                    v = int(clause_vars[chosen_clause_idx, j])
                    w_assignment[v] = 1 - w_assignment[v]
                    
                clause_sat, unsat_lits = evaluate_clauses(w_assignment)
                total_unsat = (~clause_sat).sum()
                
            if best_unsat < total_unsat:
                final_assignment = best_assign
            else:
                final_assignment = w_assignment
    else:
        # No time for polish, just check if we are good
        pass

    # Final verification
    clause_sat, _ = evaluate_clauses(final_assignment)
    if not clause_sat.all():
        # If still unsatisfied, we return the best effort. 
        # The score is fraction of solved instances, so this one is 0.
        pass
        
    return final_assignment.tolist()


# EVOLVE-BLOCK-END
