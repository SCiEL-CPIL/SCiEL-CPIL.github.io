"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Pre-process clauses: store as arrays of (var_index, is_positive)
    # var_index is 0-based
    clause_vars = []
    clause_polarity = []
    for c in clauses:
        vars_in_c = []
        pols_in_c = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            vars_in_c.append(v)
            pols_in_c.append(p)
        clause_vars.append(vars_in_c)
        clause_polarity.append(pols_in_c)
    
    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_vars):
        for v in c:
            var_to_clauses[v].append(i)
            
    # State variables for incremental updates
    assignment = [False] * n
    clause_sat = [False] * m
    unsat_count = m
    
    def update_clause_sat(ci):
        cvars = clause_vars[ci]
        cpols = clause_polarity[ci]
        for v, p in zip(cvars, cpols):
            if assignment[v] == p:
                clause_sat[ci] = True
                return
        clause_sat[ci] = False
        
    def init_state(assignment):
        nonlocal unsat_count
        unsat_count = 0
        for i in range(m):
            update_clause_sat(i)
            if not clause_sat[i]:
                unsat_count += 1
                
    def flip_var(v):
        nonlocal unsat_count
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        # Update all clauses containing v
        for ci in var_to_clauses[v]:
            was_sat = clause_sat[ci]
            # Check new satisfaction
            cvars_ci = clause_vars[ci]
            cpols_ci = clause_polarity[ci]
            is_sat = False
            for v2, p2 in zip(cvars_ci, cpols_ci):
                if assignment[v2] == p2:
                    is_sat = True
                    break
            clause_sat[ci] = is_sat
            
            if was_sat and not is_sat:
                unsat_count += 1
            elif not was_sat and is_sat:
                unsat_count -= 1

    # Initialize assignment randomly
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    init_state(assignment)
    
    if unsat_count == 0:
        return assignment
    
    best_assignment = assignment[:]
    best_unsat = unsat_count
    
    max_steps_per_restart = 100000
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        init_state(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            # Optimization: Pick a random unsatisfied clause if possible to avoid bias
            # For now, just find the first one, or a random index
            unsat_idx = -1
            # Try a few random indices to find an unsat clause quickly
            for _ in range(10):
                idx = rng.randint(0, m - 1)
                if not clause_sat[idx]:
                    unsat_idx = idx
                    break
            if unsat_idx == -1:
                # Fallback: linear scan
                for i in range(m):
                    if not clause_sat[i]:
                        unsat_idx = i
                        break
            
            if unsat_idx == -1:
                return assignment
            
            steps += 1
            
            # Get the variables in this clause
            cvars = clause_vars[unsat_idx]
            cpols = clause_polarity[unsat_idx]
            
            # For each variable in the clause, compute the score
            best_var = -1
            best_score = -10**9
            
            for v in cvars:
                gain = 0
                for ci in var_to_clauses[v]:
                    if clause_sat[ci]:
                        # Check if flipping v breaks it (v is the only satisfying literal)
                        cvars_ci = clause_vars[ci]
                        cpols_ci = clause_polarity[ci]
                        is_only = True
                        for v2, p2 in zip(cvars_ci, cpols_ci):
                            if v2 != v and assignment[v2] == p2:
                                is_only = False
                                break
                        if is_only:
                            gain -= 1
                    else:
                        # Check if flipping v satisfies it
                        cvars_ci = clause_vars[ci]
                        cpols_ci = clause_polarity[ci]
                        for idx2, v2 in enumerate(cvars_ci):
                            if v2 == v:
                                p2 = cpols_ci[idx2]
                                if (not assignment[v]) == p2:
                                    gain += 1
                                break
                
                if gain > best_score:
                    best_score = gain
                    best_var = v
            
            # With some probability, pick a random variable instead (noise)
            if rng.random() < 0.3:
                v = rng.choice(cvars)
            else:
                v = best_var
            
            if v >= 0:
                flip_var(v)
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            if unsat_count == 0:
                return assignment
            
            # Early exit if we're stuck and time is running out
            if time.time() > deadline - 5.0:
                break
    
    # Return the best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
