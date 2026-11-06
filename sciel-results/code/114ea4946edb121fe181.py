"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock kill, leave 10s margin
    deadline = time.time() + 320.0
    
    # Preprocess: convert clauses to 0-based indices and store signs
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = []
        signs = []
        for lit in c:
            v = abs(lit) - 1
            s = lit > 0
            vars_.append(v)
            signs.append(s)
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Build reverse mapping: variable -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_vars[ci][j]
            s = clause_signs[ci][j]
            var_to_clauses[v].append((ci, s))
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        unsat = []
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    def dpll_solve(assignment, time_limit):
        """
        DPLL solver with unit propagation and pure literal elimination.
        Returns a satisfying assignment if found within time_limit, else None.
        """
        dpll_assign = [None] * n
        
        def propagate(assignment, active_clauses):
            """
            Perform unit propagation and pure literal elimination.
            Returns (success, new_assignment, new_active_clauses)
            """
            changed = True
            while changed:
                changed = False
                
                # Unit propagation
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assignment[v] is None:
                            unassigned.append((v, s))
                        elif assignment[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return False, assignment, active_clauses
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        assignment[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                pure_vars = {}
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assignment[v] is None:
                            if v not in pure_vars:
                                pure_vars[v] = set()
                            pure_vars[v].add(s)
                
                for v, signs_set in pure_vars.items():
                    if len(signs_set) == 1:
                        s = signs_set.pop()
                        assignment[v] = s
                        changed = True
                
                if changed:
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if assignment[v] is not None and assignment[v] == s:
                                satisfied = True
                                break
                        if not satisfied:
                            new_active.append(ci)
                    active_clauses = new_active
            
            return True, assignment, active_clauses
        
        success, dpll_assign, active_clauses = propagate(dpll_assign, list(range(m)))
        
        if not success:
            return None
        
        if not active_clauses:
            for v in range(n):
                if dpll_assign[v] is None:
                    dpll_assign[v] = assignment[v] if assignment[v] is not None else False
            return dpll_assign
        
        # Recursive DPLL with improved branching heuristic
        def dpll_recursive(current_assign, active_clauses, heuristic_assign, time_limit):
            if time.time() > time_limit:
                return None
            
            # Propagate
            changed = True
            while changed:
                changed = False
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is None:
                            unassigned.append((v, s))
                        elif current_assign[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return None
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        current_assign[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                pure_vars = {}
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is None:
                            if v not in pure_vars:
                                pure_vars[v] = set()
                            pure_vars[v].add(s)
                
                for v, signs_set in pure_vars.items():
                    if len(signs_set) == 1:
                        s = signs_set.pop()
                        current_assign[v] = s
                        changed = True
                
                if changed:
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is not None and current_assign[v] == s:
                                satisfied = True
                                break
                        if not satisfied:
                            new_active.append(ci)
                    active_clauses = new_active
            
            if not active_clauses:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = heuristic_assign[v] if heuristic_assign[v] is not None else False
                return current_assign
            
            # Pick branch variable using weighted degree heuristic
            var_score = {}
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                for j in range(3):
                    v = vars_[j]
                    if current_assign[v] is None:
                        var_score[v] = var_score.get(v, 0) + 1
            
            if not var_score:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = heuristic_assign[v] if heuristic_assign[v] is not None else False
                return current_assign
            
            branch_var = max(var_score, key=var_score.get)
            
            for val in [True, False]:
                if time.time() > time_limit:
                    return None
                
                saved_assign = current_assign[:]
                current_assign[branch_var] = val
                
                # Propagate
                changed = True
                while changed:
                    changed = False
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        
                        unassigned = []
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is None:
                                unassigned.append((v, s))
                            elif current_assign[v] == s:
                                satisfied = True
                                break
                        
                        if satisfied:
                            continue
                        elif len(unassigned) == 0:
                            current_assign = saved_assign
                            break
                        elif len(unassigned) == 1:
                            v, s = unassigned[0]
                            current_assign[v] = s
                            changed = True
                        else:
                            new_active.append(ci)
                    
                    if current_assign is not saved_assign:
                        active_clauses = new_active
                    
                    # Pure literal elimination
                    pure_vars = {}
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is None:
                                if v not in pure_vars:
                                    pure_vars[v] = set()
                                pure_vars[v].add(s)
                    
                    for v, signs_set in pure_vars.items():
                        if len(signs_set) == 1:
                            s = signs_set.pop()
                            current_assign[v] = s
                            changed = True
                    
                    if changed:
                        new_active = []
                        for ci in active_clauses:
                            vars_ = clause_vars[ci]
                            signs = clause_signs[ci]
                            satisfied = False
                            for j in range(3):
                                v = vars_[j]
                                s = signs[j]
                                if current_assign[v] is not None and current_assign[v] == s:
                                    satisfied = True
                                    break
                            if not satisfied:
                                new_active.append(ci)
                        active_clauses = new_active
                
                if current_assign is not saved_assign:
                    if not active_clauses:
                        for v in range(n):
                            if current_assign[v] is None:
                                current_assign[v] = heuristic_assign[v] if heuristic_assign[v] is not None else False
                        return current_assign
                    else:
                        result = dpll_recursive(current_assign, active_clauses, heuristic_assign, time_limit)
                        if result is not None:
                            return result
                
                current_assign = saved_assign
            
            return None
        
        return dpll_recursive(dpll_assign, active_clauses, assignment, time_limit)
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random()
    
    # Phase 1: DPLL with time limit
    dpll_deadline = time.time() + 40.0
    dpll_result = dpll_solve([None] * n, dpll_deadline)
    if dpll_result is not None:
        if is_satisfied(dpll_result):
            return dpll_result
    
    # Phase 2: Improved WalkSAT with incremental tracking and adaptive noise
    max_restarts = 300
    steps_per_restart = 15000
    
    # Precompute clause data for faster access
    clause_data = [(clause_vars[i], clause_signs[i]) for i in range(m)]
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsatisfied clauses
        unsat_set = set()
        for i in range(m):
            vars_, signs = clause_data[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                unsat_set.add(i)
        
        current_unsat = len(unsat_set)
        
        # Adaptive noise parameter
        noise = 0.01 + 0.02 * (restart / max_restarts)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 1.0:
                break
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(list(unsat_set))
            vars_, signs = clause_data[clause_idx]
            
            # Find false literals in this clause
            false_vars = []
            for j in range(3):
                if assignment[vars_[j]] != signs[j]:
                    false_vars.append(vars_[j])
            
            if not false_vars:
                v = rng.choice(vars_)
            else:
                # Use min-unsat heuristic with randomization
                if rng.random() < noise:
                    v = rng.choice(false_vars)
                else:
                    # Evaluate each false variable flip
                    best_v = false_vars[0]
                    best_delta = float('inf')
                    for fv in false_vars:
                        # Count how many clauses would become unsatisfied if we flip fv
                        delta = 0
                        for ci, s in var_to_clauses[fv]:
                            if ci not in unsat_set:  # currently satisfied
                                # Check if this clause would become unsatisfied
                                vars_c, signs_c = clause_data[ci]
                                all_false = True
                                for j in range(3):
                                    v_c = vars_c[j]
                                    s_c = signs_c[j]
                                    if v_c == fv:
                                        # This literal would become false after flip
                                        continue
                                    if assignment[v_c] == s_c:
                                        all_false = False
                                        break
                                if all_false:
                                    delta += 1
                        if delta < best_delta:
                            best_delta = delta
                            best_v = fv
                    v = best_v
            
            # Flip the variable
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update unsat_set incrementally
            for ci, s in var_to_clauses[v]:
                vars_c, signs_c = clause_data[ci]
                was_unsat = ci in unsat_set
                
                # Check if clause is now satisfied
                sat = False
                for j in range(3):
                    if assignment[vars_c[j]] == signs_c[j]:
                        sat = True
                        break
                
                if sat:
                    if was_unsat:
                        unsat_set.remove(ci)
                        current_unsat -= 1
                else:
                    if not was_unsat:
                        unsat_set.add(ci)
                        current_unsat += 1
    
    # Phase 3: Polish with DPLL if close to solution
    if best_assignment is not None and best_unsat <= 10:
        polish_deadline = time.time() + 20.0
        polish_result = dpll_solve(best_assignment, polish_deadline)
        if polish_result is not None:
            if is_satisfied(polish_result):
                return polish_result
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
