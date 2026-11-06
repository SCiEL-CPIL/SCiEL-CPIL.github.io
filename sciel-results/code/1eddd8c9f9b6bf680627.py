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
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=bool)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = lit > 0
    
    # Build reverse mapping: variable -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = int(clause_vars[ci, j])
            s = bool(clause_signs[ci, j])
            var_to_clauses[v].append((ci, s))
    
    # Build lookup tables for fast unsat counting
    # For each variable, precompute which clauses it appears in and with what sign
    var_clause_indices = [np.array([ci for ci, s in var_to_clauses[v]], dtype=np.int32) for v in range(n)]
    var_clause_signs = [np.array([s for ci, s in var_to_clauses[v]], dtype=bool) for v in range(n)]
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses using vectorized ops."""
        assignment_arr = np.array(assignment, dtype=bool)
        # For each clause, check if any literal is satisfied
        # literal_satisfied[i,j] = assignment[clause_vars[i,j]] == clause_signs[i,j]
        lit_sat = assignment_arr[clause_vars] == clause_signs
        # A clause is satisfied if any of its 3 literals are satisfied
        clause_sat = lit_sat.any(axis=1)
        return bool(clause_sat.all())
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        assignment_arr = np.array(assignment, dtype=bool)
        lit_sat = assignment_arr[clause_vars] == clause_signs
        clause_sat = lit_sat.any(axis=1)
        return int((~clause_sat).sum())
    
    def unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        assignment_arr = np.array(assignment, dtype=bool)
        lit_sat = assignment_arr[clause_vars] == clause_signs
        clause_sat = lit_sat.any(axis=1)
        return np.where(~clause_sat)[0].tolist()
    
    def fast_count_unsat(assignment):
        """Fast unsat count using precomputed tables. Returns (count, unsat_indices)."""
        assignment_arr = np.array(assignment, dtype=bool)
        # For each clause, compute if satisfied
        # A clause is unsat if ALL its literals are falsified
        # literal falsified: assignment[var] != sign
        lit_falsified = assignment_arr[clause_vars] != clause_signs
        clause_unsat = lit_falsified.all(axis=1)
        unsat_idx = np.where(clause_unsat)[0]
        return int(len(unsat_idx)), unsat_idx
    
    def dpll_solve(assignment, time_limit):
        """
        DPLL solver with unit propagation and pure literal elimination.
        Returns a satisfying assignment if found within time_limit, else None.
        """
        dpll_assign = [None] * n
        
        def propagate(assign, active_clauses):
            """
            Perform unit propagation and pure literal elimination.
            Returns (success, new_active_clauses)
            success is False if a contradiction is found.
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
                        v = int(vars_[j])
                        s = bool(signs[j])
                        if assign[v] is None:
                            unassigned.append((v, s))
                        elif assign[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return False, active_clauses
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        if assign[v] is not None:
                            continue
                        assign[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                if active_clauses:
                    pure_vars = {}
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        for j in range(3):
                            v = int(vars_[j])
                            s = bool(signs[j])
                            if assign[v] is None:
                                if v not in pure_vars:
                                    pure_vars[v] = set()
                                pure_vars[v].add(s)
                    
                    for v, signs_set in pure_vars.items():
                        if len(signs_set) == 1:
                            s = signs_set.pop()
                            assign[v] = s
                            changed = True
                    
                    if changed:
                        new_active = []
                        for ci in active_clauses:
                            vars_ = clause_vars[ci]
                            signs = clause_signs[ci]
                            satisfied = False
                            for j in range(3):
                                v = int(vars_[j])
                                s = bool(signs[j])
                                if assign[v] is not None:
                                    if assign[v] == s:
                                        satisfied = True
                                        break
                            if not satisfied:
                                new_active.append(ci)
                        active_clauses = new_active
            
            return True, active_clauses
        
        # Initial propagation
        success, active_clauses = propagate(dpll_assign, list(range(m)))
        
        if not success:
            return None
        
        if not active_clauses:
            for v in range(n):
                if dpll_assign[v] is None:
                    dpll_assign[v] = assignment[v] if assignment[v] is not None else False
            return dpll_assign
        
        def dpll_recurse(current_assign, active_clauses, depth=0):
            if time.time() > time_limit:
                return None
            
            if depth > 60:
                return None
            
            # Propagate
            success, active_clauses = propagate(current_assign, active_clauses)
            if not success:
                return None
            
            if not active_clauses:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = assignment[v] if assignment[v] is not None else False
                return current_assign[:]
            
            # Pick branch variable using activity heuristic
            var_count = {}
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                for j in range(3):
                    v = int(vars_[j])
                    if current_assign[v] is None:
                        var_count[v] = var_count.get(v, 0) + 1
            
            if not var_count:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = assignment[v] if assignment[v] is not None else False
                return current_assign[:]
            
            branch_var = max(var_count, key=var_count.get)
            
            # Try both values, prefer the one that satisfies more clauses
            # Count how many active clauses would be satisfied by each value
            pos_sat = 0
            neg_sat = 0
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                signs = clause_signs[ci]
                for j in range(3):
                    if int(vars_[j]) == branch_var:
                        if bool(signs[j]):
                            pos_sat += 1
                        else:
                            neg_sat += 1
            
            if pos_sat >= neg_sat:
                order = [True, False]
            else:
                order = [False, True]
            
            for val in order:
                if time.time() > time_limit:
                    return None
                
                saved_assign = current_assign[:]
                current_assign[branch_var] = val
                
                success, new_active = propagate(current_assign, active_clauses)
                
                if success:
                    if not new_active:
                        for v in range(n):
                            if current_assign[v] is None:
                                current_assign[v] = assignment[v] if assignment[v] is not None else False
                        return current_assign[:]
                    else:
                        result = dpll_recurse(current_assign, new_active, depth + 1)
                        if result is not None:
                            return result
                
                # Restore
                current_assign = saved_assign
            
            return None
        
        return dpll_recurse(dpll_assign, active_clauses)
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random()
    
    # Strategy:
    # 1. Try DPLL first with a time limit
    # 2. If DPLL fails, use WalkSAT with min-unsat heuristic and restarts
    # 3. Use DPLL as a polish step for near-satisfying assignments
    
    # Phase 1: DPLL with time limit
    dpll_deadline = time.time() + 30.0
    dpll_result = dpll_solve([False] * n, dpll_deadline)
    if dpll_result is not None:
        if is_satisfied(dpll_result):
            return dpll_result
    
    # Phase 2: WalkSAT with min-unsat heuristic and restarts
    # Use vectorized operations for speed
    
    def walksat_step(assignment, noise=0.05):
        """
        One WalkSAT step with min-unsat heuristic.
        Returns the new assignment.
        """
        assignment_arr = np.array(assignment, dtype=bool)
        
        # Find unsatisfied clauses
        lit_falsified = assignment_arr[clause_vars] != clause_signs
        clause_unsat = lit_falsified.all(axis=1)
        unsat_idx = np.where(clause_unsat)[0]
        
        if len(unsat_idx) == 0:
            return assignment, 0
        
        # Pick a random unsatisfied clause
        clause_idx = int(rng.choice(unsat_idx))
        vars_ = clause_vars[clause_idx]
        signs = clause_signs[clause_idx]
        
        # Find false literals in this clause
        false_mask = assignment_arr[vars_] != signs
        false_vars = vars_[false_mask]
        
        if len(false_vars) == 0:
            # Shouldn't happen, but pick a random variable
            v = int(rng.choice(vars_))
        else:
            # For each candidate variable, compute the delta in unsat count
            # if we flip it
            best_delta = float('inf')
            best_var = int(false_vars[0])
            candidates = false_vars.tolist()
            
            for v in candidates:
                # Compute how many clauses become satisfied/unsatisfied
                # when we flip variable v
                v_clauses = var_clause_indices[v]
                v_signs = var_clause_signs[v]
                
                # Current state of these clauses
                # A clause is currently unsat if all its literals are falsified
                # After flipping v, the literal for v changes
                # If v was falsifying its literal (assignment[v] != sign), 
                # after flip it becomes satisfying
                # If v was satisfying its literal, after flip it becomes falsifying
                
                # Count change in unsat clauses
                # For each clause containing v:
                # - If clause is currently unsat:
                #   - If v's literal was falsified (which it is since v is in false_vars of unsat clause),
                #     after flip v's literal becomes satisfied, so clause becomes sat. delta = -1
                #   - But wait, we need to check if other literals also matter
                #   Actually, a clause is unsat iff ALL literals are falsified.
                #   If we flip v and v's literal was falsified, it becomes satisfied,
                #   so the clause becomes satisfied. delta = -1.
                # - If clause is currently sat:
                #   - If v's literal was satisfying, after flip it becomes falsified.
                #     The clause may become unsat if v was the only satisfying literal.
                #     We need to check if other literals are still satisfied.
                #   - If v's literal was falsified, after flip it becomes satisfied,
                #     clause stays sat. delta = 0.
                
                # More carefully:
                # Let's compute for each clause containing v whether it changes status
                
                # Get the other two literals for each clause
                # clause_vars has shape (m, 3)
                # For clause ci, the other literals are at positions other than v's position
                
                # Find positions of v in each clause
                # This is expensive, so let's use a different approach
                
                # Alternative: just compute the new unsat count for the affected clauses
                # Affected clauses are those containing v
                affected = v_clauses
                
                if len(affected) == 0:
                    delta = 0
                else:
                    # Current unsat status of affected clauses
                    # A clause is unsat if all 3 literals are falsified
                    # For affected clauses, we need to check all 3 literals
                    
                    # Current assignment
                    # After flipping v, only v's value changes
                    
                    # For each affected clause, determine if it's currently unsat
                    # and if it would be unsat after the flip
                    
                    # Currently unsat: all literals falsified
                    # After flip: v's literal flips, others same
                    
                    # Let's compute:
                    # For each affected clause ci:
                    #   Let v_pos be the position of v in clause ci
                    #   Other literals: ci, 1-v_pos, 2-v_pos (or however)
                    
                    # This is getting complex. Let's use a simpler approach:
                    # Compute the current unsat count for affected clauses
                    # and the new unsat count after flipping v
                    
                    # Current:
                    cur_lit_fals = assignment_arr[clause_vars[affected]] != clause_signs[affected]
                    cur_unsat = cur_lit_fals.all(axis=1)
                    cur_unsat_count = int(cur_unsat.sum())
                    
                    # New: flip v
                    new_assign = assignment_arr.copy()
                    new_assign[v] = not new_assign[v]
                    new_lit_fals = new_assign[clause_vars[affected]] != clause_signs[affected]
                    new_unsat = new_lit_fals.all(axis=1)
                    new_unsat_count = int(new_unsat.sum())
                    
                    delta = new_unsat_count - cur_unsat_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                elif delta == best_delta and rng.random() < noise:
                    best_var = v
            
            v = best_var
        
        assignment[v] = not assignment[v]
        return assignment, len(unsat_idx)
    
    # WalkSAT with adaptive noise and restarts
    max_restarts = 200
    base_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Adaptive noise: start high, decrease over restarts
        noise = 0.1 * (0.95 ** restart) + 0.01
        
        # Run WalkSAT
        for step in range(base_steps):
            if time.time() > deadline - 2.0:
                break
            
            assignment, unsat_count = walksat_step(assignment, noise)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
        
        # If we got close, try more steps with lower noise
        if best_unsat <= 3:
            extra_steps = 2000
            for step in range(extra_steps):
                if time.time() > deadline - 2.0:
                    break
                assignment, unsat_count = walksat_step(assignment, 0.01)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
    
    # Phase 3: If we have a near-satisfying assignment, try DPLL polish
    if best_assignment is not None and best_unsat <= 5:
        polish_deadline = time.time() + 20.0
        polish_result = dpll_solve(best_assignment, polish_deadline)
        if polish_result is not None:
            if is_satisfied(polish_result):
                return polish_result
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: random assignment
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
