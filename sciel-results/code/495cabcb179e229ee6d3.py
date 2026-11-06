"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    deadline = time.time() + 295.0
    
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
    
    # Build reverse mapping: variable -> list of (clause_idx, position_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_vars[ci][j]
            var_to_clauses[v].append((ci, j))
    
    def is_satisfied(assignment):
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
    
    # ============ DPLL with proper backtracking ============
    def dpll(assignment, time_limit):
        """
        DPLL solver with unit propagation and pure literal elimination.
        assignment: list of True/False/None (None = unassigned)
        Returns satisfying assignment or None.
        """
        # Work on a copy
        current = assignment[:]
        
        def propagate(assign):
            """
            Unit propagation + pure literal elimination.
            Returns (success, new_active_clauses).
            Modifies assign in-place.
            """
            active = list(range(m))
            changed = True
            while changed:
                changed = False
                new_active = []
                for ci in active:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assign[v] is None:
                            unassigned.append((v, s))
                        elif assign[v] == s:
                            satisfied = True
                            break
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return False, active
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        assign[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                active = new_active
                
                # Pure literal elimination
                if active:
                    pure_vars = {}
                    for ci in active:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if assign[v] is None:
                                if v not in pure_vars:
                                    pure_vars[v] = set()
                                pure_vars[v].add(s)
                    for v, sset in pure_vars.items():
                        if len(sset) == 1:
                            assign[v] = sset.pop()
                            changed = True
                    if changed:
                        new_active = []
                        for ci in active:
                            vars_ = clause_vars[ci]
                            signs = clause_signs[ci]
                            satisfied = False
                            for j in range(3):
                                v = vars_[j]
                                s = signs[j]
                                if assign[v] is not None and assign[v] == s:
                                    satisfied = True
                                    break
                            if not satisfied:
                                new_active.append(ci)
                        active = new_active
            return True, active
        
        # Initial propagation
        success, active = propagate(current)
        if not success:
            return None
        if not active:
            for v in range(n):
                if current[v] is None:
                    current[v] = False
            return current
        
        # Recursive DPLL
        def dpll_recursive(assign, active):
            if time.time() > time_limit:
                return None
            
            if not active:
                for v in range(n):
                    if assign[v] is None:
                        assign[v] = False
                return assign[:]
            
            # Choose variable: most constrained (appears most in active clauses)
            var_count = {}
            for ci in active:
                vars_ = clause_vars[ci]
                for j in range(3):
                    v = vars_[j]
                    if assign[v] is None:
                        var_count[v] = var_count.get(v, 0) + 1
            
            if not var_count:
                for v in range(n):
                    if assign[v] is None:
                        assign[v] = False
                return assign[:]
            
            branch_var = max(var_count, key=var_count.get)
            
            for val in [True, False]:
                if time.time() > time_limit:
                    return None
                saved = assign[:]
                assign[branch_var] = val
                success, new_active = propagate(assign)
                if success:
                    if not new_active:
                        for v in range(n):
                            if assign[v] is None:
                                assign[v] = False
                        return assign[:]
                    else:
                        result = dpll_recursive(assign, new_active)
                        if result is not None:
                            return result
                assign = saved
            return None
        
        return dpll_recursive(current, active)
    
    # ============ WalkSAT with incremental sat_count ============
    def walksat_polish(assignment, time_limit, max_steps=20000):
        """
        Local search polish using incremental clause satisfaction counts.
        Returns improved assignment.
        """
        current = assignment[:]
        
        # Compute initial sat_count: number of true literals per clause
        sat_count = [0] * m
        for ci in range(m):
            vars_ = clause_vars[ci]
            signs = clause_signs[ci]
            for j in range(3):
                if current[vars_[j]] == signs[j]:
                    sat_count[ci] += 1
        
        def get_unsat():
            return [ci for ci in range(m) if sat_count[ci] == 0]
        
        best_unsat = len(get_unsat())
        best_assign = current[:]
        
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            
            unsat = get_unsat()
            if not unsat:
                return current
            
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assign = current[:]
            
            # Pick random unsat clause
            ci = random.choice(unsat)
            vars_ = clause_vars[ci]
            signs = clause_signs[ci]
            
            # Find falsified literals in this clause
            false_vars = [vars_[j] for j in range(3) if current[vars_[j]] != signs[j]]
            
            if not false_vars:
                break
            
            # For each candidate, compute delta in #unsat clauses
            best_var = None
            best_delta = float('inf')
            for v in false_vars:
                delta = 0
                for (cj, j) in var_to_clauses[v]:
                    v_sign = clause_signs[cj][j]
                    # Currently: current[v] != v_sign (since v is in false_vars of an unsat clause,
                    # but v might satisfy other clauses)
                    cur_sat = (current[v] == v_sign)
                    if cur_sat:
                        # Flipping makes it false
                        if sat_count[cj] == 1:
                            delta += 1  # clause becomes unsat
                    else:
                        # Flipping makes it true
                        if sat_count[cj] == 0:
                            delta -= 1  # clause becomes sat
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
            
            # Randomization: 20% chance pick random false literal
            if random.random() < 0.2:
                v = random.choice(false_vars)
            else:
                v = best_var
            
            # Apply flip
            current[v] = not current[v]
            for (cj, j) in var_to_clauses[v]:
                v_sign = clause_signs[cj][j]
                if current[v] == v_sign:
                    sat_count[cj] += 1
                else:
                    sat_count[cj] -= 1
        
        # Return best found
        if best_unsat < len(get_unsat()):
            return best_assign
        return current
    
    rng = random.Random()
    
    # ============ Main strategy ============
    
    # Phase 1: Try DPLL with generous time limit
    dpll_deadline = time.time() + 60.0
    dpll_result = dpll([None] * n, dpll_deadline)
    if dpll_result is not None and is_satisfied(dpll_result):
        return dpll_result
    
    # Phase 2: WalkSAT with restarts
    best_assignment = None
    best_unsat = float('inf')
    
    max_restarts = 200
    steps_per_restart = 15000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if is_satisfied(assignment):
            return assignment
        
        # WalkSAT polish
        polished = walksat_polish(assignment, deadline - 1.0, steps_per_restart)
        
        if is_satisfied(polished):
            return polished
        
        u = count_unsat(polished)
        if u < best_unsat:
            best_unsat = u
            best_assignment = polished[:]
    
    # Phase 3: If near-satisfying, try DPLL polish from that assignment
    if best_assignment is not None and best_unsat <= 10:
        polish_deadline = time.time() + 30.0
        # Convert to partial assignment: fix variables that are "confident"
        # For simplicity, just run DPLL with all unassigned
        dpll_result = dpll([None] * n, polish_deadline)
        if dpll_result is not None and is_satisfied(dpll_result):
            return dpll_result
        
        # Also try WalkSAT polish again with more steps
        final_polish = walksat_polish(best_assignment, deadline - 1.0, 50000)
        if is_satisfied(final_polish):
            return final_polish
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
