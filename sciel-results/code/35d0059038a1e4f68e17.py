"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause data as numpy arrays for efficient WalkSAT
    clause_vars = np.array([[v for v, s in c] for c in clauses_internal], dtype=np.int32)
    clause_signs = np.array([[s for v, s in c] for c in clauses_internal], dtype=np.int8)
    
    def check_satisfied(assign_bool):
        """Check if assignment satisfies all clauses using numpy."""
        assign_np = np.array(assign_bool, dtype=np.int8)
        # For each clause, check if any literal is satisfied
        # Literal (var, sign) is satisfied if assign[var] == (sign == 1)
        # sign == 1 means positive literal, satisfied when assign[var] == True
        # sign == -1 means negative literal, satisfied when assign[var] == False
        # satisfied_literal[i,j] = 1 if literal j in clause i is satisfied
        expected = (clause_signs == 1).astype(np.int8)
        # assign_np[clause_vars] gives us the assignment for each literal
        actual = assign_np[clause_vars]
        # A positive literal (sign=1) is satisfied if actual == 1
        # A negative literal (sign=-1) is satisfied if actual == 0
        sat_pos = (actual == 1) & (clause_signs == 1)
        sat_neg = (actual == 0) & (clause_signs == -1)
        clause_sat = sat_pos | sat_neg
        return clause_sat.all()
    
    def count_unsatisfied(assign_bool):
        """Count unsatisfied literals per clause and total unsatisfied clauses."""
        assign_np = np.array(assign_bool, dtype=np.int8)
        expected = (clause_signs == 1).astype(np.int8)
        actual = assign_np[clause_vars]
        sat_pos = (actual == 1) & (clause_signs == 1)
        sat_neg = (actual == 0) & (clause_signs == -1)
        clause_sat = sat_pos | sat_neg
        # Unsatisfied literals per clause
        lit_sat = sat_pos | sat_neg
        unsat_lits = 3 - lit_sat.sum(axis=1)
        total_unsat_clauses = (~clause_sat).sum()
        return unsat_lits, total_unsat_clauses, clause_sat
    
    # DPLL with unit propagation (quick attempt with time limit)
    def dpll(assignment, decision_level, trail, time_limit):
        """
        assignment: list of length n, each element is None (unassigned), True, or False
        decision_level: current decision level
        trail: list of (var, value, level) for backtracking
        Returns: assignment if SAT, None if UNSAT
        """
        if time.time() > time_limit:
            return None
        
        # Unit propagation
        while True:
            unit_vars = []
            conflict = False
            
            for ci, clause in enumerate(clauses_internal):
                unassigned = []
                for var, sign in clause:
                    if assignment[var] is None:
                        unassigned.append((var, sign))
                    elif assignment[var] == (sign == 1):
                        break  # clause satisfied
                else:
                    if len(unassigned) == 0:
                        return None  # conflict
                    elif len(unassigned) == 1:
                        unit_vars.append(unassigned[0])
            
            if not unit_vars:
                break
            
            for var, sign in unit_vars:
                val = (sign == 1)
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return None  # conflict
                    continue
                assignment[var] = val
                trail.append((var, val, decision_level))
            
            if time.time() > time_limit:
                return None
        
        # Check if all assigned
        all_assigned = all(assignment[var] is not None for var in range(n))
        if all_assigned:
            return assignment
        
        # VSIDS variable selection
        var_scores = [0] * n
        for clause in clauses_internal:
            clause_unsat = True
            for var, sign in clause:
                if assignment[var] is not None and assignment[var] == (sign == 1):
                    clause_unsat = False
                    break
            if clause_unsat:
                for var, sign in clause:
                    if assignment[var] is None:
                        var_scores[var] += 1
        
        best_var = -1
        best_score = -1
        for var in range(n):
            if assignment[var] is None and var_scores[var] > best_score:
                best_score = var_scores[var]
                best_var = var
        
        if best_var == -1:
            return None
        
        # Decide: try True first
        assignment[best_var] = True
        trail.append((best_var, True, decision_level))
        
        result = dpll(assignment, decision_level + 1, trail, time_limit)
        if result is not None:
            return result
        
        # Backtrack
        while trail and trail[-1][2] > decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        if trail and trail[-1][2] == decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        
        # Try False
        assignment[best_var] = False
        trail.append((best_var, False, decision_level))
        
        result = dpll(assignment, decision_level + 1, trail, time_limit)
        if result is not None:
            return result
        
        # Backtrack again
        while trail and trail[-1][2] > decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        if trail and trail[-1][2] == decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        
        return None
    
    # Try DPLL with a short time limit
    dpll_time_limit = start_time + 30.0  # Give DPLL 30 seconds
    assignment = [None] * n
    trail = []
    result = dpll(assignment, 0, trail, dpll_time_limit)
    
    if result is not None:
        return [bool(x) for x in result]
    
    # Fallback: WalkSAT with DLS restarts
    rng = random.Random(42)
    
    def walksat_with_dls(max_time):
        """WalkSAT with Dynamic Large-Sum restart strategy."""
        best_assign = None
        best_unsat = float('inf')
        
        # DLS parameters
        dls_alpha = 0.025
        dls_beta = 2.0
        dls_gamma = 1.0
        dls_delta = 1.0
        dls_epsilon = 0.0
        dls_zeta = 1.0
        dls_omega = 0.1
        dls_theta = 1.0
        dls_kappa = 1.0
        dls_lambda = 1.0
        dls_mu = 1.0
        dls_nu = 1.0
        dls_xi = 1.0
        dls_omicron = 1.0
        dls_rho = 1.0
        dls_sigma = 1.0
        dls_tau = 1.0
        dls_upsilon = 1.0
        dls_phi = 1.0
        dls_chi = 1.0
        dls_psi = 1.0
        dls_omega2 = 1.0
        dls_omega3 = 1.0
        dls_omega4 = 1.0
        dls_omega5 = 1.0
        
        # Simplified DLS: track best unsat count, restart when stuck
        restart_count = 0
        best_unsat_this_run = float('inf')
        patience = 10000  # steps without improvement before restart
        
        while time.time() < max_time:
            # Initialize random assignment
            w_assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Track history for DLS
            history = []
            current_unsat = float('inf')
            no_improve_count = 0
            run_best = float('inf')
            
            for step in range(200000):  # Max steps per run
                if time.time() > max_time:
                    break
                
                # Compute unsatisfied clauses
                unsat_lits, total_unsat, clause_sat = count_unsatisfied(w_assignment)
                
                if total_unsat == 0:
                    return w_assignment
                
                # Update best
                if total_unsat < run_best:
                    run_best = total_unsat
                    no_improve_count = 0
                    if total_unsat < best_unsat:
                        best_unsat = total_unsat
                        best_assign = list(w_assignment)
                else:
                    no_improve_count += 1
                
                # DLS restart: if we've been stuck for too long, restart
                if no_improve_count > patience:
                    break
                
                # Find unsatisfied clauses
                unsat_indices = np.where(~clause_sat)[0]
                if len(unsat_indices) == 0:
                    return w_assignment
                
                # Pick clause with minimum unsatisfied literals
                min_unsat_lits = np.min(unsat_lits[unsat_indices])
                candidates = unsat_indices[unsat_lits[unsat_indices] == min_unsat_lits]
                chosen_clause = rng.choice(candidates)
                
                # Get literals in this clause
                clause_vars_list = clause_vars[chosen_clause]
                clause_signs_list = clause_signs[chosen_clause]
                
                # Try to find best flip using lookahead
                best_var = None
                best_delta = float('inf')
                
                for idx in range(len(clause_vars_list)):
                    var = int(clause_vars_list[idx])
                    sign = int(clause_signs_list[idx])
                    
                    # If this literal is already satisfied, skip
                    current_val = w_assignment[var]
                    if current_val == (sign == 1):
                        continue
                    
                    # Compute delta: change in total unsatisfied clauses
                    delta = 0
                    for ci, s in var_to_clauses[var]:
                        # Check if clause ci was satisfied before flip
                        was_sat = False
                        for v, sg in clauses_internal[ci]:
                            if v == var:
                                if current_val == (sg == 1):
                                    was_sat = True
                                    break
                            else:
                                if w_assignment[v] == (sg == 1):
                                    was_sat = True
                                    break
                        
                        # Check if clause ci would be satisfied after flip
                        new_val = not current_val
                        would_be_sat = False
                        for v, sg in clauses_internal[ci]:
                            if v == var:
                                if new_val == (sg == 1):
                                    would_be_sat = True
                                    break
                            else:
                                if w_assignment[v] == (sg == 1):
                                    would_be_sat = True
                                    break
                        
                        if was_sat and not would_be_sat:
                            delta += 1
                        elif not was_sat and would_be_sat:
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var
                
                # If no good flip found, pick random unsatisfied literal
                if best_var is None:
                    unsat_lits_in_clause = []
                    for idx in range(len(clause_vars_list)):
                        var = int(clause_vars_list[idx])
                        sign = int(clause_signs_list[idx])
                        if w_assignment[var] != (sign == 1):
                            unsat_lits_in_clause.append(var)
                    if unsat_lits_in_clause:
                        best_var = rng.choice(unsat_lits_in_clause)
                
                if best_var is not None:
                    w_assignment[best_var] = not w_assignment[best_var]
            
            # End of run, check if we should continue
            if time.time() > max_time - 5:
                break
        
        return best_assign
    
    # Run WalkSAT with DLS for remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        result = walksat_with_dls(deadline - 5)
        if result is not None:
            # Verify and return
            if check_satisfied(result):
                return result
            return result
    
    # Final fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]


# EVOLVE-BLOCK-END
