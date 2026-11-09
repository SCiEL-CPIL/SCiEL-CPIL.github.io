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
    deadline = start_time + 295.0  # 5s margin before 300s limit (330s wall clock)
    
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
        actual = assign_np[clause_vars]
        sat_pos = (actual == 1) & (clause_signs == 1)
        sat_neg = (actual == 0) & (clause_signs == -1)
        clause_sat = sat_pos | sat_neg
        return clause_sat.all()
    
    def count_unsatisfied(assign_bool):
        """Count unsatisfied clauses."""
        assign_np = np.array(assign_bool, dtype=np.int8)
        actual = assign_np[clause_vars]
        sat_pos = (actual == 1) & (clause_signs == 1)
        sat_neg = (actual == 0) & (clause_signs == -1)
        clause_sat = sat_pos | sat_neg
        total_unsat_clauses = int((~clause_sat).sum())
        return total_unsat_clauses, clause_sat
    
    # DPLL with unit propagation
    def dpll(assignment, decision_level, trail, time_limit):
        """
        assignment: list of length n, each element is None (unassigned), True, or False
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
    
    # Try DPLL with a time limit
    dpll_time_limit = start_time + 20.0  # Give DPLL 20 seconds
    assignment = [None] * n
    trail = []
    result = dpll(assignment, 0, trail, dpll_time_limit)
    
    if result is not None:
        return [bool(x) for x in result]
    
    # Fallback: WalkSAT with improved heuristics
    rng = random.Random(42)
    
    def walksat_with_restarts(max_time):
        """WalkSAT with multiple restarts and lookahead."""
        best_assign = None
        best_unsat = float('inf')
        
        # Parameters
        noise_start = 0.45  # Start with more noise
        noise_decay = 0.9999  # Gradually reduce noise
        patience = 5000  # Steps without improvement before restart
        
        elapsed = time.time() - start_time
        remaining = max_time - start_time
        
        while time.time() < max_time - 2:
            noise = noise_start * (noise_decay ** (elapsed * 10))
            noise = max(0.01, min(0.6, noise))
            
            # Initialize random assignment
            w_assignment = [rng.random() < 0.5 for _ in range(n)]
            
            current_unsat, clause_sat = count_unsatisfied(w_assignment)
            run_best = current_unsat
            no_improve_count = 0
            
            for step in range(100000):  # Max steps per run
                if time.time() > max_time - 2:
                    break
                
                if current_unsat == 0:
                    return w_assignment
                
                if current_unsat < run_best:
                    run_best = current_unsat
                    no_improve_count = 0
                    if current_unsat < best_unsat:
                        best_unsat = current_unsat
                        best_assign = list(w_assignment)
                else:
                    no_improve_count += 1
                
                if no_improve_count > patience:
                    break
                
                # Find unsatisfied clauses
                unsat_indices = np.where(~clause_sat)[0]
                if len(unsat_indices) == 0:
                    return w_assignment
                
                # Pick a random unsatisfied clause
                chosen_clause = int(rng.choice(unsat_indices))
                
                # Get literals in this clause
                clause_vars_list = clause_vars[chosen_clause]
                clause_signs_list = clause_signs[chosen_clause]
                
                # With probability noise, pick random unsatisfied literal
                # Otherwise, pick the literal that minimizes delta
                if rng.random() < noise:
                    unsat_lits = []
                    for idx in range(3):
                        var = int(clause_vars_list[idx])
                        sign = int(clause_signs_list[idx])
                        if w_assignment[var] != (sign == 1):
                            unsat_lits.append(var)
                    if unsat_lits:
                        best_var = rng.choice(unsat_lits)
                    else:
                        best_var = int(rng.choice(clause_vars_list))
                else:
                    # Lookahead: pick literal with minimum delta
                    best_var = None
                    best_delta = float('inf')
                    
                    for idx in range(3):
                        var = int(clause_vars_list[idx])
                        sign = int(clause_signs_list[idx])
                        
                        current_val = w_assignment[var]
                        if current_val == (sign == 1):
                            continue
                        
                        # Compute delta
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
                    
                    if best_var is None:
                        unsat_lits = []
                        for idx in range(3):
                            var = int(clause_vars_list[idx])
                            sign = int(clause_signs_list[idx])
                            if w_assignment[var] != (sign == 1):
                                unsat_lits.append(var)
                        if unsat_lits:
                            best_var = rng.choice(unsat_lits)
                    
                    if best_var is None:
                        best_var = int(rng.choice(clause_vars_list))
                
                w_assignment[best_var] = not w_assignment[best_var]
                
                # Update unsatisfied count incrementally
                current_unsat, clause_sat = count_unsatisfied(w_assignment)
                elapsed = time.time() - start_time
        
        return best_assign
    
    # Run WalkSAT for remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        result = walksat_with_restarts(deadline - 3)
        if result is not None:
            if check_satisfied(result):
                return result
            return result
    
    # Final fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]


# EVOLVE-BLOCK-END
