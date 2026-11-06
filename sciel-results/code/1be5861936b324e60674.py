"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 330s kill
    
    # Convert clauses to 0-based indices for faster access
    # Store as (var_index, sign) where sign is 1 for positive, 0 for negative
    clauses_0 = []
    for c in clauses:
        c0 = [(abs(l) - 1, 1 if l > 0 else 0) for l in c]
        clauses_0.append(c0)
    
    # Try DPLL first with a time limit
    result = _dpll(n, clauses_0, deadline)
    if result is not None:
        return result
    
    # Fall back to WalkSAT with vectorized operations
    result = _walksat(n, clauses_0, deadline)
    return result


def _dpll(n, clauses, deadline):
    """DPLL with unit propagation and pure literal elimination."""
    
    def check_time():
        return time.time() > deadline
    
    # assignment: -1 unassigned, 0 False, 1 True
    assignment = [-1] * n
    # active_clauses: list of clause indices still active
    active = set(range(len(clauses)))
    
    # Precompute literal presence for pure literal elimination
    # For each variable, track which clauses contain it positively/negatively
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for var, sign in c:
            if sign:
                pos_clauses[var].append(i)
            else:
                neg_clauses[var].append(i)
    
    def propagate():
        """Unit propagation. Returns False if conflict, True otherwise."""
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            unit_var = -1
            unit_sign = 0
            for ci in list(active):
                c = clauses[ci]
                # Count unassigned literals
                unassigned = []
                satisfied = False
                for var, sign in c:
                    if assignment[var] != -1:
                        if assignment[var] == sign:
                            satisfied = True
                            break
                    else:
                        unassigned.append((var, sign))
                if satisfied:
                    active.discard(ci)
                    continue
                if len(unassigned) == 0:
                    return False  # conflict
                if len(unassigned) == 1:
                    unit_var, unit_sign = unassigned[0]
                    active.discard(ci)
                    break
            if unit_var != -1:
                assignment[unit_var] = unit_sign
                changed = True
        
        # Pure literal elimination
        for var in range(n):
            if assignment[var] != -1:
                continue
            # Check if variable appears with only one sign in active clauses
            has_pos = False
            has_neg = False
            for ci in pos_clauses[var]:
                if ci in active:
                    has_pos = True
                    break
            for ci in neg_clauses[var]:
                if ci in active:
                    has_neg = True
                    break
            if has_pos and not has_neg:
                assignment[var] = 1
            elif has_neg and not has_pos:
                assignment[var] = 0
            else:
                continue
            # Remove clauses satisfied by this assignment
            for ci in list(active):
                c = clauses[ci]
                for v, s in c:
                    if v == var and assignment[v] == s:
                        active.discard(ci)
                        break
            if check_time():
                return None  # timeout signal
        
        return True
    
    def is_satisfied():
        for ci in active:
            c = clauses[ci]
            for var, sign in c:
                if assignment[var] == sign:
                    break
            else:
                return False
        return True
    
    def choose_var():
        """Choose an unassigned variable using a simple heuristic."""
        # Count occurrences in active clauses
        counts = [0] * n
        for ci in active:
            for var, sign in clauses[ci]:
                if assignment[var] == -1:
                    counts[var] += 1
        max_count = 0
        best_var = -1
        for var in range(n):
            if assignment[var] == -1 and counts[var] > max_count:
                max_count = counts[var]
                best_var = var
        return best_var
    
    # Initialize
    prop_result = propagate()
    if prop_result is None:
        return None  # timeout
    if prop_result is False:
        return None  # unsat (shouldn't happen for planted SAT)
    
    if is_satisfied():
        return [assignment[i] == 1 for i in range(n)]
    
    # Recursive DPLL
    def dpll_rec(depth):
        if check_time():
            return None
        
        # Check if all satisfied
        if is_satisfied():
            return [assignment[i] == 1 for i in range(n)]
        
        var = choose_var()
        if var == -1:
            return None
        
        # Try True
        assignment[var] = 1
        prop_result = propagate()
        if prop_result is None:
            assignment[var] = -1
            return None
        if prop_result is not False:
            result = dpll_rec(depth + 1)
            if result is not None:
                return result
        assignment[var] = -1
        
        # Try False
        assignment[var] = 0
        prop_result = propagate()
        if prop_result is None:
            assignment[var] = -1
            return None
        if prop_result is not False:
            result = dpll_rec(depth + 1)
            if result is not None:
                return result
        assignment[var] = -1
        
        return None
    
    return dpll_rec(0)


def _walksat(n, clauses, deadline):
    """Vectorized WalkSAT with restarts."""
    rng = random.Random(42)
    end_time = deadline
    
    # Precompute clause data for vectorized operations
    # For each clause, store variable indices and signs as arrays
    clause_vars = np.array([[var for var, sign in c] for c in clauses], dtype=np.int32)
    clause_signs = np.array([[sign for var, sign in c] for c in clauses], dtype=np.int32)
    num_clauses = len(clauses)
    
    # Random initial assignment
    assignment = np.random.random(n) < 0.5
    # Convert to int array: 1 for True, 0 for False
    assign_int = assignment.astype(np.int32)
    
    def check_satisfied():
        # For each clause, check if any literal is satisfied
        # literal satisfied if assign_int[var] == sign
        # Create matrix: rows=clauses, cols=literals
        # sat_matrix[i,j] = 1 if clause i literal j is satisfied
        # var_indices: shape (num_clauses, 3)
        # assign_int[var_indices]: shape (num_clauses, 3)
        # clause_signs: shape (num_clauses, 3)
        # satisfied where assign_int[var_indices] == clause_signs
        sat = (assign_int[clause_vars] == clause_signs)
        # A clause is satisfied if any literal is satisfied
        clause_sat = sat.any(axis=1)
        return clause_sat.all()
    
    def count_unsat():
        sat = (assign_int[clause_vars] == clause_signs)
        clause_sat = sat.any(axis=1)
        return num_clauses - clause_sat.sum()
    
    # Try multiple restarts
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > end_time - 5:
            break
        
        if restart > 0:
            assign_int = (np.random.random(n) < 0.5).astype(np.int32)
        
        if check_satisfied():
            return assign_int.astype(bool)
        
        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > end_time - 2:
                break
            
            # Find unsatisfied clauses
            sat = (assign_int[clause_vars] == clause_signs)
            clause_sat = sat.any(axis=1)
            unsat_indices = np.where(~clause_sat)[0]
            
            if len(unsat_indices) == 0:
                return assign_int.astype(bool)
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            clause = clauses[clause_idx]
            
            # Count how many literals in this clause would be satisfied if we flip each variable
            # For each literal in the clause, count how many other literals in the clause
            # are currently satisfied (so flipping this var won't break them)
            best_var = -1
            best_score = -1
            for var, sign in clause:
                # If we flip var, does this literal become satisfied?
                current_val = assign_int[var]
                new_val = 1 - current_val
                this_lit_sat = (new_val == sign)
                
                # Count how many other literals in the clause are currently satisfied
                # (they will remain satisfied since we're only flipping var)
                other_sat = 0
                for v2, s2 in clause:
                    if v2 == var:
                        continue
                    if assign_int[v2] == s2:
                        other_sat += 1
                
                score = other_sat
                if this_lit_sat:
                    score += 10  # bonus for satisfying the literal
                if score > best_score:
                    best_score = score
                    best_var = var
            
            if best_var == -1:
                # Random flip
                var = rng.choice(clause)[0]
            else:
                var = best_var
            
            assign_int[var] = 1 - assign_int[var]
            
            if check_satisfied():
                return assign_int.astype(bool)
    
    # Return best effort
    return assign_int.astype(bool)
# EVOLVE-BLOCK-END
