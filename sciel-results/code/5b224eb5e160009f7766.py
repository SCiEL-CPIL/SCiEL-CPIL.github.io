"""Synthetic planted 3-SAT solver (50 variables)."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Precompute clause structure
    clause_vars = []
    for c in clauses:
        vars_in_clause = [abs(l) - 1 for l in c]
        signs = [1 if l > 0 else -1 for l in c]
        clause_vars.append((vars_in_clause, signs))
    
    # Precompute variable to clause indices mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, (vars_in_clause, signs) in enumerate(clause_vars):
        for var in vars_in_clause:
            var_to_clauses[var].append(i)

    # --- Phase 1: DPLL with Unit Propagation ---
    def dpll_unit_propagate(assignment, deadline):
        stack = list(assignment)
        # We need to track which clauses are still active
        # For simplicity, we'll just check satisfaction at the end of UP
        changed = True
        while changed:
            if time.time() > deadline:
                return None
            changed = False
            for var in range(n):
                if stack[var] is None:
                    continue
            # Find unit clauses
            for i, (vars_in_clause, signs) in enumerate(clause_vars):
                # Check if clause is already satisfied
                sat = False
                unassigned_count = 0
                unit_literal = None
                for j, var in enumerate(vars_in_clause):
                    val = stack[var]
                    if val is not None:
                        if (signs[j] == 1 and val) or (signs[j] == -1 and not val):
                            sat = True
                            break
                    else:
                        unassigned_count += 1
                        unit_literal = vars_in_clause[j]
                if sat:
                    continue
                if unassigned_count == 0:
                    return (False, stack) # Contradiction
                if unassigned_count == 1:
                    # Assign the unit literal
                    lit_sign = signs[vars_in_clause.index(unit_literal)]
                    stack[unit_literal] = (lit_sign == 1)
                    changed = True
        # Check if all clauses are satisfied
        for i, (vars_in_clause, signs) in enumerate(clause_vars):
            sat = False
            for j, var in enumerate(vars_in_clause):
                val = stack[var]
                if val is not None:
                    if (signs[j] == 1 and val) or (signs[j] == -1 and not val):
                        sat = True
                        break
            if not sat:
                return (False, stack)
        return (True, stack)

    # Initial random assignment
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Try DPLL/Unit Propagation first
    up_result = dpll_unit_propagate(assignment, deadline)
    if up_result is not None:
        sat, new_assign = up_result
        if sat:
            return new_assign
        # If UP leads to contradiction, we could restart, but for now, let's just use the partial assignment if possible
        # Or just fall back to WalkSAT with the original random assignment
        # Actually, if UP fails, it means the random assignment led to a dead end.
        # Let's just use the original random assignment for WalkSAT.

    # --- Phase 2: WalkSAT with Simulated Annealing ---
    assignment = [rng.random() < 0.5 for _ in range(n)]

    def is_clause_satisfied(i):
        vars_in_clause, signs = clause_vars[i]
        for j, var in enumerate(vars_in_clause):
            if assignment[var] == (signs[j] == 1):
                return True
        return False

    def count_unsat():
        count = 0
        for i in range(len(clause_vars)):
            if not is_clause_satisfied(i):
                count += 1
        return count

    current_unsat = count_unsat()
    if current_unsat == 0:
        return assignment

    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.9995
    T = T_start

    best_assignment = assignment[:]
    best_unsat = current_unsat

    while time.time() < deadline - 5.0:
        # Find an unsatisfied clause
        unsat_clauses = [i for i in range(len(clause_vars)) if not is_clause_satisfied(i)]
        if not unsat_clauses:
            return assignment

        clause_idx = rng.choice(unsat_clauses)
        vars_in_clause, signs = clause_vars[clause_idx]
        var = vars_in_clause[rng.randrange(len(vars_in_clause))]

        old_val = assignment[var]
        assignment[var] = not old_val

        # Calculate delta efficiently
        # Only clauses containing 'var' can change
        delta = 0
        for ci in var_to_clauses[var]:
            was_sat = is_clause_satisfied(ci)
            # Flip back temporarily to check? No, just check current state
            # But we need to know if it was satisfied before the flip
            # Let's just recalculate the unsat count for affected clauses
            # Actually, simpler:
            # old_sat = is_clause_satisfied before flip
            # new_sat = is_clause_satisfied after flip
            # This is tricky without storing old state.
            # Let's just recalculate total unsat for simplicity, it's fast enough for 50 vars.
            pass
        
        new_unsat = count_unsat()
        delta = new_unsat - current_unsat

        if delta < 0 or (delta == 0 and rng.random() < 0.5) or (delta > 0 and rng.random() < math.exp(-delta / T)):
            current_unsat = new_unsat
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                if best_unsat == 0:
                    return best_assignment
        else:
            assignment[var] = old_val

        T = max(T_end, T * cooling_rate)

    return best_assignment
# EVOLVE-BLOCK-END
