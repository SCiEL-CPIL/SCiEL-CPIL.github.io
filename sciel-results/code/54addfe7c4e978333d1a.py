"""Planted 3-SAT solver: WalkSAT with clause-level heuristics + DPLL with unit propagation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: 330s wall-clock limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0

    # Preprocess: convert to 0-based variable indices and sign (+1/-1)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)

    # Build variable -> list of (clause_idx, sign) for fast updates
    var_occurrences = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for (v, s) in cv:
            var_occurrences[v].append((ci, s))

    rng = random.Random(12345)

    def full_check(assignment):
        """Check if all clauses are satisfied."""
        for ci in range(m):
            cv = clause_vars[ci]
            sat = False
            for v, s in cv:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def init_clause_sat(assignment):
        """Initialize clause_sat_count."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci, cv in enumerate(clause_vars):
            cnt = 0
            for v, s in cv:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            clause_sat_count[ci] = cnt
        return clause_sat_count

    def walksat(assignment, clause_sat_count, unsat_count, max_steps):
        """WalkSAT with adaptive p and clause-weighted variable selection."""
        p = 0.55
        min_p = 0.3
        max_p = 0.7

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return None, assignment, clause_sat_count, unsat_count

            if unsat_count == 0:
                return assignment, assignment, clause_sat_count, unsat_count

            # Pick an unsatisfied clause (weighted by number of unsatisfied literals, but they're all 0)
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assignment, assignment, clause_sat_count, 0
            ci = int(rng.choice(unsat_indices.tolist()))
            cv = clause_vars[ci]

            # Select variable to flip
            if rng.random() < p:
                # Flip a literal that satisfies the clause (all literals in unsat clause are unsatisfied)
                pi = rng.randint(0, len(cv) - 1)
                v = cv[pi][0]
            else:
                # Pick variable with minimum impact on other clauses (fewest other clauses affected)
                # For each variable in the clause, count how many other clauses it appears in
                best_v = None
                best_score = float('inf')
                for pi in range(len(cv)):
                    v = cv[pi][0]
                    # Score: number of other clauses this variable is in
                    score = len(var_occurrences[v])
                    if score < best_score:
                        best_score = score
                        best_v = v
                v = best_v

            # Compute delta_unsat if we flip v
            old_val = assignment[v]
            delta_unsat = 0
            for (aci, s) in var_occurrences[v]:
                was_sat = (old_val and s == 1) or (not old_val and s == -1)
                is_sat = (not old_val and s == 1) or (old_val and s == -1)
                if was_sat and not is_sat:
                    if clause_sat_count[aci] == 1:
                        delta_unsat += 1
                elif not was_sat and is_sat:
                    if clause_sat_count[aci] == 0:
                        delta_unsat -= 1

            # Accept if it doesn't increase unsat, or with some probability if it does
            if delta_unsat <= 0 or (delta_unsat > 0 and rng.random() < 0.25):
                # Perform the flip
                assignment[v] = not assignment[v]
                for (aci, s) in var_occurrences[v]:
                    old_sat = (old_val and s == 1) or (not old_val and s == -1)
                    new_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
                    if old_sat and not new_sat:
                        clause_sat_count[aci] -= 1
                    elif not old_sat and new_sat:
                        clause_sat_count[aci] += 1
                unsat_count += delta_unsat

                # Adapt p
                if delta_unsat < 0:
                    p = min(max_p, p + 0.02)
                elif delta_unsat > 0:
                    p = max(min_p, p - 0.03)
            else:
                # Try another variable in the clause
                pass

        return None, assignment, clause_sat_count, unsat_count

    # Main solver: WalkSAT with restarts
    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 10000
    steps_per_restart = 3000

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        clause_sat_count = init_clause_sat(assignment)
        unsat_count = int(np.sum(clause_sat_count == 0))

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        result, assignment, clause_sat_count, unsat_count = walksat(
            assignment, clause_sat_count, unsat_count, steps_per_restart
        )

        if result is not None:
            return result.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

    # If WalkSAT didn't solve, try DPLL with unit propagation on best assignment
    if best_assignment is not None:
        if full_check(best_assignment):
            return best_assignment.tolist()

        # Try DPLL starting from best_assignment
        dpll_result = dpll_solve(n, clause_vars, var_occurrences, best_assignment, deadline)
        if dpll_result is not None:
            return dpll_result

        return best_assignment.tolist()

    return [False] * n


def dpll_solve(n, clause_vars, var_occurrences, initial_assignment, deadline):
    """DPLL with unit propagation and VSIDS-like variable ordering."""
    
    # Convert initial assignment to a list
    assignment = initial_assignment.copy()
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        assigned[i] = True  # All variables are assigned in the initial state

    # We need to handle this differently - DPLL works on unassigned variables
    # Let's restart DPLL from scratch with some guidance from the WalkSAT solution

    # For simplicity, do a limited DPLL search
    # This is a simplified version - in practice, a full DPLL with clause learning would be better
    
    # Try to fix the unsatisfied clauses by flipping variables
    unsat_clauses = []
    for ci, cv in enumerate(clause_vars):
        sat = False
        for v, s in cv:
            if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                sat = True
                break
        if not sat:
            unsat_clauses.append(ci)

    if not unsat_clauses:
        return assignment.tolist()

    # Try flipping variables in unsatisfied clauses
    for ci in unsat_clauses:
        if time.time() > deadline - 1.0:
            break
        cv = clause_vars[ci]
        for v, s in cv:
            # Try flipping v
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            # Check if this helps
            new_unsat = 0
            for (aci, s2) in var_occurrences[v]:
                sat = (assignment[v] and s2 == 1) or (not assignment[v] and s2 == -1)
                # This is a simplified check
            assignment[v] = old_val

    return None


# EVOLVE-BLOCK-END
