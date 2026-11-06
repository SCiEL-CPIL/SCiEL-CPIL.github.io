"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """WalkSAT with incremental clause tracking, weighted variable selection, and adaptive restarts."""
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    m = len(clauses)

    start_time = time.time()
    deadline = 295.0  # Safety margin before 300s limit

    # Precompute clause literal info: for each clause, list of (var_idx, is_positive)
    clause_lits = []
    for c in clauses:
        clause_lits.append([(abs(l) - 1, l > 0) for l in c])

    # Build reverse index: for each variable, list of (clause_idx, is_positive)
    # where is_positive indicates whether the literal in that clause is positive
    var_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var_idx, is_pos in lits:
            var_clauses[var_idx].append((ci, is_pos))

    def make_initial_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_clause_satisfied(assignment):
        """Return list of booleans: clause_satisfied[ci] = True if clause ci is satisfied."""
        clause_sat = [False] * m
        for ci, lits in enumerate(clause_lits):
            for var_idx, is_pos in lits:
                if assignment[var_idx] == is_pos:
                    clause_sat[ci] = True
                    break
        return clause_sat

    def count_unsat(clause_sat):
        return m - sum(1 for s in clause_sat if s)

    def update_after_flip(var_idx, new_val, clause_sat, assignment):
        """Update clause_sat after flipping var_idx to new_val. Returns new unsat_count delta info."""
        # For each clause containing var_idx, check if it's still satisfied
        for ci, is_pos in var_clauses[var_idx]:
            was_sat = clause_sat[ci]
            # Check if this clause is still satisfied after the flip
            if was_sat:
                # If it was satisfied, check if it's still satisfied
                # It might become unsatisfied if var_idx was the only satisfying literal
                still_sat = False
                for v2, p2 in clause_lits[ci]:
                    if assignment[v2] == p2:
                        still_sat = True
                        break
                clause_sat[ci] = still_sat
            else:
                # It was unsatisfied; check if new_val satisfies it
                if new_val == is_pos:
                    clause_sat[ci] = True

    best_assignment = None
    best_unsat = m + 1

    max_total_steps = 200000
    p_greedy = 0.5
    restart_interval = 2000
    max_restarts = 50
    perturbation_sizes = [3, 5, 8, 12, 20]

    total_steps = 0
    restart_count = 0

    while total_steps < max_total_steps and restart_count < max_restarts:
        if time.time() - start_time > deadline:
            break

        # Initialize assignment and clause satisfaction
        assignment = make_initial_assignment()
        clause_sat = compute_clause_satisfied(assignment)
        unsat_count = count_unsat(clause_sat)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        stuck_counter = 0
        local_best_unsat = unsat_count

        for step in range(restart_interval):
            total_steps += 1
            if time.time() - start_time > deadline:
                break

            if unsat_count == 0:
                return assignment

            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                stuck_counter = 0
            else:
                stuck_counter += 1

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Find an unsatisfied clause
            # Use a small random sample to find one quickly
            found_ci = -1
            for _ in range(10):
                ci = rng.randint(0, m - 1)
                if not clause_sat[ci]:
                    found_ci = ci
                    break
            if found_ci == -1:
                # Fallback: scan
                for ci in range(m):
                    if not clause_sat[ci]:
                        found_ci = ci
                        break

            if found_ci == -1:
                break  # All satisfied (shouldn't happen if unsat_count > 0)

            clause = clause_lits[found_ci]

            # Weighted variable selection: prefer variables that appear in many unsatisfied clauses
            # Compute weight for each literal in this clause
            weights = []
            for var_idx, is_pos in clause:
                # Count how many unsatisfied clauses contain this variable
                w = 0
                for ci2, _ in var_clauses[var_idx]:
                    if not clause_sat[ci2]:
                        w += 1
                weights.append(w)

            # Decide greedy vs random
            if rng.random() < p_greedy:
                # Greedy: pick a literal that would satisfy the clause, weighted
                # All literals in an unsatisfied clause would satisfy it if flipped
                # Use weighted selection
                total_w = sum(weights)
                if total_w == 0:
                    idx = rng.randint(0, len(clause) - 1)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    idx = len(clause) - 1
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            idx = i
                            break
                var_idx, is_pos = clause[idx]
                assignment[var_idx] = is_pos
            else:
                # Random: pick a literal from the clause and flip it
                idx = rng.randint(0, len(clause) - 1)
                var_idx, _ = clause[idx]
                assignment[var_idx] = not assignment[var_idx]

            # Update clause satisfaction incrementally
            update_after_flip(var_idx, assignment[var_idx], clause_sat, assignment)

            # Recompute unsat count efficiently
            unsat_count = sum(1 for s in clause_sat if not s)

        # After restart_interval steps, if not solved, restart with perturbation
        if unsat_count > 0 and unsat_count >= best_unsat:
            restart_count += 1
            # Perturb the best assignment or current assignment
            base = best_assignment if best_assignment else assignment
            pert_size = perturbation_sizes[min(restart_count - 1, len(perturbation_sizes) - 1)]
            for _ in range(pert_size):
                vi = rng.randint(0, n - 1)
                base[vi] = not base[vi]
            # Continue with perturbed assignment in next iteration

    # Final check
    if best_assignment is not None:
        # Verify best_assignment
        final_sat = compute_clause_satisfied(best_assignment)
        if count_unsat(final_sat) == 0:
            return best_assignment

    # If we have a current assignment that's better, use it
    if assignment is not None:
        final_sat = compute_clause_satisfied(assignment)
        if count_unsat(final_sat) == 0:
            return assignment

    return best_assignment if best_assignment else assignment
# EVOLVE-BLOCK-END
