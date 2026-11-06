# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit

    # Convert clauses to 0-based signed literals
    # For each clause, store (var_idx, sign) where sign is +1 for positive, -1 for negative
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    m = len(clauses_0based)

    # Build data structures for efficient clause evaluation
    # For each variable, store list of (clause_idx, sign) where it appears
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # Precompute clause sizes (all should be 3 for 3-SAT)
    # Use numpy for vectorized operations where possible

    def _walksat_with_restart(rng, max_restarts=50, steps_per_restart=5000):
        """WalkSAT with restarts and adaptive flip selection."""
        best_assignment = None
        best_unsat_count = m + 1

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initial assignment
            assignment = [rng.random() < 0.5 for _ in range(n)]

            # Compute initial unsatisfied clause count
            # Track which clauses are unsatisfied
            clause_unsat = np.zeros(m, dtype=bool)
            for ci, clause in enumerate(clauses_0based):
                sat = False
                for var, sign in clause:
                    if assignment[var] == (sign > 0):
                        sat = True
                        break
                clause_unsat[ci] = not sat

            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

            # WalkSAT loop
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_assignment

                if unsat_count == 0:
                    return assignment

                # Pick a random unsatisfied clause
                unsat_indices = np.where(clause_unsat)[0]
                ci = int(rng.choice(unsat_indices))
                clause = clauses_0based[ci]

                # Count how many literals in this clause are satisfied
                # and identify which variable to flip
                # Use a scoring heuristic: prefer flipping a variable that
                # satisfies this clause and breaks few other unsatisfied clauses

                best_var = -1
                best_score = -float('inf')

                for var, sign in clause:
                    # Check if flipping this variable would satisfy the clause
                    # Currently the clause is unsatisfied, so all its literals are false
                    # Flipping var: new_val = not assignment[var]
                    # The literal (var, sign) is satisfied if new_val == (sign > 0)
                    # Since currently unsatisfied, assignment[var] != (sign > 0)
                    # So flipping will make it satisfied

                    # Score: number of other clauses this flip would break minus satisfy
                    # For efficiency, only consider variables in this clause
                    score = 0
                    for other_ci, other_sign in var_clauses[var]:
                        if other_ci == ci:
                            continue
                        # Would this flip satisfy or break the other clause?
                        other_clause = clauses_0based[other_ci]
                        # Check if other_clause is currently unsatisfied
                        if clause_unsat[other_ci]:
                            # If we flip var, does it satisfy this other clause?
                            # The literal in other clause for var has sign other_sign
                            # After flip, assignment[var] = not assignment[var]
                            # Satisfied if not assignment[var] == (other_sign > 0)
                            # Which is assignment[var] != (other_sign > 0)
                            # Currently, since other clause is unsat, all its literals are false
                            # So assignment[var] != (other_sign > 0) is true
                            # Thus flipping would satisfy this clause -> good
                            score += 1
                        else:
                            # Other clause is satisfied. Would flipping break it?
                            # The literal (var, other_sign) in other clause
                            # Currently satisfied: assignment[var] == (other_sign > 0)
                            # After flip: not assignment[var] == (other_sign > 0)?
                            # i.e., assignment[var] != (other_sign > 0) -> false
                            # So this literal becomes false
                            # But the clause might still be satisfied by other literals
                            # For simplicity, count as potentially breaking
                            # Check if var is the only satisfying literal
                            only_sat = True
                            for v2, s2 in other_clause:
                                if v2 == var:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                score -= 1

                    # Add random noise for exploration
                    score += rng.random() * 0.5

                    if score > best_score:
                        best_score = score
                        best_var = var

                if best_var == -1:
                    break

                # Flip the variable
                old_val = assignment[best_var]
                assignment[best_var] = not old_val

                # Update clause satisfaction
                for other_ci, other_sign in var_clauses[best_var]:
                    other_clause = clauses_0based[other_ci]
                    # Recompute if this clause is satisfied
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat

                if unsat_count == 0:
                    return assignment

                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]

        return best_assignment if best_assignment is not None else [False] * n

    def _local_search_with_tabu(rng, max_iters=20000):
        """Local search with tabu list to escape local minima."""
        assignment = [rng.random() < 0.5 for _ in range(n)]
        tabu = [0] * n  # tabu expiry step
        tabu_len = 50

        # Compute initial state
        clause_unsat = np.zeros(m, dtype=bool)
        for ci, clause in enumerate(clauses_0based):
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat

        unsat_count = int(np.sum(clause_unsat))
        if unsat_count == 0:
            return assignment

        best_assignment = assignment[:]
        best_unsat = unsat_count

        for step in range(max_iters):
            if time.time() > deadline:
                return best_assignment

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick random unsatisfied clause
            unsat_indices = np.where(clause_unsat)[0]
            if len(unsat_indices) == 0:
                return assignment
            ci = int(rng.choice(unsat_indices))
            clause = clauses_0based[ci]

            # Find best flip among variables in this clause
            best_var = -1
            best_delta = -float('inf')

            for var, sign in clause:
                if step < tabu[var]:
                    # Check if it's an aspiration move (improves best)
                    # Skip for simplicity
                    continue

                # Compute delta: change in unsat count if we flip var
                delta = 0
                for other_ci, other_sign in var_clauses[var]:
                    other_clause = clauses_0based[other_ci]
                    if clause_unsat[other_ci]:
                        # Flipping might satisfy this clause
                        # Check if var is the only way to satisfy it
                        # Actually, since clause is unsat, all literals are false
                        # Flipping var makes its literal true -> clause becomes sat
                        delta -= 1
                    else:
                        # Clause is sat. Flipping might break it if var is the only sat literal
                        only_sat = True
                        for v2, s2 in other_clause:
                            if v2 == var:
                                continue
                            if assignment[v2] == (s2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            delta += 1

                # Add noise
                delta += rng.random() * 0.3

                if delta < best_delta:
                    best_delta = delta
                    best_var = var

            if best_var == -1:
                # All variables tabu, pick randomly from clause
                var, sign = clause[rng.randrange(len(clause))]
                best_var = var

            # Flip
            assignment[best_var] = not assignment[best_var]
            tabu[best_var] = step + tabu_len

            # Update clause states
            for other_ci, other_sign in var_clauses[best_var]:
                other_clause = clauses_0based[other_ci]
                sat = False
                for v, s in other_clause:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                was_unsat = clause_unsat[other_ci]
                is_unsat = not sat
                if was_unsat and not is_unsat:
                    unsat_count -= 1
                elif not was_unsat and is_unsat:
                    unsat_count += 1
                clause_unsat[other_ci] = is_unsat

        return best_assignment

    def _simulated_annealing(rng, max_iters=30000):
        """Simulated annealing for SAT."""
        assignment = [rng.random() < 0.5 for _ in range(n)]

        clause_unsat = np.zeros(m, dtype=bool)
        for ci, clause in enumerate(clauses_0based):
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat

        unsat_count = int(np.sum(clause_unsat))
        if unsat_count == 0:
            return assignment

        best_assignment = assignment[:]
        best_unsat = unsat_count

        T0 = 1.0
        T_end = 0.01
        alpha = (T_end / T0) ** (1.0 / max_iters)

        T = T0
        for step in range(max_iters):
            if time.time() > deadline:
                return best_assignment

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick random variable to flip
            var = rng.randrange(n)
            old_val = assignment[var]
            assignment[var] = not old_val

            # Compute new unsat count delta
            delta = 0
            for other_ci, other_sign in var_clauses[var]:
                other_clause = clauses_0based[other_ci]
                sat = False
                for v, s in other_clause:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                is_unsat = not sat
                was_unsat = clause_unsat[other_ci]
                if was_unsat and not is_unsat:
                    delta -= 1
                elif not was_unsat and is_unsat:
                    delta += 1
                clause_unsat[other_ci] = is_unsat

            new_unsat = unsat_count + delta

            # Accept or reject
            if new_unsat < unsat_count:
                unsat_count = new_unsat
            else:
                # Accept with probability exp(-delta / T)
                if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                    unsat_count = new_unsat
                else:
                    # Reject: undo the flip
                    assignment[var] = old_val
                    for other_ci, other_sign in var_clauses[var]:
                        other_clause = clauses_0based[other_ci]
                        sat = False
                        for v, s in other_clause:
                            if assignment[v] == (s > 0):
                                sat = True
                                break
                        clause_unsat[other_ci] = not sat
                    # Recompute unsat_count properly
                    unsat_count = int(np.sum(clause_unsat))

            T *= alpha

        return best_assignment

    # Try multiple strategies and return the best result
    rng = random.Random(42)

    # Strategy 1: WalkSAT with restarts
    result = _walksat_with_restart(rng, max_restarts=30, steps_per_restart=8000)
    if time.time() < deadline - 5:
        # Verify
        all_sat = True
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if result[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return result

    # Strategy 2: Local search with tabu
    if time.time() < deadline - 10:
        result2 = _local_search_with_tabu(random.Random(123), max_iters=15000)
        all_sat = True
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if result2[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return result2

    # Strategy 3: Simulated annealing
    if time.time() < deadline - 10:
        result3 = _simulated_annealing(random.Random(777), max_iters=25000)
        all_sat = True
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if result3[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return result3

    # Return the best result found
    return result
# EVOLVE-BLOCK-END
