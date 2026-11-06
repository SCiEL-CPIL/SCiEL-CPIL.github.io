# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s limit

    # Preprocess: convert clauses to 0-based signed literals
    # Store as list of lists of (var_idx, sign) where sign is +1 or -1
    clause_vars = []
    for clause in clauses:
        cl = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cl.append((var, sign))
        clause_vars.append(cl)

    m = len(clause_vars)

    # Build clause-to-variable incidence for faster updates
    var_to_clauses = [[] for _ in range(n)]
    for i, cl in enumerate(clause_vars):
        for var, sign in cl:
            var_to_clauses[var].append(i)

    # Convert to numpy arrays for vectorized satisfaction checking
    # clause_vars is a list of lists of (var, sign)
    # We'll use a different approach: maintain clause_sat_counts

    def random_assignment(rng):
        return np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

    def compute_clause_sat(assignment):
        """Compute number of satisfied literals per clause."""
        sat_counts = np.zeros(m, dtype=np.int32)
        for i, cl in enumerate(clause_vars):
            count = 0
            for var, sign in cl:
                val = assignment[var]
                if (sign == 1 and val) or (sign == -1 and not val):
                    count += 1
            sat_counts[i] = count
        return sat_counts

    def is_satisfied(assignment, sat_counts):
        return np.all(sat_counts > 0)

    def walksat_phase(assignment, sat_counts, rng, max_steps):
        """WalkSAT with weighted literal selection."""
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, sat_counts, False
            if np.all(sat_counts > 0):
                return assignment, sat_counts, True

            # Find unsatisfied clauses
            unsat_mask = sat_counts == 0
            unsat_idx = np.where(unsat_mask)[0]
            if len(unsat_idx) == 0:
                return assignment, sat_counts, True

            # Weighted selection of unsatisfied clause (prefer shorter)
            # Use inverse length as weight
            lengths = np.array([len(clause_vars[i]) for i in unsat_idx], dtype=np.float64)
            weights = 1.0 / lengths
            weights /= weights.sum()
            chosen_clause = unsat_idx[rng.choice(len(unsat_idx), p=weights)]

            cl = clause_vars[chosen_clause]

            # Compute flip scores: for each literal, how many other clauses would be affected
            # Use a simple heuristic: prefer literals that appear in many other clauses
            # and that, when flipped, would satisfy more clauses
            best_var = -1
            best_score = -1
            for var, sign in cl:
                # Current assignment of this variable
                current_val = assignment[var]
                # If we flip this variable, this clause becomes satisfied (since it's unsat)
                # Score: number of other clauses containing this variable that are currently satisfied
                # and would become unsatisfied, vs those that are unsatisfied and would become satisfied
                satisfied_would_lose = 0
                unsat_would_gain = 0
                for ci in var_to_clauses[var]:
                    if ci == chosen_clause:
                        continue
                    # Check if this clause is currently satisfied
                    if sat_counts[ci] > 0:
                        # Check if flipping var would make this clause unsatisfied
                        # i.e., this clause has exactly 1 satisfied literal and it's this var
                        if sat_counts[ci] == 1:
                            # Check if the satisfied literal in this clause is this var
                            for v2, s2 in clause_vars[ci]:
                                if v2 == var:
                                    # This clause is satisfied only by this variable
                                    # Flipping would make it unsatisfied
                                    satisfied_would_lose += 1
                                    break
                    else:
                        # Clause is unsatisfied; flipping var might satisfy it
                        # (it would satisfy if this is the only unsatisfied... no, 3-SAT clause needs 1)
                        # Actually if clause is unsat, all literals are false. Flipping var makes one true.
                        unsat_would_gain += 1
                score = unsat_would_gain * 2 - satisfied_would_lose
                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var == -1:
                # Fallback: random literal
                var, sign = rng.choice(cl)
            else:
                # Find the sign for this variable in the clause
                sign = 1
                for v, s in cl:
                    if v == best_var:
                        sign = s
                        break
                var = best_var

            # Flip the variable
            new_val = not assignment[var]
            assignment[var] = new_val

            # Update sat_counts for all clauses containing this variable
            for ci in var_to_clauses[var]:
                cl_i = clause_vars[ci]
                old_count = sat_counts[ci]
                # Recompute this clause's sat count
                new_count = 0
                for v, s in cl_i:
                    val = assignment[v]
                    if (s == 1 and val) or (s == -1 and not val):
                        new_count += 1
                sat_counts[ci] = new_count

        return assignment, sat_counts, False

    def local_search_phase(assignment, sat_counts, rng, max_steps):
        """Steepest-descent local search with random restarts within the phase."""
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, sat_counts, False
            if np.all(sat_counts > 0):
                return assignment, sat_counts, True

            # Try all single flips and pick the best one
            best_var = -1
            best_delta = -1  # We want to minimize unsatisfied clauses
            current_unsat = np.sum(sat_counts == 0)

            # Sample a subset of variables for efficiency
            sample_size = min(n, 50)
            sample_vars = rng.sample(range(n), sample_size)

            for var in sample_vars:
                # Simulate flip
                old_val = assignment[var]
                assignment[var] = not old_val

                # Compute change in unsat count
                new_unsat = current_unsat
                for ci in var_to_clauses[var]:
                    cl_i = clause_vars[ci]
                    old_sat = sat_counts[ci] > 0
                    # Recompute sat count for this clause
                    count = 0
                    for v, s in cl_i:
                        val = assignment[v]
                        if (s == 1 and val) or (s == -1 and not val):
                            count += 1
                    new_sat = count > 0
                    if old_sat and not new_sat:
                        new_unsat += 1
                    elif not old_sat and new_sat:
                        new_unsat -= 1

                delta = current_unsat - new_unsat  # positive is good

                # Undo flip
                assignment[var] = old_val

                if delta > best_delta:
                    best_delta = delta
                    best_var = var

            if best_var == -1 or best_delta <= 0:
                # No improving move; random kick
                var = rng.randint(0, n - 1)
                assignment[var] = not assignment[var]
                for ci in var_to_clauses[var]:
                    cl_i = clause_vars[ci]
                    count = 0
                    for v, s in cl_i:
                        val = assignment[v]
                        if (s == 1 and val) or (s == -1 and not val):
                            count += 1
                    sat_counts[ci] = count
            else:
                # Apply best flip
                assignment[best_var] = not assignment[best_var]
                for ci in var_to_clauses[best_var]:
                    cl_i = clause_vars[ci]
                    count = 0
                    for v, s in cl_i:
                        val = assignment[v]
                        if (s == 1 and val) or (s == -1 and not val):
                            count += 1
                    sat_counts[ci] = count

        return assignment, sat_counts, False

    # Main search loop with multiple strategies
    rng = random.Random(42)

    best_assignment = None
    best_unsat = float('inf')

    # Phase 1: Try WalkSAT with multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        assignment = random_assignment(rng)
        sat_counts = compute_clause_sat(assignment)
        unsat = np.sum(sat_counts == 0)

        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment.copy()

        if unsat == 0:
            return assignment.tolist()

        # WalkSAT phase
        max_steps = max(1000, n * 20)
        assignment, sat_counts, solved = walksat_phase(assignment, sat_counts, rng, max_steps)

        if solved:
            return assignment.tolist()

        unsat = np.sum(sat_counts == 0)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment.copy()

        # Phase 2: Local search polish
        if unsat > 0 and unsat <= 20:
            max_steps_ls = max(500, n * 5)
            assignment, sat_counts, solved = local_search_phase(assignment, sat_counts, rng, max_steps_ls)
            if solved:
                return assignment.tolist()
            unsat = np.sum(sat_counts == 0)
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = assignment.copy()

    # If we have a best assignment, try to polish it
    if best_assignment is not None and best_unsat > 0:
        assignment = best_assignment.copy()
        sat_counts = compute_clause_sat(assignment)
        # Try local search on the best found
        max_steps_ls = max(1000, n * 10)
        assignment, sat_counts, solved = local_search_phase(assignment, sat_counts, rng, max_steps_ls)
        if solved:
            return assignment.tolist()

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()

    # Fallback: random assignment
    assignment = random_assignment(rng)
    return assignment.tolist()

# EVOLVE-BLOCK-END
