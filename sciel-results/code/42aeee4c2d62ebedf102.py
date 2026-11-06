"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0

    # Preprocess: convert to 0-based, store as (var_idx, sign) where sign is 1 for positive, 0 for negative
    clauses_0based = []
    for clause in clauses:
        c = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            c.append((var_idx, sign))
        clauses_0based.append(c)

    num_clauses = len(clauses_0based)

    # Precompute which clauses contain each variable
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses_0based):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(i)

    def count_unsat(assignment):
        count = 0
        for clause in clauses_0based:
            sat = False
            for var_idx, sign in clause:
                if assignment[var_idx] == sign:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    best_assignment = None
    best_unsat = num_clauses + 1

    def walksat(assignment, max_steps, rng, p_walk=0.55):
        nonlocal best_assignment, best_unsat

        for step in range(max_steps):
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_indices = []
            for i, clause in enumerate(clauses_0based):
                sat = False
                for var_idx, sign in clause:
                    if assignment[var_idx] == sign:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(i)

            if not unsat_indices:
                best_unsat = 0
                best_assignment = assignment[:]
                return assignment

            unsat_count = len(unsat_indices)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clauses_0based[clause_idx]

            # Compute break count for each literal in the clause
            best_var = -1
            best_break = float('inf')

            for var_idx, sign in clause:
                break_count = 0
                for ci in var_to_clauses[var_idx]:
                    c = clauses_0based[ci]
                    # Check if currently satisfied
                    sat = False
                    for v, s in c:
                        if assignment[v] == s:
                            sat = True
                            break
                    if sat:
                        # Check if still satisfied after flipping var_idx
                        still_sat = False
                        for v, s in c:
                            if v == var_idx:
                                new_val = 1 - assignment[v]
                                if new_val == s:
                                    still_sat = True
                                    break
                            else:
                                if assignment[v] == s:
                                    still_sat = True
                                    break
                        if not still_sat:
                            break_count += 1

                if break_count < best_break:
                    best_break = break_count
                    best_var = var_idx

            # Choose variable to flip
            if rng.random() < p_walk:
                var_idx, sign = clause[rng.randrange(len(clause))]
            else:
                var_idx = best_var
                for v, s in clause:
                    if v == var_idx:
                        sign = s
                        break

            assignment[var_idx] = sign

        return assignment

    rng = random.Random(0)
    steps_per_restart = 5000
    num_restarts = 0

    while time.time() < deadline - 10:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        p_walk = 0.5 + rng.random() * 0.2
        walksat(assignment, steps_per_restart, rng, p_walk)

        num_restarts += 1
        if best_unsat == 0:
            break
        if num_restarts % 5 == 0:
            steps_per_restart = max(1000, steps_per_restart // 2)

    if best_assignment is not None:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
