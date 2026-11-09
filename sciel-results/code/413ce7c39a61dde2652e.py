"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0

    # Pre-process: convert to 0-indexed (var, is_positive)
    pre_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            if lit > 0:
                pc.append((lit - 1, True))
            else:
                pc.append((-lit - 1, False))
        pre_clauses.append(pc)
    m = len(pre_clauses)

    # Track which clauses are unsatisfied
    # For each clause, store the literals as (var, is_pos)
    
    def is_satisfied(c, assignment):
        for var, is_pos in c:
            if assignment[var] == is_pos:
                return True
        return False

    def count_unsat(assignment):
        return sum(1 for c in pre_clauses if not is_satisfied(c, assignment))

    best_assignment = None
    best_unsat = m

    rng = random.Random(99999)

    max_restarts = 200
    max_steps_per_restart = 15000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break

            # Find an unsatisfied clause
            # Randomly sample clauses until we find an unsat one
            unsat_clause = None
            for _ in range(50):
                ci = rng.randrange(m)
                if not is_satisfied(pre_clauses[ci], assignment):
                    unsat_clause = pre_clauses[ci]
                    break

            if unsat_clause is None:
                # Might be all satisfied, verify
                if count_unsat(assignment) == 0:
                    return assignment
                # Fallback: pick a random clause
                unsat_clause = pre_clauses[rng.randrange(m)]

            # Flip a literal from the unsatisfied clause
            var, is_pos = rng.choice(unsat_clause)
            assignment[var] = not assignment[var]

            steps += 1
            # Check satisfaction periodically
            if steps % 10 == 0:
                unsat_count = count_unsat(assignment)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
