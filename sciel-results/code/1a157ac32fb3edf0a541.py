# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    rng = random.Random(0)

    # Deadline guard: leave a safety margin before the 330s wall-clock kill.
    deadline = time.time() + 280.0

    def satisfied(assignment, clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)

    def all_satisfied(assignment):
        for c in clauses:
            if not any(assignment[abs(l) - 1] == (l > 0) for l in c):
                return False
        return True

    def run_walksat(assignment, steps, noise):
        """One WalkSAT run with a fixed step budget and noise parameter."""
        for _ in range(steps):
            # Find unsatisfied clauses.
            unsat = []
            for c in clauses:
                if not any(assignment[abs(l) - 1] == (l > 0) for l in c):
                    unsat.append(c)
            if not unsat:
                return True
            clause = rng.choice(unsat)
            # Count how many unsatisfied clauses each literal in this clause
            # would satisfy if flipped (i.e., how many unsat clauses contain it).
            # Pick the variable with the highest count (ties broken randomly);
            # with probability `noise`, flip a random variable in the clause.
            if rng.random() < noise:
                literal = rng.choice(clause)
            else:
                # Compute for each literal in the clause how many unsatisfied
                # clauses it appears in (weight).
                best_literal = None
                best_count = -1
                tie_breaker = []
                for l in clause:
                    idx = abs(l) - 1
                    count = 0
                    for uc in unsat:
                        for ul in uc:
                            if abs(ul) - 1 == idx:
                                count += 1
                                break
                    if count > best_count:
                        best_count = count
                        best_literal = l
                        tie_breaker = [l]
                    elif count == best_count:
                        tie_breaker.append(l)
                if tie_breaker:
                    literal = rng.choice(tie_breaker)
                else:
                    literal = rng.choice(clause)
            idx = abs(literal) - 1
            assignment[idx] = not assignment[idx]
        return all_satisfied(assignment)

    # Initial assignment: random.
    base_assignment = [rng.random() < 0.5 for _ in range(n)]

    # Try a sequence of restarts with increasing budgets and varying noise.
    # The idea: short quick tries first, then longer ones if time permits.
    budgets = [2000, 5000, 10000, 20000, 40000, 80000, 150000, 300000]
    noises = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # First, try a few short runs with different noise levels.
    for i in range(min(len(budgets), 4)):
        if time.time() > deadline:
            break
        assignment = base_assignment[:]
        steps = budgets[i]
        noise = noises[i]
        if run_walksat(assignment, steps, noise):
            return assignment

    # If not solved yet, do longer runs with a good noise level.
    # Use noise ~0.2-0.3 which is often effective for planted 3-SAT.
    for i in range(4, len(budgets)):
        if time.time() > deadline:
            break
        assignment = base_assignment[:]
        steps = budgets[i]
        noise = noises[i]
        if run_walksat(assignment, steps, noise):
            return assignment

    # Final: one more long run with moderate noise.
    if time.time() < deadline:
        assignment = base_assignment[:]
        steps = 500000
        noise = 0.25
        if run_walksat(assignment, steps, noise):
            return assignment

    # Return the last assignment (best effort).
    return assignment
# EVOLVE-BLOCK-END
