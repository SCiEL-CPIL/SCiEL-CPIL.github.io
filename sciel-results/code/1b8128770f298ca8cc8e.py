"""Synthetic planted 3-SAT solver (150 variables)."""
# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # safety margin before 300s limit

    # Precompute variable occurrences: var_occ[v] = list of (clause_idx, sign)
    # where sign is +1 if literal is positive, -1 if negative.
    var_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    def random_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_clause_counts(assignment):
        """Return array of satisfied literal counts per clause (0..3)."""
        counts = [0] * m
        for ci, clause in enumerate(clauses):
            c = 0
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    c += 1
            counts[ci] = c
        return counts

    def flip_variable(v, assignment, counts):
        """Flip variable v, update counts incrementally. Return (delta_unsat, best_unsat_count)."""
        new_val = not assignment[v]
        delta_unsat = 0
        for ci, sign in var_occ[v]:
            old_sat = (assignment[v] == (sign == 1))
            new_sat = (new_val == (sign == 1))
            if old_sat and not new_sat:
                # This literal was satisfying, now not
                counts[ci] -= 1
                if counts[ci] == 0:
                    delta_unsat += 1
            elif not old_sat and new_sat:
                # This literal was not satisfying, now is
                counts[ci] += 1
                if counts[ci] == 1:
                    delta_unsat -= 1
        assignment[v] = new_val
        return delta_unsat

    def walksat(assignment, counts, rng, steps, break_threshold=1):
        """
        WalkSAT with break-count minimization heuristic.
        Returns True if solved.
        """
        for _ in range(steps):
            if time.time() > deadline:
                return False
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            clause = clauses[ci]

            # Score each literal in the clause by break count
            best_var = -1
            best_break = float('inf')
            best_keep = -1
            for lit in clause:
                v = abs(lit) - 1
                # Count how many currently-satisfied literals in var_occ[v] would break
                break_count = 0
                for oci, osign in var_occ[v]:
                    if oci == ci:
                        continue
                    if counts[oci] == 1:
                        # This clause has exactly one satisfied literal; check if it's this variable
                        # Actually we need to check if flipping v would make this clause unsat
                        # The clause oci has count 1, meaning exactly one literal is satisfied.
                        # If that satisfied literal is for variable v, flipping v breaks it.
                        # We need to verify: is the satisfied literal in clause oci the one for v?
                        # We can check: for the literal of v in clause oci, is it currently satisfied?
                        # We stored (oci, osign) where osign is the sign of v in clause oci.
                        # v is currently assignment[v]. The literal is satisfied if assignment[v] == (osign==1).
                        if assignment[v] == (osign == 1):
                            break_count += 1
                keep_count = 1  # flipping v will satisfy the literal in clause ci
                # Use weighted score: minimize break, maximize keep
                score = break_count * 10 - keep_count
                if score < best_break or (score == best_break and rng.random() < 0.5):
                    best_break = score
                    best_var = v

            if best_var >= 0:
                flip_variable(best_var, assignment, counts)

            # With some probability, do a random flip (WalkSAT style)
            if rng.random() < 0.05:
                v = rng.randrange(n)
                flip_variable(v, assignment, counts)

        # Final check
        return all(counts[ci] > 0 for ci in range(m))

    def walksat_random(assignment, counts, rng, steps):
        """Pure random WalkSAT (pick random literal from random unsat clause)."""
        for _ in range(steps):
            if time.time() > deadline:
                return False
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True
            ci = rng.choice(unsat)
            clause = clauses[ci]
            lit = rng.choice(clause)
            v = abs(lit) - 1
            flip_variable(v, assignment, counts)
        return all(counts[ci] > 0 for ci in range(m))

    def min_conflicts(assignment, counts, rng, steps):
        """
        Min-conflicts heuristic: for each unsat clause, pick a variable to flip
        that minimizes the number of newly-unsatisfied clauses.
        """
        for _ in range(steps):
            if time.time() > deadline:
                return False
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True

            # Pick random unsat clause
            ci = rng.choice(unsat)
            clause = clauses[ci]

            # For each variable in the clause, compute conflict count if we flip it
            best_var = -1
            best_conflict = float('inf')
            for lit in clause:
                v = abs(lit) - 1
                # Count how many clauses containing v would become unsatisfied
                conflict = 0
                for oci, osign in var_occ[v]:
                    if counts[oci] == 1 and assignment[v] == (osign == 1):
                        conflict += 1
                if conflict < best_conflict:
                    best_conflict = conflict
                    best_var = v

            if best_var >= 0:
                flip_variable(best_var, assignment, counts)

        return all(counts[ci] > 0 for ci in range(m))

    # Main search loop with multiple restarts and alternating strategies
    best_assignment = None
    best_unsat = m + 1
    seed = 0
    restart_count = 0
    steps_per_restart = 5000

    while time.time() < deadline:
        rng = random.Random(seed)
        assignment = random_assignment(rng)
        counts = compute_clause_counts(assignment)
        cur_unsat = sum(1 for c in counts if c == 0)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        if cur_unsat == 0:
            return assignment

        # Try different strategies
        strategy = restart_count % 3
        if strategy == 0:
            solved = walksat(assignment, counts, rng, steps_per_restart)
        elif strategy == 1:
            solved = walksat_random(assignment, counts, rng, steps_per_restart)
        else:
            solved = min_conflicts(assignment, counts, rng, steps_per_restart)

        if solved:
            return assignment

        # Update best
        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        restart_count += 1
        seed += 1

        # Gradually increase steps per restart to allow deeper search
        if restart_count % 10 == 0:
            steps_per_restart = min(steps_per_restart * 2, 50000)

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
