"""Synthetic planted 3-SAT solver (150 variables) with multiple strategies."""
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s kill

    # Precompute clause literals as (var_idx, polarity) for faster access
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    num_clauses = len(clauses_opt)

    def is_satisfied(assignment, clause):
        for var_idx, polarity in clause:
            if assignment[var_idx] == polarity:
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for clause in clauses_opt:
            sat = False
            for var_idx, polarity in clause:
                if assignment[var_idx] == polarity:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walksat(assignment, rng, max_steps=50000):
        """WalkSAT with adaptive noise parameter."""
        for step in range(max_steps):
            # Find all unsatisfied clauses
            unsat_clauses = []
            for clause in clauses_opt:
                sat = False
                for var_idx, polarity in clause:
                    if assignment[var_idx] == polarity:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)

            if not unsat_clauses:
                return True

            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)

            # Count how many literals in this clause would be satisfied if we flip each variable
            best_var = None
            best_score = -1
            best_ties = []

            for var_idx, polarity in clause:
                # If we flip this variable, how many other clauses become satisfied/unsatisfied?
                # Simple heuristic: prefer flipping variables that appear in many unsat clauses
                score = 0
                for c in unsat_clauses:
                    for v, p in c:
                        if v == var_idx:
                            score += 1
                            break
                if score > best_score:
                    best_score = score
                    best_ties = [var_idx]
                elif score == best_score:
                    best_ties.append(var_idx)

            if best_ties:
                var_idx = rng.choice(best_ties)
                assignment[var_idx] = not assignment[var_idx]
            else:
                # Fallback: random flip
                var_idx, polarity = rng.choice(clause)
                assignment[var_idx] = not assignment[var_idx]

            # Check deadline
            if time.time() > deadline:
                return False

        return False

    def local_search(assignment, rng, max_steps=30000):
        """Local search focusing on minimizing unsatisfied clauses."""
        current_unsat = count_unsat(assignment)
        best_assignment = assignment[:]
        best_unsat = current_unsat

        for step in range(max_steps):
            if current_unsat == 0:
                return True, assignment

            # Find unsatisfied clauses
            unsat_clauses = []
            for clause in clauses_opt:
                sat = False
                for var_idx, polarity in clause:
                    if assignment[var_idx] == polarity:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)

            if not unsat_clauses:
                return True, assignment

            # Try flipping each variable in a random unsat clause
            clause = rng.choice(unsat_clauses)
            candidates = []
            for var_idx, polarity in clause:
                new_assignment = assignment[:]
                new_assignment[var_idx] = not new_assignment[var_idx]
                new_unsat = count_unsat(new_assignment)
                candidates.append((new_unsat, var_idx))

            # Pick the flip that minimizes unsat count, with some randomness
            candidates.sort(key=lambda x: x[0])
            # Sometimes accept a worse move to escape local minima
            if rng.random() < 0.1 and len(candidates) > 1:
                # Accept a random candidate
                _, var_idx = rng.choice(candidates)
            else:
                # Accept the best or second best
                idx = 0 if rng.random() < 0.8 else min(1, len(candidates) - 1)
                _, var_idx = candidates[idx]

            assignment[var_idx] = not assignment[var_idx]
            current_unsat = count_unsat(assignment)

            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]

            if time.time() > deadline:
                break

        return best_unsat == 0, best_assignment

    def dls(assignment, rng, max_depth=20):
        """Depth-limited search with clause learning."""
        # Simple DLS: try to satisfy all clauses by flipping variables
        for _ in range(max_depth):
            unsat_clauses = []
            for clause in clauses_opt:
                sat = False
                for var_idx, polarity in clause:
                    if assignment[var_idx] == polarity:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)

            if not unsat_clauses:
                return True

            # Pick the most constrained clause (most variables already set to wrong value)
            # For 3-SAT, just pick randomly
            clause = rng.choice(unsat_clauses)

            # Try to satisfy this clause by flipping one of its variables
            for var_idx, polarity in clause:
                if not assignment[var_idx] == polarity:
                    assignment[var_idx] = not assignment[var_idx]
                    break

            if time.time() > deadline:
                return False

        return False

    # Strategy: Try multiple approaches with restarts
    best_assignment = None
    best_unsat = float('inf')

    # First, try a random assignment and see if it works
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    unsat = count_unsat(assignment)
    if unsat == 0:
        return assignment
    best_unsat = unsat
    best_assignment = assignment[:]

    # Try WalkSAT with different random seeds
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        seed = restart * 1000 + 42
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try WalkSAT
        success = walksat(assignment, rng, max_steps=50000)
        if success:
            return assignment

        # Track best
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]

        # Try local search from current state
        success2, best_from_ls = local_search(assignment, rng, max_steps=20000)
        if success2:
            return best_from_ls
        unsat2 = count_unsat(best_from_ls)
        if unsat2 < best_unsat:
            best_unsat = unsat2
            best_assignment = best_from_ls[:]

    # If we still haven't found a solution, try more aggressive search
    if time.time() < deadline - 10:
        # Try DLS with multiple restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            rng = random.Random(restart * 777 + 13)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            success = dls(assignment, rng, max_depth=50)
            if success:
                return assignment
            unsat = count_unsat(assignment)
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = assignment[:]

    return best_assignment if best_assignment else [False] * n
