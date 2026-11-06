"""Stronger planted 3-SAT solver: vectorized adaptive WalkSAT with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _build_structures(n, clauses):
    """Precompute literal arrays for fast vectorized evaluation."""
    # literals as (var_index, polarity) where polarity is 1 if positive, -1 if negative
    # We store per-clause arrays of variable indices and signed literals.
    m = len(clauses)
    if m == 0:
        return None, None
    # Use int8 for literals (1-based signed), int16 for var indices
    lit_arr = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        lit_arr[i, 0] = c[0]
        lit_arr[i, 1] = c[1]
        lit_arr[i, 2] = c[2]
    var_arr = np.abs(lit_arr) - 1  # 0-based variable index
    return lit_arr, var_arr


def _eval_clauses(assignment, lit_arr, var_arr):
    """Return boolean array: is each clause satisfied?"""
    # assignment[var] is bool; we need (assignment[var] == (lit > 0))
    # lit > 0 means the literal is satisfied when assignment[var] is True
    # So satisfied_literal = assignment[var] if lit>0 else (not assignment[var])
    # = (assignment[var] == (lit > 0))
    # Vectorized: get assignment values for each literal's variable
    a = assignment[var_arr]  # shape (m, 3), bool
    pos = lit_arr > 0        # shape (m, 3), bool
    sat_literal = a == pos   # shape (m, 3)
    return sat_literal.any(axis=1)  # shape (m,)


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Trivial: no clauses -> all False works
    if m == 0:
        return [False] * n

    start_time = time.time()
    DEADLINE = 290.0  # safety margin before 300s kill

    lit_arr, var_arr = _build_structures(n, clauses)

    # Precompute for each variable, the list of (clause_index, position) where it appears
    # This helps for efficient flip cost, but for speed we'll use a simpler approach.

    # Build a per-variable occurrence list for unit-clause bias
    var_occurrence = [[] for _ in range(n)]
    for i in range(m):
        for pos in range(3):
            v = var_arr[i, pos]
            var_occurrence[v].append((i, pos))

    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(12345)

    restart = 0
    while time.time() - start_time < DEADLINE:
        restart += 1

        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        # Evaluate
        sat_mask = _eval_clauses(assignment, lit_arr, var_arr)
        unsat_count = int((~sat_mask).sum())

        if unsat_count == 0:
            return assignment.tolist()

        # Track best
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # Adaptive WalkSAT with phase-change schedule
        # p starts low, increases with time in this restart
        max_steps = 20000
        p_min = 0.2
        p_max = 0.95

        # Precompute unsat clause indices
        unsat_indices = np.where(~sat_mask)[0]

        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Adaptive p: ramp up based on step and unsat_count
            # Use a schedule that increases p over the restart
            progress = step / max_steps
            p = p_min + (p_max - p_min) * progress

            # Pick an unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause_lits = lit_arr[ci]
            clause_vars = var_arr[ci]

            # Check for unit clause: exactly one literal is false, and flipping it satisfies
            # Actually, for the chosen unsat clause, all 3 literals are currently false.
            # We want to pick a literal to flip. Use a heuristic:
            # Count how many other clauses each literal's variable appears in,
            # and prefer flipping a variable that is "critical" for many clauses.
            # Simpler: with probability p, pick randomly; else pick the literal
            # whose variable appears in the most unsatisfied clauses.

            if rng.random() < p:
                # Random literal from the clause
                pos = rng.randrange(3)
            else:
                # Heuristic: pick the literal whose variable is in the most unsat clauses
                # Count unsat clauses per variable for the 3 candidates
                best_pos = 0
                best_score = -1
                for pos in range(3):
                    v = clause_vars[pos]
                    # Count how many unsat clauses contain this variable
                    # This is expensive; approximate by checking a sample or using precomputed
                    # For speed, just use the literal that is "most constrained"
                    # We'll use a cheaper proxy: the variable with the most total occurrences
                    score = len(var_occurrence[v])
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                pos = best_pos

            v = clause_vars[pos]
            assignment[v] = not assignment[v]

            # Update: re-evaluate only clauses containing variable v
            # But for simplicity and correctness, re-evaluate all clauses (vectorized)
            sat_mask = _eval_clauses(assignment, lit_arr, var_arr)
            unsat_count = int((~sat_mask).sum())
            unsat_indices = np.where(~sat_mask)[0]

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

        # End of restart
        if best_assignment is not None and best_unsat == 0:
            return best_assignment.tolist()

    # Deadline reached: return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
