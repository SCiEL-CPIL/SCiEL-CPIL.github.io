# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 325.0  # safety margin vs 330s limit

    # Precompute clause structure: (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    # Variable occurrence index: var -> list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))

    def make_state(seed):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        true_count = [0] * m
        for ci, cv in enumerate(clause_vars):
            cnt = 0
            for vi, pos in cv:
                if assignment[vi] == pos:
                    cnt += 1
            true_count[ci] = cnt
        return rng, assignment, true_count

    def is_solved(true_count):
        # All clauses must have at least one true literal
        for tc in true_count:
            if tc == 0:
                return False
        return True

    def flip(assignment, true_count, var_idx, new_val):
        """Flip variable var_idx to new_val, update true_count incrementally."""
        if assignment[var_idx] == new_val:
            return
        assignment[var_idx] = new_val
        for ci, pos in var_clauses[var_idx]:
            if pos == new_val:
                true_count[ci] += 1
            else:
                true_count[ci] -= 1

    def walk_sat(rng, assignment, true_count, max_steps, p, deadline):
        """Run WalkSAT from current state. Returns (solved, assignment, true_count)."""
        for step in range(max_steps):
            if time.time() >= deadline:
                return False, assignment, true_count

            # Find an unsatisfied clause
            # Use a quick scan; for large m we could maintain a set but scan is fine
            unsat_ci = -1
            for ci in range(m):
                if true_count[ci] == 0:
                    unsat_ci = ci
                    break
            if unsat_ci == -1:
                return True, assignment, true_count

            # Choose a literal to flip
            cv = clause_vars[unsat_ci]
            # Weighted choice: prefer literals that satisfy the clause
            # For a 3-clause, p is probability of choosing a satisfying literal
            # We pick among the 3 literals; if it's currently false, flipping it
            # satisfies the clause.
            # Standard WalkSAT: with prob p pick a literal that makes clause true,
            # else pick randomly.
            if rng.random() < p:
                # Pick a currently-false literal in the clause
                false_lits = [i for i in range(len(cv))
                              if assignment[cv[i][0]] != cv[i][1]]
                if false_lits:
                    idx = rng.choice(false_lits)
                else:
                    idx = rng.randrange(len(cv))
            else:
                idx = rng.randrange(len(cv))

            vi, pos = cv[idx]
            flip(assignment, true_count, vi, not assignment[vi])

        # Final check
        for ci in range(m):
            if true_count[ci] == 0:
                return False, assignment, true_count
        return True, assignment, true_count

    # Try multiple seeds/restarts
    best_assignment = None
    best_true_count = None
    best_unsat = m + 1  # number of unsatisfied clauses for best

    seed_base = 0
    while time.time() < deadline:
        rng, assignment, true_count = make_state(seed_base)

        # Run WalkSAT with adaptive step count
        # Start with a generous step budget; reduce as time runs out
        remaining = deadline - time.time()
        if remaining < 0.5:
            break

        # Step budget proportional to remaining time (rough estimate)
        # Each step is O(occurrences) ~ small; assume ~1000 steps per 0.01s
        max_steps = int(remaining * 100000)
        if max_steps < 1000:
            max_steps = 1000
        if max_steps > 500000:
            max_steps = 500000

        p = 0.55  # tuned for planted 3-SAT

        solved, assignment, true_count = walk_sat(
            rng, assignment, true_count, max_steps, p, deadline
        )

        # Track best partial solution
        unsat_count = sum(1 for tc in true_count if tc == 0)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            best_true_count = true_count[:]

        if solved:
            return assignment

        # If we're getting close to deadline, do a final polish
        if remaining < 2.0:
            break

        seed_base += 1

    # Final polish: if we have a good partial assignment, try to fix remaining
    if best_assignment is not None:
        remaining = deadline - time.time()
        if remaining > 0.1:
            # Use the best partial state as starting point
            rng = random.Random(seed_base + 1000)
            assignment = best_assignment[:]
            true_count = best_true_count[:]
            max_steps = int(remaining * 100000)
            if max_steps < 500:
                max_steps = 500
            solved, assignment, true_count = walk_sat(
                rng, assignment, true_count, max_steps, 0.5, deadline
            )
            if solved:
                return assignment
            # Return best found
            return assignment

    # Fallback: return last assignment
    if best_assignment is not None:
        return best_assignment

    # Absolute fallback
    return [False] * n
# EVOLVE-BLOCK-END
