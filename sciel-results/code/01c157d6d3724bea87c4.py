"""Planted 3-SAT solver: Efficient WalkSAT with min-conflicts + phase saving + local repair."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: 330s wall-clock limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0

    # Preprocess: convert to 0-based variable indices and sign (+1/-1)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)

    # Build variable -> list of (clause_idx, sign) for fast updates
    var_occurrences = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for (v, s) in cv:
            var_occurrences[v].append((ci, s))

    rng = random.Random(42)

    def init_state(assignment):
        """Initialize clause_sat_count and return unsat_count."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            cv = clause_vars[ci]
            cnt = 0
            for v, s in cv:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            clause_sat_count[ci] = cnt
        unsat_count = int(np.sum(clause_sat_count == 0))
        return clause_sat_count, unsat_count

    def full_check(assignment):
        """Check if all clauses are satisfied."""
        for ci in range(m):
            cv = clause_vars[ci]
            sat = False
            for v, s in cv:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def flip_var(assignment, v, clause_sat_count):
        """Flip variable v and update clause_sat_count. Returns delta_unsat."""
        old_val = assignment[v]
        delta_unsat = 0
        for (aci, s) in var_occurrences[v]:
            was_sat = (old_val and s == 1) or (not old_val and s == -1)
            is_sat = (not old_val and s == 1) or (old_val and s == -1)
            if was_sat and not is_sat:
                clause_sat_count[aci] -= 1
                if clause_sat_count[aci] == 0:
                    delta_unsat += 1
            elif not was_sat and is_sat:
                clause_sat_count[aci] += 1
                if clause_sat_count[aci] == 1:
                    delta_unsat -= 1
        assignment[v] = not old_val
        return delta_unsat

    def unflip_var(assignment, v, clause_sat_count):
        """Unflip variable v and update clause_sat_count. Returns delta_unsat."""
        old_val = assignment[v]
        delta_unsat = 0
        for (aci, s) in var_occurrences[v]:
            was_sat = (old_val and s == 1) or (not old_val and s == -1)
            is_sat = (not old_val and s == 1) or (old_val and s == -1)
            if was_sat and not is_sat:
                clause_sat_count[aci] -= 1
                if clause_sat_count[aci] == 0:
                    delta_unsat += 1
            elif not was_sat and is_sat:
                clause_sat_count[aci] += 1
                if clause_sat_count[aci] == 1:
                    delta_unsat -= 1
        assignment[v] = not old_val
        return delta_unsat

    def walksat(assignment, clause_sat_count, unsat_count, max_steps):
        """
        WalkSAT with adaptive p and min-conflicts variable selection.
        Returns (solved, assignment, clause_sat_count, unsat_count).
        """
        p = 0.5
        min_p = 0.25
        max_p = 0.75

        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                return False, assignment, clause_sat_count, unsat_count

            if unsat_count == 0:
                return True, assignment, clause_sat_count, 0

            # Pick an unsatisfied clause uniformly at random
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return True, assignment, clause_sat_count, 0
            ci = int(rng.choice(unsat_indices.tolist()))
            cv = clause_vars[ci]

            # Select variable to flip
            if rng.random() < p:
                # Random flip within the clause
                pi = rng.randint(0, len(cv) - 1)
                v = cv[pi][0]
            else:
                # Min-conflicts: pick variable whose flip reduces unsat count the most
                best_v = None
                best_delta = float('inf')
                for pi in range(len(cv)):
                    v = cv[pi][0]
                    old_val = assignment[v]
                    delta = 0
                    for (aci, s) in var_occurrences[v]:
                        was_sat = (old_val and s == 1) or (not old_val and s == -1)
                        is_sat = (not old_val and s == 1) or (old_val and s == -1)
                        if was_sat and not is_sat:
                            if clause_sat_count[aci] == 1:
                                delta += 1
                        elif not was_sat and is_sat:
                            if clause_sat_count[aci] == 0:
                                delta -= 1
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                v = best_v

            # Compute delta_unsat for flipping v
            old_val = assignment[v]
            delta_unsat = 0
            for (aci, s) in var_occurrences[v]:
                was_sat = (old_val and s == 1) or (not old_val and s == -1)
                is_sat = (not old_val and s == 1) or (old_val and s == -1)
                if was_sat and not is_sat:
                    if clause_sat_count[aci] == 1:
                        delta_unsat += 1
                elif not was_sat and is_sat:
                    if clause_sat_count[aci] == 0:
                        delta_unsat -= 1

            # Accept with probability based on delta
            if delta_unsat <= 0:
                accept = True
            else:
                accept = rng.random() < 0.3

            if accept:
                # Perform the flip
                flip_var(assignment, v, clause_sat_count)
                unsat_count += delta_unsat

                # Adapt p
                if delta_unsat < 0:
                    p = min(max_p, p + 0.01)
                elif delta_unsat > 0:
                    p = max(min_p, p - 0.02)

        return False, assignment, clause_sat_count, unsat_count

    # Main solver: WalkSAT with restarts and phase saving
    best_assignment = None
    best_unsat_count = m + 1
    phase = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

    max_restarts = 50000
    base_steps = 2000

    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        # Initialize: with some probability use phase, otherwise random
        if rng.random() < 0.3:
            assignment = phase.copy()
        else:
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        clause_sat_count, unsat_count = init_state(assignment)

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
            # Update phase with current assignment
            phase[:] = assignment

        # Run WalkSAT
        solved, assignment, clause_sat_count, unsat_count = walksat(
            assignment, clause_sat_count, unsat_count, base_steps
        )

        if solved:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
            # Update phase
            phase[:] = assignment

    # If WalkSAT didn't solve, try local repair from best assignment
    if best_assignment is not None:
        if full_check(best_assignment):
            return best_assignment.tolist()

        # Simple local repair: for each unsatisfied clause, try flipping one of its literals
        assignment = best_assignment.copy()
        clause_sat_count, unsat_count = init_state(assignment)

        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            if unsat_count == 0:
                break
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                break
            ci = int(unsat_indices[0])
            cv = clause_vars[ci]
            # Try each literal in the clause, pick the one with smallest delta
            best_v = None
            best_delta = float('inf')
            for v, s in cv:
                old_val = assignment[v]
                delta = 0
                for (aci, s2) in var_occurrences[v]:
                    was_sat = (old_val and s2 == 1) or (not old_val and s2 == -1)
                    is_sat = (not old_val and s2 == 1) or (old_val and s2 == -1)
                    if was_sat and not is_sat:
                        if clause_sat_count[aci] == 1:
                            delta += 1
                    elif not was_sat and is_sat:
                        if clause_sat_count[aci] == 0:
                            delta -= 1
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
            if best_v is not None and best_delta <= 0:
                flip_var(assignment, best_v, clause_sat_count)
                unsat_count += best_delta

        if unsat_count == 0:
            return assignment.tolist()

        if int(np.sum(clause_sat_count == 0)) < best_unsat_count:
            best_unsat_count = int(np.sum(clause_sat_count == 0))
            best_assignment = assignment.copy()

        return best_assignment.tolist()

    return [False] * n


# EVOLVE-BLOCK-END
