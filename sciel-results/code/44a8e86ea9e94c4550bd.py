"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using a robust WalkSAT variant with:
    - Deadline guard (safety margin before 330s wall-clock limit)
    - Restarts with increasing noise
    - Heuristic variable selection (min-unsat heuristic + random tie-breaking)
    - Tracking best assignment found
    """
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 320  # 10s safety margin

    # Pre-compute clause structure for speed
    num_clauses = len(clauses)
    # Store clauses as lists of (var_idx, sign) for fast access
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        signs_in_c = [1 if l > 0 else 0 for l in c]
        clause_vars.append(vars_in_c)
        clause_signs.append(signs_in_c)

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(num_clauses):
        for v in clause_vars[ci]:
            var_to_clauses[v].append(ci)

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for ci in range(num_clauses):
            sat = False
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            for i in range(len(cv)):
                if assignment[cv[i]] == (cs[i] == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat_clauses(assignment):
        """Count number of unsatisfied clauses."""
        cnt = 0
        for ci in range(num_clauses):
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            sat = False
            for i in range(len(cv)):
                if assignment[cv[i]] == (cs[i] == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def walk_sat_trial(assignment, rng, noise, max_steps, deadline):
        """
        Run one WalkSAT trial from the given assignment.
        Returns (assignment, solved) where solved indicates if all clauses are satisfied.
        """
        # Track unsatisfied clauses
        unsat_clauses = set()
        for ci in range(num_clauses):
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            sat = False
            for i in range(len(cv)):
                if assignment[cv[i]] == (cs[i] == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.add(ci)

        if not unsat_clauses:
            return assignment, True

        for step in range(max_steps):
            if time.time() >= deadline:
                return assignment, False

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            cv = clause_vars[ci]
            cs = clause_signs[ci]

            # Find which literals in this clause are false (i.e., variable has wrong value)
            false_vars = []
            for i in range(len(cv)):
                if assignment[cv[i]] != (cs[i] == 1):
                    false_vars.append(i)

            if not false_vars:
                # Shouldn't happen if clause is unsat, but safety
                continue

            # Choose which variable to flip
            if rng.random() < noise:
                # Random flip
                idx = rng.choice(false_vars)
            else:
                # Min-unsat heuristic: pick variable whose flip minimizes new unsatisfied clauses
                best_idx = false_vars[0]
                best_score = float('inf')
                for idx in false_vars:
                    v = cv[idx]
                    # Simulate flip
                    score = 0
                    for cj in var_to_clauses[v]:
                        cjv = clause_vars[cj]
                        cjs = clause_signs[cj]
                        sat = False
                        for i in range(len(cjv)):
                            if cjv[i] == v:
                                # This variable is being flipped
                                if assignment[v] != (cjs[i] == 1):
                                    # Currently this literal is false, after flip it becomes true
                                    # So this clause might become satisfied
                                    pass
                                else:
                                    # Currently true, after flip becomes false
                                    # Need to check if other literals satisfy
                                    other_sat = False
                                    for j in range(len(cjv)):
                                        if j != i and assignment[cjv[j]] == (cjs[j] == 1):
                                            other_sat = True
                                            break
                                    if not other_sat:
                                        score += 1
                            else:
                                if assignment[cjv[i]] == (cjs[i] == 1):
                                    sat = True
                                    break
                        if not sat:
                            score += 1
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                idx = best_idx

            v = cv[idx]
            assignment[v] = not assignment[v]

            # Update unsatisfied clauses
            # Clauses that were satisfied and now might not be
            # Clauses that were unsatisfied and now might be
            # For simplicity, recompute affected clauses
            new_unsat = set()
            for ci2 in var_to_clauses[v]:
                cv2 = clause_vars[ci2]
                cs2 = clause_signs[ci2]
                sat = False
                for i in range(len(cv2)):
                    if assignment[cv2[i]] == (cs2[i] == 1):
                        sat = True
                        break
                if not sat:
                    new_unsat.add(ci2)
                else:
                    # Check if it was previously unsat and is now sat - remove from tracking
                    pass

            # Actually, let's just recompute all unsat for correctness (n=50 is small)
            unsat_clauses = set()
            for ci2 in range(num_clauses):
                cv2 = clause_vars[ci2]
                cs2 = clause_signs[ci2]
                sat = False
                for i in range(len(cv2)):
                    if assignment[cv2[i]] == (cs2[i] == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.add(ci2)

            if not unsat_clauses:
                return assignment, True

        return assignment, False

    best_assignment = None
    best_unsat_count = float('inf')

    rng = random.Random(42)

    # Multiple restarts with increasing noise
    max_restarts = 100
    base_noise = 0.2
    max_noise = 0.8
    max_steps_per_trial = 5000

    for restart in range(max_restarts):
        if time.time() >= deadline:
            break

        # Noise increases with restarts
        noise = min(base_noise + restart * 0.01, max_noise)

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Quick check if already satisfied
        if is_satisfied(assignment):
            return assignment

        unsat_count = count_unsat_clauses(assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        result, solved = walk_sat_trial(assignment, rng, noise, max_steps_per_trial, deadline)

        if solved:
            return result

        unsat_count = count_unsat_clauses(result)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = result[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
