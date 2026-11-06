"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # WalkSAT with adaptive clause weighting, min-conflicts style tie-breaking,
    # periodic restarts, and a wall-clock deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Precompute literal -> variable index and sign for speed
    # We'll keep assignment as a list of booleans (True = positive)
    # and use 0-based indexing internally.

    deadline = time.time() + 280.0  # 20s safety margin before 300s kill

    # Helper: check if a clause is satisfied
    def clause_satisfied(clause, assignment):
        for l in clause:
            v = abs(l) - 1
            if assignment[v] == (l > 0):
                return True
        return False

    def count_unsat_clauses(assignment):
        cnt = 0
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    best_assignment = None
    best_unsat = m + 1

    max_restarts = 50
    steps_per_restart = 5000

    for restart in range(max_restarts):
        if time.time() >= deadline:
            break

        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Track unsatisfied clause count and per-variable flip counts
        unsat_count = count_unsat_clauses(assignment)
        if unsat_count == 0:
            return assignment

        # Build list of unsat clauses
        unsat_clauses = [c for c in clauses if not clause_satisfied(c, assignment)]

        # Per-variable weight (number of unsat clauses containing that var)
        var_weight = [0] * n
        for c in unsat_clauses:
            for l in c:
                var_weight[abs(l) - 1] += 1

        # Walk loop
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break

            if unsat_count == 0:
                return assignment

            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick an unsat clause: with prob 0.8 pick one with max weight, else random
            if rng.random() < 0.8:
                max_w = max(var_weight)
                candidates = [c for c in unsat_clauses if max(
                    var_weight[abs(l) - 1] for l in c) >= max_w * 0.8]
                if not candidates:
                    candidates = unsat_clauses
                clause = rng.choice(candidates)
            else:
                clause = rng.choice(unsat_clauses)

            # Pick literal to flip: with prob 0.6 pick the one that minimizes
            # increase in unsat (i.e., maximize satisfied clauses after flip),
            # else random
            if rng.random() < 0.6:
                best_literal = None
                best_score = None
                for l in clause:
                    v = abs(l) - 1
                    # Simulate flip
                    old_val = assignment[v]
                    assignment[v] = not old_val
                    # Count change in unsat: only clauses containing v matter
                    delta = 0
                    for c in clauses:
                        has_v = False
                        for ll in c:
                            if abs(ll) - 1 == v:
                                has_v = True
                                break
                        if not has_v:
                            continue
                        # Check if c is satisfied before and after
                        # We already flipped, so check current state
                        sat_now = False
                        for ll in c:
                            vv = abs(ll) - 1
                            if assignment[vv] == (ll > 0):
                                sat_now = True
                                break
                        # We need to know if it was satisfied before the flip
                        # This is expensive; approximate by just checking current
                        # Actually let's use a different approach:
                        # count how many clauses containing v become unsat/sat
                    # This is O(m*3) per literal, too slow for 50 vars * 3 lits
                    # Let's use a simpler heuristic: prefer flipping the literal
                    # whose variable has the highest weight in unsat clauses
                    assignment[v] = old_val
                    score = var_weight[v]
                    if best_score is None or score > best_score:
                        best_score = score
                        best_literal = l
                if best_literal is not None:
                    literal = best_literal
                else:
                    literal = rng.choice(clause)
            else:
                literal = rng.choice(clause)

            v = abs(literal) - 1
            old_val = assignment[v]
            assignment[v] = not old_val

            # Update unsat_count and var_weight incrementally
            # Find all clauses containing variable v
            for c in clauses:
                has_v = False
                for ll in c:
                    if abs(ll) - 1 == v:
                        has_v = True
                        break
                if not has_v:
                    continue
                # Check satisfaction before and after
                # After flip:
                sat_after = False
                for ll in c:
                    vv = abs(ll) - 1
                    if assignment[vv] == (ll > 0):
                        sat_after = True
                        break
                # Before flip (temporarily revert for check, or compute analytically)
                # Analytical: the only thing that changed is assignment[v]
                # A clause is satisfied if any literal is true.
                # Let's check if c was satisfied before:
                # If after flip it's satisfied, check if it was also satisfied before.
                # If after flip it's not satisfied, it was definitely not satisfied before
                # (since flipping can only change satisfaction status of clauses containing v).
                # Actually flipping a variable can make a clause go from sat to unsat or unsat to sat.
                
                # Simpler: check if c is in unsat_clauses currently
                # This is getting complicated. Let's just recount for clauses containing v.
                pass

            # For correctness and simplicity with 50 vars and moderate clause counts,
            # let's just recount unsat clauses that contain variable v
            new_unsat_count = unsat_count
            # Remove v from var_weight temporarily
            for c in unsat_clauses:
                has_v = False
                for ll in c:
                    if abs(ll) - 1 == v:
                        has_v = True
                        break
                if has_v:
                    # Check if c is now satisfied
                    sat_now = False
                    for ll in c:
                        vv = abs(ll) - 1
                        if assignment[vv] == (ll > 0):
                            sat_now = True
                            break
                    if sat_now:
                        new_unsat_count -= 1
            # Check clauses that were satisfied but now unsat
            for c in clauses:
                if c in unsat_clauses:
                    continue
                has_v = False
                for ll in c:
                    if abs(ll) - 1 == v:
                        has_v = True
                        break
                if not has_v:
                    continue
                sat_now = False
                for ll in c:
                    vv = abs(ll) - 1
                    if assignment[vv] == (ll > 0):
                        sat_now = True
                        break
                if not sat_now:
                    new_unsat_count += 1

            unsat_count = new_unsat_count
            unsat_clauses = [c for c in clauses if not clause_satisfied(c, assignment)]

            # Update var_weight
            var_weight = [0] * n
            for c in unsat_clauses:
                for l in c:
                    var_weight[abs(l) - 1] += 1

        # End of restart loop
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
