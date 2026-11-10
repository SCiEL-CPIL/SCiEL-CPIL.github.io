"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # 290s safety margin under 330s limit

    # Precompute clause structure
    clause_vars = []
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    def random_assignment():
        return [random.random() < 0.5 for _ in range(n)]

    def compute_clause_counts(assignment):
        """Compute for each clause how many literals are satisfied."""
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        return counts

    def flip_variable(assignment, vi, counts):
        """Flip variable vi and update clause counts incrementally."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        for ci, sign in var_occurrences[vi]:
            was_sat = old_val == (sign == 1)
            now_sat = new_val == (sign == 1)
            if was_sat and not now_sat:
                counts[ci] -= 1
            elif not was_sat and now_sat:
                counts[ci] += 1

    def is_satisfied(counts):
        return all(c >= 1 for c in counts)

    def count_unsat(counts):
        return sum(1 for c in counts if c == 0)

    def walk_sat(assignment, counts, max_steps, rng, noise=0.3):
        """WalkSAT with incremental tracking and noise parameter.
        Returns (success, assignment, counts)."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            ci = rng.choice(unsat)
            lits_in_clause = clause_vars[ci]

            best_vi = None
            best_score = None
            for vi, sign in lits_in_clause:
                break_count = 0
                for oci, osign in var_occurrences[vi]:
                    if oci == ci:
                        continue
                    cur_sat = assignment[vi] == (osign == 1)
                    if cur_sat and counts[oci] == 1:
                        break_count += 1
                # Smooth surrogate: use soft penalty with noise
                score = break_count + rng.random() * noise
                if best_score is None or score < best_score:
                    best_score = score
                    best_vi = vi

            if best_vi is not None:
                flip_variable(assignment, best_vi, counts)

        return False, assignment, counts

    def local_search(assignment, counts, max_steps, rng):
        """Local search: pick variable that satisfies most unsat clauses."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            var_score = [0] * n
            for ci in unsat:
                for vi, sign in clause_vars[ci]:
                    var_score[vi] += 1

            max_score = max(var_score) if var_score else 0
            if max_score == 0:
                vi = rng.randrange(n)
            else:
                candidates = [vi for vi in range(n) if var_score[vi] == max_score]
                vi = rng.choice(candidates)

            flip_variable(assignment, vi, counts)

        return False, assignment, counts

    def polish(assignment, counts, max_steps, rng):
        """High-precision polish: targeted flips to fix remaining unsat clauses.
        Uses a more aggressive strategy with smaller noise and more steps."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            # Try to fix each unsat clause by finding a flip that reduces total unsat
            improved = False
            for ci in unsat:
                for vi, sign in clause_vars[ci]:
                    # Simulate flip
                    old_val = assignment[vi]
                    delta_unsat = 0
                    for oci, osign in var_occurrences[vi]:
                        cur_sat = old_val == (osign == 1)
                        new_sat = (not old_val) == (osign == 1)
                        if cur_sat and not new_sat:
                            if counts[oci] == 1:
                                delta_unsat += 1
                        elif not cur_sat and new_sat:
                            delta_unsat -= 1
                    # Also account for the clause ci itself: flipping vi satisfies one literal
                    # Since ci is unsat (counts[ci]==0), flipping vi will satisfy it
                    # But we already counted oci==ci in the loop above if ci contains vi
                    # Actually, in the loop, for oci==ci: cur_sat is False (unsat), new_sat is True
                    # So delta_unsat -= 1 for that clause. That's correct.
                    
                    if delta_unsat < 0:
                        # This flip reduces unsat count
                        flip_variable(assignment, vi, counts)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # No single flip improves; try random flip with small probability
                # or pick the flip that minimizes delta_unsat
                best_delta = 0
                best_vi = None
                for ci in unsat:
                    for vi, sign in clause_vars[ci]:
                        old_val = assignment[vi]
                        delta_unsat = 0
                        for oci, osign in var_occurrences[vi]:
                            cur_sat = old_val == (osign == 1)
                            new_sat = (not old_val) == (osign == 1)
                            if cur_sat and not new_sat:
                                if counts[oci] == 1:
                                    delta_unsat += 1
                            elif not cur_sat and new_sat:
                                delta_unsat -= 1
                        if delta_unsat < best_delta:
                            best_delta = delta_unsat
                            best_vi = vi
                if best_vi is not None:
                    flip_variable(assignment, best_vi, counts)
                else:
                    # Truly stuck; random flip
                    vi = rng.randrange(n)
                    flip_variable(assignment, vi, counts)

        return False, assignment, counts

    # Main loop with restarts
    best_assignment = None
    best_unsat_count = m + 1

    for attempt in range(200):
        if time.time() > deadline:
            break

        rng = random.Random(attempt * 7919 + 13)
        assignment = random_assignment()
        counts = compute_clause_counts(assignment)

        if is_satisfied(counts):
            return assignment

        cur_unsat = count_unsat(counts)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        # Alternate between WalkSAT (with varying noise) and local search
        if attempt % 3 == 0:
            success, assignment, counts = walk_sat(assignment, counts, 5000, rng, noise=0.2)
        elif attempt % 3 == 1:
            success, assignment, counts = walk_sat(assignment, counts, 5000, rng, noise=0.5)
        else:
            success, assignment, counts = local_search(assignment, counts, 5000, rng)

        if success:
            return assignment

        cur_unsat = count_unsat(counts)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    # Final polish phase: use the best assignment found and try to solve it
    if best_assignment is not None and best_unsat_count < m:
        polish_rng = random.Random(12345)
        polish_assignment = best_assignment[:]
        polish_counts = compute_clause_counts(polish_assignment)
        if not is_satisfied(polish_counts):
            # Run polish with increasing effort
            for polish_attempt in range(5):
                if time.time() > deadline:
                    break
                p_rng = random.Random(polish_attempt * 31 + 7)
                success, polish_assignment, polish_counts = polish(
                    polish_assignment, polish_counts, 20000, p_rng
                )
                if success:
                    return polish_assignment
                cur_unsat = count_unsat(polish_counts)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = polish_assignment[:]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
