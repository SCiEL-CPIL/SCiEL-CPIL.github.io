"""WalkSAT + DLS hybrid with adaptive restarts and polish for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 300s kill

    # Precompute: for each variable, which clauses contain it (and with what sign)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append((ci, lit > 0))

    # Precompute clause lengths for DLS
    clause_lens = np.array([len(c) for c in clauses])

    # Build a lookup: for each clause, list of (var_idx, is_positive)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            cv.append((abs(lit) - 1, lit > 0))
        clause_vars.append(cv)

    # Precompute clause satisfaction counts for fast gains/losses
    # sat_count[ci] = number of literals in clause ci that are currently satisfied
    # This allows O(1) check of whether a clause is uniquely satisfied

    def make_satisfied_array(assignment):
        """Return (satisfied bool array, satisfaction count array)."""
        sat = np.zeros(m, dtype=bool)
        sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            cnt = 0
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    cnt += 1
            sat[ci] = cnt > 0
            sat_count[ci] = cnt
        return sat, sat_count

    def flip_var(assignment, var_idx, sat, sat_count):
        """Flip variable var_idx and update sat/sat_count incrementally."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == new_val:
                # This literal now satisfies the clause
                sat_count[ci] += 1
                sat[ci] = True
            else:
                # This literal no longer satisfies; decrement count
                sat_count[ci] -= 1
                if sat_count[ci] == 0:
                    sat[ci] = False

    def walksat_step(assignment, sat, sat_count, rng, p_walk):
        """One WalkSAT step: pick an unsat clause, flip a variable.
        With probability p_walk, use WalkSAT heuristic; otherwise flip randomly."""
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return
        ci = int(rng.choice(unsat_idx))
        clause = clause_vars[ci]
        if rng.random() < p_walk:
            # WalkSAT: pick variable with best (gains - losses)
            best_var = -1
            best_score = -1000
            for v, is_pos in clause:
                gains = 0
                losses = 0
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if is_pos2 == assignment[v]:
                        # Currently satisfies c2; check if uniquely satisfied
                        if sat_count[c2] == 1:
                            losses += 1
                    else:
                        gains += 1
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat, sat_count)
            else:
                v, _ = rng.choice(clause)
                flip_var(assignment, v, sat, sat_count)
        else:
            v, _ = rng.choice(clause)
            flip_var(assignment, v, sat, sat_count)

    def dls_step(assignment, sat, sat_count, rng):
        """DLS step: pick the shortest unsatisfied clause, flip a random var in it."""
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return
        # Find shortest unsat clause (use clause_lens for speed)
        min_len = 10
        best_ci = int(unsat_idx[0])
        for ci in unsat_idx:
            l = clause_lens[ci]
            if l < min_len:
                min_len = l
                best_ci = int(ci)
                if l <= 1:
                    break
        clause = clause_vars[best_ci]
        v, _ = rng.choice(clause)
        flip_var(assignment, v, sat, sat_count)

    def polish_pass(assignment, sat, sat_count, rng, max_iters=20000):
        """Aggressive polish: try to satisfy remaining unsat clauses with focused search."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            unsat_idx = np.where(~sat)[0]
            if len(unsat_idx) == 0:
                return assignment
            # Pick a random unsat clause
            ci = int(rng.choice(unsat_idx))
            clause = clause_vars[ci]
            # Try each variable in the clause, pick best gain
            best_var = -1
            best_score = -1000
            for v, is_pos in clause:
                gains = 0
                losses = 0
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if is_pos2 == assignment[v]:
                        if sat_count[c2] == 1:
                            losses += 1
                    else:
                        gains += 1
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat, sat_count)
            else:
                v, _ = rng.choice(clause)
                flip_var(assignment, v, sat, sat_count)
            # Check if solved
            if not np.any(~sat):
                return assignment
        return assignment

    best_assignment = None
    best_unsat = m + 1

    for restart in range(300):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat, sat_count = make_satisfied_array(assignment)
        cur_unsat = int(np.sum(~sat))
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        steps_in_restart = 0
        max_steps = 8000
        dls_interval = 150
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.65

        for step in range(max_steps):
            if time.time() > deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            # WalkSAT step
            walksat_step(assignment, sat, sat_count, rng, p_walk)
            steps_in_restart += 1

            # Periodic DLS for escape
            if steps_in_restart % dls_interval == 0:
                dls_step(assignment, sat, sat_count, rng)

            # Check progress every 20 steps (more frequent for better adaptation)
            if step % 20 == 0:
                cur_unsat = int(np.sum(~sat))
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    p_walk = max(0.30, p_walk - 0.02)
                else:
                    stall_count += 1
                prev_unsat = cur_unsat

            # If we're very close (few unsat), do a polish pass
            if cur_unsat <= 5 and step % 100 == 0:
                assignment = polish_pass(assignment, sat, sat_count, rng, max_iters=5000)
                cur_unsat = int(np.sum(~sat))
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]

            # Adaptive restart within this restart attempt
            if stall_count > 12:
                break

        # After main loop, if we got close, try a final polish
        cur_unsat = int(np.sum(~sat))
        if cur_unsat <= 10:
            assignment = polish_pass(assignment, sat, sat_count, rng, max_iters=10000)
            cur_unsat = int(np.sum(~sat))
            if cur_unsat == 0:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    # Final polish on best assignment found
    if best_assignment is not None and best_unsat <= 15:
        final_rng = random.Random(999999)
        best_sat, best_sat_count = make_satisfied_array(best_assignment)
        best_assignment = polish_pass(best_assignment, best_sat, best_sat_count, final_rng, max_iters=20000)
        if not np.any(~best_sat):
            return best_assignment

    return best_assignment if best_assignment else [False] * n
# EVOLVE-BLOCK-END
