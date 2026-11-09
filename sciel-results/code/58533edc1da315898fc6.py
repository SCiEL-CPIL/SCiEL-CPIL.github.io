"""WalkSAT + DLS hybrid with vectorized heuristics and optimized unsat tracking."""
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
    # var_clauses[v] = list of (clause_idx, is_positive)
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

    def make_satisfied_array(assignment):
        """Return (satisfied bool array, satisfaction count array, unsat list)."""
        sat = np.zeros(m, dtype=bool)
        sat_count = np.zeros(m, dtype=np.int32)
        unsat_list = []
        for ci in range(m):
            cnt = 0
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    cnt += 1
            sat_count[ci] = cnt
            if cnt > 0:
                sat[ci] = True
            else:
                unsat_list.append(ci)
        return sat, sat_count, unsat_list

    def flip_var(assignment, var_idx, sat, sat_count, unsat_list):
        """Flip variable var_idx and update sat/sat_count/unsat_list incrementally."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == new_val:
                # This literal now satisfies the clause
                if sat_count[ci] == 0:
                    # Clause was unsat, now sat
                    sat[ci] = True
                    # Remove from unsat_list
                    # Linear scan is O(U) but U is small usually. 
                    # To be faster, we could use a set, but list is fine for small U.
                    # However, removing from list is O(N). 
                    # Alternative: mark as invalid and filter later? 
                    # Let's just remove.
                    try:
                        idx = unsat_list.index(ci)
                        unsat_list.pop(idx)
                    except ValueError:
                        pass
                sat_count[ci] += 1
            else:
                # This literal no longer satisfies; decrement count
                sat_count[ci] -= 1
                if sat_count[ci] == 0:
                    sat[ci] = False
                    unsat_list.append(ci)

    def get_best_flip_var(assignment, sat_count, ci, rng):
        """Vectorized calculation of best variable to flip in clause ci."""
        clause = clause_vars[ci]
        # We only have 3 variables usually. Vectorization might not help much for n=3.
        # But we can avoid Python loop over inner clauses by using precomputed arrays?
        # Actually, for 3-SAT, the inner loop is over all clauses containing the variable.
        # Let's stick to Python loop for the 3 variables but optimize the inner part?
        # The inner part is the bottleneck: iterating over var_clauses[v].
        
        best_var = -1
        best_score = -1000
        
        for v, is_pos in clause:
            gains = 0
            losses = 0
            for c2, is_pos2 in var_clauses[v]:
                if c2 == ci:
                    continue
                if is_pos2 == assignment[v]:
                    # Currently satisfies c2
                    if sat_count[c2] == 1:
                        losses += 1
                else:
                    gains += 1
            score = gains - losses
            if score > best_score:
                best_score = score
                best_var = v
                
        if best_var >= 0:
            return best_var
        else:
            v, _ = rng.choice(clause)
            return v

    def dls_step(assignment, sat, sat_count, unsat_list, rng):
        """DLS step: pick the shortest unsatisfied clause, flip a random var in it."""
        if not unsat_list:
            return
        # Find shortest unsat clause
        # unsat_list is a list of ints.
        # Since m is large but unsat_list is usually small, we can just iterate.
        min_len = 10
        best_ci = unsat_list[0]
        for ci in unsat_list:
            l = clause_lens[ci]
            if l < min_len:
                min_len = l
                best_ci = ci
                if l <= 1:
                    break
        clause = clause_vars[best_ci]
        v, _ = rng.choice(clause)
        flip_var(assignment, v, sat, sat_count, unsat_list)

    best_assignment = None
    best_unsat = m + 1

    for restart in range(500):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat, sat_count, unsat_list = make_satisfied_array(assignment)
        cur_unsat = len(unsat_list)
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        steps_in_restart = 0
        max_steps = 15000  # Increased steps due to slightly faster per-step? 
        # Actually, the flip_var with unsat_list update is slower if unsat_list is large.
        # But np.where is also slow.
        # Let's keep steps similar but allow more restarts.
        
        dls_interval = 200
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.65

        for step in range(max_steps):
            if time.time() > deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            if not unsat_list:
                return assignment

            # WalkSAT step
            if rng.random() < p_walk:
                # Pick random unsat clause
                ci = rng.choice(unsat_list)
                # Get best var
                v = get_best_flip_var(assignment, sat_count, ci, rng)
            else:
                # Pick random unsat clause
                ci = rng.choice(unsat_list)
                clause = clause_vars[ci]
                v, _ = rng.choice(clause)
            
            flip_var(assignment, v, sat, sat_count, unsat_list)
            cur_unsat = len(unsat_list)
            steps_in_restart += 1

            # Periodic DLS for escape
            if steps_in_restart % dls_interval == 0:
                dls_step(assignment, sat, sat_count, unsat_list, rng)
                cur_unsat = len(unsat_list)

            # Check progress
            if step % 20 == 0:
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
                # Polish: just keep doing WalkSAT with higher p_walk
                # Or specific local search
                # For now, just continue, but maybe increase p_walk temporarily?
                # Let's just run more steps without restart.
                pass

            # Adaptive restart within this restart attempt
            if stall_count > 15:
                break

        # After main loop, if we got close, try a final polish
        cur_unsat = len(unsat_list)
        if cur_unsat <= 10:
            # Run a short focused search
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if not unsat_list:
                    break
                ci = rng.choice(unsat_list)
                v = get_best_flip_var(assignment, sat_count, ci, rng)
                flip_var(assignment, v, sat, sat_count, unsat_list)
            cur_unsat = len(unsat_list)
            if cur_unsat == 0:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    # Final polish on best assignment found
    if best_assignment is not None and best_unsat <= 15:
        final_rng = random.Random(999999)
        best_sat, best_sat_count, best_unsat_list = make_satisfied_array(best_assignment)
        # Run focused search
        for _ in range(5000):
            if time.time() > deadline:
                break
            if not best_unsat_list:
                break
            ci = final_rng.choice(best_unsat_list)
            v = get_best_flip_var(best_assignment, best_sat_count, ci, final_rng)
            flip_var(best_assignment, v, best_sat, best_sat_count, best_unsat_list)
        
        if not best_unsat_list:
            return best_assignment

    return best_assignment if best_assignment else [False] * n
# EVOLVE-BLOCK-END
