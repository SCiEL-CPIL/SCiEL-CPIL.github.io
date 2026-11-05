"""Improved planted 3-SAT solver: fast WalkSAT with incremental clause tracking and time guard."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline: 300s wall-clock, use 295s safety margin
    deadline = time.time() + 295.0

    # Precompute: for each variable, list of (clause_idx, literal_sign) pairs
    # literal_sign: +1 if positive literal, -1 if negative
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            var_clauses[vi].append((ci, sign))

    best_assignment = None
    best_unsat = m + 1

    def count_unsat(assign):
        cnt = 0
        for clause in clauses:
            sat = False
            for l in clause:
                if assign[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    rng_seed = 42
    restart = 0

    while time.time() < deadline - 2.0:
        rng = random.Random(rng_seed + restart)
        restart += 1

        # Random initial assignment
        assign = [rng.random() < 0.5 for _ in range(n)]

        # Build clause satisfaction state incrementally
        # clause_sat[ci] = number of satisfied literals in clause ci
        clause_sat = [0] * m
        for ci, clause in enumerate(clauses):
            for l in clause:
                if assign[abs(l) - 1] == (l > 0):
                    clause_sat[ci] += 1

        unsat_count = sum(1 for s in clause_sat if s == 0)

        if unsat_count == 0:
            return assign

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assign[:]

        # WalkSAT with weighted variable selection
        # Use a smooth surrogate: minimize sum over unsat clauses of exp(-w * sat_literals)
        # In practice: pick variable that maximizes (gain - loss) with random tiebreak
        # Incremental update of clause_sat

        max_steps = 30000
        step = 0
        # Time per restart: cap at ~3 seconds
        restart_deadline = min(deadline - 2.0, time.time() + 3.0)

        while step < max_steps and time.time() < restart_deadline:
            # Find an unsatisfied clause
            unsat_indices = [i for i in range(m) if clause_sat[i] == 0]
            if not unsat_indices:
                return assign

            if len(unsat_indices) < best_unsat:
                best_unsat = len(unsat_indices)
                best_assignment = assign[:]

            ci = rng.choice(unsat_indices)
            clause = clauses[ci]

            # For each literal in the clause, compute flip score
            best_var = -1
            best_score = -1e18
            for l in clause:
                vi = abs(l) - 1
                # Compute gain and loss by checking all clauses containing variable vi
                gain = 0
                loss = 0
                old_val = assign[vi]
                new_val = not old_val
                for (cidx, sign) in var_clauses[vi]:
                    # Current satisfaction of this clause
                    cur_sat = clause_sat[cidx]
                    # After flip, how does this clause change?
                    # The literal (vi, sign) is satisfied if assign[vi] == (sign > 0)
                    lit_sat_before = (old_val == (sign > 0))
                    lit_sat_after = (new_val == (sign > 0))
                    if lit_sat_before and not lit_sat_after:
                        # This literal goes from sat to unsat
                        if cur_sat == 1:
                            # Clause was satisfied only by this literal -> becomes unsat
                            loss += 1
                    elif not lit_sat_before and lit_sat_after:
                        # This literal goes from unsat to sat
                        if cur_sat == 0:
                            # Clause was unsat -> becomes sat
                            gain += 1
                score = gain - loss
                # Add small random component for diversity
                score += rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_var = vi

            if best_var >= 0:
                old_val = assign[best_var]
                assign[best_var] = not old_val
                # Update clause_sat for all clauses containing best_var
                for (cidx, sign) in var_clauses[best_var]:
                    lit_sat_before = (old_val == (sign > 0))
                    lit_sat_after = (not old_val == (sign > 0))
                    if lit_sat_before and not lit_sat_after:
                        clause_sat[cidx] -= 1
                    elif not lit_sat_before and lit_sat_after:
                        clause_sat[cidx] += 1

                # Update unsat_count
                # We need to track it incrementally too, but for simplicity recompute occasionally
                # Actually let's track it:
                # For each clause we modified, check if it crossed 0 boundary
                # This is complex; let's just recompute unsat_count every N steps
                if step % 500 == 0:
                    unsat_count = sum(1 for s in clause_sat if s == 0)
                    if unsat_count == 0:
                        return assign
                    if unsat_count < best_unsat:
                        best_unsat = unsat_count
                        best_assignment = assign[:]

            step += 1

        # End of restart
        final_unsat = sum(1 for s in clause_sat if s == 0)
        if final_unsat == 0:
            return assign
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = assign[:]

    return best_assignment
# EVOLVE-BLOCK-END
