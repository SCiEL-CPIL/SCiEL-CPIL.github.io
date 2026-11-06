import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 290

    # Precompute variable incidences for fast local updates.
    # var_clauses[v] = list of (clause_idx, sign), where sign=+1 for +v, -1 for -v.
    var_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            v_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v_idx].append((c_idx, sign))

    # Clause-level data for incremental satisfaction maintenance.
    # clause_lits[c] = [(var_idx, sign), ...]
    clause_lits = []
    for clause in clauses:
        clause_lits.append([(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause])

    rng = random.Random()

    max_steps_per_restart = 20000
    noise_prob = 0.02
    stall_limit = 5000

    def initialize_state(assignment):
        # sat_count[c] = number of literals in clause c satisfied by assignment.
        # A clause is satisfied iff sat_count[c] > 0.
        sat_count = [0] * m
        unsat_count = 0
        for c_idx, lits in enumerate(clause_lits):
            cnt = 0
            for v_idx, sign in lits:
                if assignment[v_idx] == (sign == 1):
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                unsat_count += 1
        return sat_count, unsat_count

    def pick_unsat_clause(unsat_indices):
        # Random unsat clause selection.
        return unsat_indices[rng.randrange(len(unsat_indices))]

    while time.time() < deadline:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count, unsat_count = initialize_state(assignment)

        if unsat_count == 0:
            return assignment

        # Maintain a set/list of unsatisfied clause indices.
        # For n=50 and moderate m, list removal is acceptable, but we use a set for O(1) membership and list for random choice.
        # We'll keep both: unsat_set for membership, unsat_list for random access.
        # To avoid expensive list scans on removal, we use a lazy approach:
        # - unsat_set holds current unsat clauses.
        # - When picking, sample from unsat_set by converting to list only when needed? That can be costly.
        # Alternative: maintain a list and swap-remove. We'll rebuild list only if it gets too stale.
        # Simpler: use list + set; on removal, mark in set and do lazy cleanup periodically.

        unsat_set = set()
        for c_idx in range(m):
            if sat_count[c_idx] == 0:
                unsat_set.add(c_idx)

        steps = 0
        no_improve_steps = 0
        best_unsat = unsat_count

        while steps < max_steps_per_restart and time.time() < deadline:
            if not unsat_set:
                return assignment

            # Pick an unsatisfied clause.
            # To avoid O(len(unsat_set)) list conversion every step, we maintain an unsat_list.
            # We'll rebuild unsat_list if it is empty or if we suspect staleness.
            # For simplicity and speed, use list(unsat_set) only when needed; with m likely < few thousands, this is okay.
            # But to be safer, maintain unsat_list with lazy deletion.
            # Given constraints, we use list(unsat_set) but throttle by checking size.
            # If m is large, this could be costly. We'll implement a hybrid:
            # - Keep unsat_list; on removal, just remove from set.
            # - When unsat_list is empty or every 1000 steps, rebuild from set.
            # This avoids frequent full scans.

            if not hasattr(pick_unsat_clause, "_unsat_list"):
                pick_unsat_clause._unsat_list = []
                pick_unsat_clause._rebuild_counter = 0

            pick_unsat_clause._rebuild_counter += 1
            if not pick_unsat_clause._unsat_list or pick_unsat_clause._rebuild_counter >= 1000:
                pick_unsat_clause._unsat_list = list(unsat_set)
                pick_unsat_clause._rebuild_counter = 0

            if not pick_unsat_clause._unsat_list:
                return assignment

            c_idx = pick_unsat_clause._unsat_list[rng.randrange(len(pick_unsat_clause._unsat_list))]

            # If the sampled clause is no longer unsat (stale), rebuild immediately.
            if sat_count[c_idx] > 0:
                pick_unsat_clause._unsat_list = list(unsat_set)
                if not pick_unsat_clause._unsat_list:
                    return assignment
                c_idx = pick_unsat_clause._unsat_list[rng.randrange(len(pick_unsat_clause._unsat_list))]

            # Variable selection heuristic:
            # For each literal in the clause, estimate the change in unsat_count if flipped.
            # We use the standard WalkSAT score:
            #   score(v) = (# clauses that would become satisfied) - (# clauses that would become unsatisfied)
            # We compute this incrementally using var_clauses and sat_count.
            best_var = None
            best_score = -10**9

            for lit in clauses[c_idx]:
                v_idx = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                current_val = assignment[v_idx]
                new_val = not current_val

                # For each clause containing v_idx, determine if flipping v_idx changes its satisfaction status.
                # A clause's sat_count changes by +1 if the literal becomes satisfied, -1 if it becomes unsatisfied.
                # We only care about clauses where sat_count is 0 or 1, because:
                # - If sat_count >= 2, flipping one literal cannot make it unsatisfied.
                # - If sat_count == 0, flipping to satisfy makes it satisfied (if the literal is the one we're flipping).
                # - If sat_count == 1, flipping the only satisfied literal makes it unsatisfied.

                score = 0
                for (clause_i, lit_sign) in var_clauses[v_idx]:
                    sc = sat_count[clause_i]
                    if sc == 0:
                        # Clause is unsatisfied. If flipping v_idx satisfies this literal, it becomes satisfied.
                        if lit_sign == sign and new_val == (lit_sign == 1):
                            score += 1
                    elif sc == 1:
                        # Clause is satisfied by exactly one literal. If that literal is this one and we unsatisfy it, it becomes unsatisfied.
                        if lit_sign == sign and current_val == (lit_sign == 1):
                            score -= 1
                    # If sc >= 2, no change in satisfaction status.

                # Add a small random tie-breaker to avoid deterministic cycles.
                score += rng.random() * 0.1

                if score > best_score:
                    best_score = score
                    best_var = v_idx

            # Noise: occasionally pick a random variable from the clause.
            if rng.random() < noise_prob:
                chosen_lit = rng.choice(clauses[c_idx])
                v_idx = abs(chosen_lit) - 1
            else:
                v_idx = best_var

            # Flip the variable.
            old_val = assignment[v_idx]
            new_val = not old_val
            assignment[v_idx] = new_val

            # Update sat_count and unsat_set for all clauses containing v_idx.
            for (clause_i, lit_sign) in var_clauses[v_idx]:
                # Determine if the literal was satisfied before and after the flip.
                was_sat = old_val == (lit_sign == 1)
                is_sat = new_val == (lit_sign == 1)

                if was_sat and not is_sat:
                    sat_count[clause_i] -= 1
                    if sat_count[clause_i] == 0:
                        unsat_set.add(clause_i)
                elif not was_sat and is_sat:
                    sat_count[clause_i] += 1
                    if sat_count[clause_i] == 1:
                        unsat_set.discard(clause_i)
                # If both were sat or both were unsat, no change in sat_count.

            steps += 1
            no_improve_steps += 1

            # Track best unsat count for restart decision.
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                no_improve_steps = 0

            # Restart if no improvement for too long.
            if no_improve_steps > stall_limit:
                break

    # Return the last assignment; if not satisfying, the instance scores 0.
    return assignment
