"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before the 330s kill

    # Pre-compute clauses as 0-based (idx, pos) tuples
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)

    # Build variable -> list of (clause_idx, literal_position_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for k, (idx, _pos) in enumerate(c):
            var_to_clauses[idx].append((ci, k))

    # Precompute for each clause ci: for each literal position k, the variable idx
    # This is just clauses_idx, but we'll use it directly.

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1

    max_flips_per_run = 5000
    plateau_threshold = 60
    base_noise = 0.15
    max_noise = 0.85
    noise_increase = 0.04
    noise_check_interval = 80

    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # masks[ci] = bitmask of satisfied literal positions in clause ci
        masks = [0] * m
        unsat_list = []  # list of unsatisfied clause indices
        unsat_set = set()  # for O(1) membership

        for ci in range(m):
            c = clauses_idx[ci]
            mask = 0
            for k, (idx, pos) in enumerate(c):
                if assignment[idx] == pos:
                    mask |= (1 << k)
            masks[ci] = mask
            if mask == 0:
                unsat_list.append(ci)
                unsat_set.add(ci)

        if not unsat_list:
            return assignment

        if len(unsat_list) < best_unsat:
            best_unsat = len(unsat_list)
            best_assignment = assignment[:]

        # Per-variable count of how many unsat clauses contain this variable
        # This is maintained incrementally for the greedy heuristic
        var_unsat_count = [0] * n
        for ci in unsat_list:
            seen_vars = set()
            for idx, _pos in clauses_idx[ci]:
                if idx not in seen_vars:
                    var_unsat_count[idx] += 1
                    seen_vars.add(idx)

        noise = base_noise
        stuck = 0
        run_best_unsat = len(unsat_list)

        for _flip in range(max_flips_per_run):
            if time.time() > deadline:
                break
            if not unsat_list:
                return assignment

            cur_unsat = len(unsat_list)
            if cur_unsat < run_best_unsat:
                run_best_unsat = cur_unsat
                stuck = 0
            else:
                stuck += 1

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            if stuck > plateau_threshold:
                break

            # Pick an unsatisfied clause uniformly
            ci = rng.choice(unsat_list)
            c = clauses_idx[ci]

            # Find unsatisfied literal positions in this clause
            unsat_positions = []
            for k, (idx, pos) in enumerate(c):
                if not (assignment[idx] == pos):
                    unsat_positions.append(k)

            if unsat_positions and rng.random() > noise:
                # Greedy: pick the unsat literal whose variable appears in the
                # fewest other unsat clauses (excluding ci)
                best_k = unsat_positions[0]
                best_score = var_unsat_count[c[best_k][0]]
                for k in unsat_positions[1:]:
                    idx = c[k][0]
                    score = var_unsat_count[idx]
                    if score < best_score:
                        best_score = score
                        best_k = k
                k = best_k
            else:
                k = rng.randrange(3)

            idx, _pos = c[k]
            old_val = assignment[idx]
            assignment[idx] = not old_val

            # Incrementally update masks and unsat tracking for clauses touching idx
            for cj, k2 in var_to_clauses[idx]:
                old_mask = masks[cj]
                # Determine if this variable's literal was satisfied before/after
                pos2 = clauses_idx[cj][k2][1]
                was_sat = (old_val == pos2)
                now_sat = (assignment[idx] == pos2)
                if was_sat and not now_sat:
                    new_mask = old_mask & ~(1 << k2)
                elif not was_sat and now_sat:
                    new_mask = old_mask | (1 << k2)
                else:
                    new_mask = old_mask
                masks[cj] = new_mask

                was_unsat = (old_mask == 0)
                is_unsat = (new_mask == 0)

                if was_unsat and not is_unsat:
                    # Clause cj just became satisfied
                    unsat_set.discard(cj)
                    # Remove from unsat_list (lazy: mark and rebuild later)
                    # For now, we'll rebuild unsat_list periodically
                    # Update var_unsat_count
                    seen_vars = set()
                    for v_idx, _ in clauses_idx[cj]:
                        if v_idx not in seen_vars:
                            var_unsat_count[v_idx] -= 1
                            seen_vars.add(v_idx)
                elif not was_unsat and is_unsat:
                    # Clause cj just became unsatisfied
                    unsat_set.add(cj)
                    unsat_list.append(cj)
                    seen_vars = set()
                    for v_idx, _ in clauses_idx[cj]:
                        if v_idx not in seen_vars:
                            var_unsat_count[v_idx] += 1
                            seen_vars.add(v_idx)

            # Periodically rebuild unsat_list to remove stale entries
            if _flip % 200 == 199:
                unsat_list = [ci2 for ci2 in unsat_list if ci2 in unsat_set]

            # Adaptive noise
            if stuck % noise_check_interval == 0 and stuck > 0:
                noise = min(max_noise, noise + noise_increase)

        if time.time() > deadline:
            break

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
