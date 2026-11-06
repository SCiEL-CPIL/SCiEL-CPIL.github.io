"""Combined 3-SAT solver: preprocessing + CDCL + improved WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 320.0

    # ==================== PREPROCESSING ====================
    # Unit propagation and pure literal elimination
    # assignment[i] = None (unassigned), True, or False
    assignment = [None] * n
    # active_clauses: list of lists of (var_idx, polarity) tuples
    active_clauses = []
    for c in clauses:
        active_clauses.append([(abs(l) - 1, l > 0) for l in c])

    # Pure literal elimination
    def pure_literal_elimination():
        """Assign pure literals. Returns True if any assignment was made."""
        # Count occurrences of each literal
        pos_count = [0] * n
        neg_count = [0] * n
        for cv in active_clauses:
            for vi, pos in cv:
                if pos:
                    pos_count[vi] += 1
                else:
                    neg_count[vi] += 1

        made = False
        for vi in range(n):
            if assignment[vi] is not None:
                continue
            if pos_count[vi] == 0 and neg_count[vi] > 0:
                assignment[vi] = False
                made = True
            elif neg_count[vi] == 0 and pos_count[vi] > 0:
                assignment[vi] = True
                made = True
        return made

    # Unit propagation
    def unit_propagation():
        """Perform unit propagation. Returns True if any assignment was made, False if conflict."""
        changed = True
        while changed:
            changed = False
            for i in range(len(active_clauses)):
                cv = active_clauses[i]
                if len(cv) == 0:
                    return False  # Conflict
                if len(cv) == 1:
                    vi, pos = cv[0]
                    val = pos
                    if assignment[vi] is not None:
                        if assignment[vi] != val:
                            return False  # Conflict
                        else:
                            active_clauses[i] = []
                            changed = True
                            continue
                    assignment[vi] = val
                    changed = True
                    # Propagate to other clauses
                    for j in range(len(active_clauses)):
                        if not active_clauses[j]:
                            continue
                        new_cv = []
                        satisfied = False
                        for vj, pj in active_clauses[j]:
                            if vj == vi:
                                if pj == val:
                                    satisfied = True
                                    break
                                else:
                                    continue  # Remove false literal
                            else:
                                new_cv.append((vj, pj))
                        if satisfied:
                            active_clauses[j] = []
                        else:
                            active_clauses[j] = new_cv
                    break  # Restart loop after one assignment

        return True

    # Apply preprocessing
    for _ in range(10):
        if not pure_literal_elimination() and not unit_propagation():
            # Conflict during preprocessing
            # Fall back to random assignment
            return [random.random() < 0.5 for _ in range(n)]
        if not pure_literal_elimination() and not unit_propagation():
            break

    # Check if all variables are assigned
    if all(a is not None for a in assignment):
        # Verify satisfaction
        for c in clauses:
            sat = False
            for l in c:
                vi = abs(l) - 1
                if assignment[vi] == (l > 0):
                    sat = True
                    break
            if not sat:
                return [random.random() < 0.5 for _ in range(n)]
        return assignment

    # Build reduced problem
    # Filter out satisfied and empty clauses
    reduced_clauses = []
    for cv in active_clauses:
        if not cv:
            continue  # Satisfied
        new_cv = []
        for vi, pos in cv:
            if assignment[vi] is not None:
                if assignment[vi] == pos:
                    new_cv = None  # Satisfied
                    break
                # else: false literal, skip
            else:
                new_cv.append((vi, pos))
        if new_cv is None:
            continue  # Satisfied
        if new_cv:
            reduced_clauses.append(new_cv)

    # If no clauses remain, we're done
    if not reduced_clauses:
        for i in range(n):
            if assignment[i] is None:
                assignment[i] = False
        return assignment

    # ==================== SOLVE REDUCED PROBLEM ====================
    # Build variable list for reduced problem
    # Map original variable index to reduced index
    unassigned_vars = [i for i in range(n) if assignment[i] is None]
    var_map = {vi: i for i, vi in enumerate(unassigned_vars)}
    r = len(unassigned_vars)  # Number of reduced variables

    # Build reduced clauses with new variable indices
    reduced_clauses_new = []
    for cv in reduced_clauses:
        new_cv = [(var_map[vi], pos) for vi, pos in cv]
        reduced_clauses_new.append(new_cv)

    rm = len(reduced_clauses_new)

    # Pre-compute structures for reduced problem
    clause_vars = reduced_clauses_new
    var_clauses = [[] for _ in range(r)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))

    def init_state(assignment_r):
        sat_count = [0] * rm
        false_vars = [set() for _ in range(rm)]
        for ci, cv in enumerate(clause_vars):
            for vi, pos in cv:
                if assignment_r[vi] == pos:
                    sat_count[ci] += 1
                else:
                    false_vars[ci].add(vi)
        return sat_count, false_vars

    def count_unsat(sat_count):
        return sum(1 for sc in sat_count if sc == 0)

    def find_unsat_clause(sat_count):
        for ci in range(rm):
            if sat_count[ci] == 0:
                return ci
        return -1

    def flip_variable(vi, assignment_r, sat_count, false_vars):
        old_val = assignment_r[vi]
        new_val = not old_val
        assignment_r[vi] = new_val

        delta = 0
        for (ci, pos) in var_clauses[vi]:
            if old_val == pos:
                sat_count[ci] -= 1
                false_vars[ci].add(vi)
                if sat_count[ci] == 0:
                    delta += 1
            else:
                sat_count[ci] += 1
                false_vars[ci].discard(vi)
                if sat_count[ci] == 1:
                    delta -= 1
        return delta

    # ==================== CDCL SOLVER (simplified) ====================
    def cdcl_solve():
        """Simple CDCL solver for the reduced problem."""
        values = [0] * r  # 0=unassigned, 1=True, -1=False
        trail = []
        trail_lim = [0]
        level = [0] * r
        reason = [None] * r

        # Watches
        watches = [[] for _ in range(r)]
        all_clauses = [cv[:] for cv in clause_vars]

        for ci, cv in enumerate(all_clauses):
            if len(cv) >= 2:
                v1, p1 = cv[0]
                v2, p2 = cv[1]
                watches[v1].append((ci, p1))
                watches[v2].append((ci, p2))
            elif len(cv) == 1:
                v1, p1 = cv[0]
                watches[v1].append((ci, p1))

        unit_queue = []

        def is_satisfied(ci):
            cv = all_clauses[ci]
            for vi, pos in cv:
                if values[vi] != 0 and values[vi] == (1 if pos else -1):
                    return True
            return False

        def propagate():
            while unit_queue:
                vi = unit_queue.pop()
                if values[vi] == 0:
                    continue

                new_watches = []
                for (ci, polarity) in watches[vi]:
                    cv = all_clauses[ci]
                    if values[vi] == (1 if polarity else -1):
                        new_watches.append((ci, polarity))
                        continue

                    found_new_watch = False
                    for j, (vj, pj) in enumerate(cv):
                        if vj == vi:
                            continue
                        if values[vj] == 0:
                            watches[vj].append((ci, pj))
                            found_new_watch = True
                            break
                        elif values[vj] == (1 if pj else -1):
                            found_new_watch = True
                            break

                    if not found_new_watch:
                        return ci

                watches[vi] = new_watches

            return -1

        def analyze_conflict(conflict_ci):
            learned = []
            seen = set()
            current_level = len(trail_lim) - 1

            ci = conflict_ci
            cv = all_clauses[ci][:]

            while cv:
                max_level = -1
                max_var = -1
                max_pol = True
                for vi, pos in cv:
                    if level[vi] > max_level:
                        max_level = level[vi]
                        max_var = vi
                        max_pol = pos

                if max_var == -1 or max_level < current_level:
                    break

                learned.append((max_var, not max_pol))
                seen.add(max_var)

                if reason[max_var] is None:
                    break

                reason_ci = reason[max_var]
                reason_cv = all_clauses[reason_ci]

                new_cv = []
                for vi, pos in reason_cv:
                    if vi in seen:
                        continue
                    if values[vi] == 0 or level[vi] < current_level:
                        new_cv.append((vi, pos))
                        seen.add(vi)

                cv = new_cv

            if not learned:
                return -1, 0

            # Determine backjump level
            levels = []
            for vi, pos in learned:
                if level[vi] > 0:
                    levels.append(level[vi])
            levels.sort(reverse=True)
            if len(levels) >= 2:
                backjump_level = levels[1]
            elif len(levels) == 1:
                backjump_level = levels[0]
            else:
                backjump_level = 0

            learned_ci = len(all_clauses)
            all_clauses.append(learned)

            if len(learned) >= 2:
                v1, p1 = learned[0]
                v2, p2 = learned[1]
                watches[v1].append((learned_ci, p1))
                watches[v2].append((learned_ci, p2))
            elif len(learned) == 1:
                v1, p1 = learned[0]
                watches[v1].append((learned_ci, p1))
                unit_queue.append(v1)

            return learned_ci, backjump_level

        def backtrack(to_level):
            while len(trail) > (trail_lim[to_level] if to_level < len(trail_lim) else 0):
                vi = trail.pop()
                values[vi] = 0
                level[vi] = 0
                reason[vi] = None
            while len(trail_lim) > to_level + 1:
                trail_lim.pop()

        decisions = 0
        max_decisions = 200000

        while decisions < max_decisions:
            if time.time() > deadline - 10.0:
                return None

            conflict = propagate()

            if conflict != -1:
                if len(trail_lim) <= 1:
                    return None

                learned_ci, backjump_level = analyze_conflict(conflict)
                if learned_ci == -1:
                    return None
                backtrack(backjump_level)
                decisions += 50
                continue

            if len(trail) == r:
                for ci, cv in enumerate(all_clauses):
                    if not is_satisfied(ci):
                        return None
                return [values[vi] == 1 for vi in range(r)]

            # Pick decision variable: most frequent in unsatisfied clauses
            best_var = -1
            best_score = -1
            for vi in range(r):
                if values[vi] != 0:
                    continue
                score = 0
                for (ci, pos) in var_clauses[vi]:
                    cv = all_clauses[ci]
                    sat = False
                    for vj, pj in cv:
                        if values[vj] != 0 and values[vj] == (1 if pj else -1):
                            sat = True
                            break
                    if not sat:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_var = vi

            if best_var == -1:
                break

            values[best_var] = -1
            level[best_var] = len(trail_lim)
            reason[best_var] = None
            trail.append(best_var)
            trail_lim.append(len(trail))
            decisions += 1

        return None

    # Try CDCL
    cdcl_result = cdcl_solve()
    if cdcl_result is not None:
        # Verify
        valid = True
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if cdcl_result[vi] == pos:
                    sat = True
                    break
            if not sat:
                valid = False
                break
        if valid:
            # Map back to original assignment
            for i, vi in enumerate(unassigned_vars):
                assignment[vi] = cdcl_result[i]
            return assignment

    # ==================== WALKSAT FALLBACK ====================
    rng = random.Random(12345)

    # Initialize with random assignment for reduced problem
    assignment_r = [rng.random() < 0.5 for _ in range(r)]
    sat_count, false_vars = init_state(assignment_r)
    cur_unsat = count_unsat(sat_count)

    if cur_unsat == 0:
        for i, vi in enumerate(unassigned_vars):
            assignment[vi] = assignment_r[i]
        return assignment

    best_unsat = cur_unsat
    best_assignment_r = assignment_r[:]

    # WalkSAT parameters
    w = 0.75  # Probability of flipping a literal in the unsat clause (higher for planted SAT)
    w_min = 0.1
    w_max = 0.95

    max_restarts = 1000

    # Luby sequence for restart lengths
    def luby_sequence():
        k = 1
        while True:
            for _ in range(k):
                yield k
            k *= 2

    luby_iter = luby_sequence()
    current_restart_limit = next(luby_iter)

    steps_in_restart = 0
    total_restarts = 0
    w = 0.75

    while time.time() < deadline - 5.0:
        if cur_unsat == 0:
            break

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment_r = assignment_r[:]

        # Restart logic
        if steps_in_restart > current_restart_limit:
            total_restarts += 1
            if total_restarts >= max_restarts:
                break

            # Re-randomize with slight bias toward best assignment
            if rng.random() < 0.3:
                # Perturb best assignment
                assignment_r = best_assignment_r[:]
                for _ in range(max(1, r // 10)):
                    assignment_r[rng.randrange(r)] = not assignment_r[rng.randrange(r)]
            else:
                assignment_r = [rng.random() < 0.5 for _ in range(r)]
            sat_count, false_vars = init_state(assignment_r)
            cur_unsat = count_unsat(sat_count)

            if cur_unsat == 0:
                break

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment_r = assignment_r[:]

            steps_in_restart = 0
            current_restart_limit = next(luby_iter)
            if current_restart_limit > 50000:
                current_restart_limit = 50000
            # Reset w with some randomness
            w = rng.uniform(0.5, 0.9)
            continue

        # Find an unsatisfied clause - prefer shorter unsatisfied clauses
        ci = find_unsat_clause(sat_count)
        if ci == -1:
            break

        # Choose variable to flip
        if rng.random() < w:
            # Flip a literal in the unsat clause
            cv = clause_vars[ci]
            idx = rng.randrange(len(cv))
            vi = cv[idx][0]
        else:
            # Min-conflict: pick variable whose flip reduces unsat the most
            fv = false_vars[ci]
            if fv:
                best_vi = -1
                best_delta = float('inf')
                # Sample more variables for better quality
                fv_list = list(fv)
                sample_size = min(len(fv_list), 8)
                sampled = rng.sample(fv_list, sample_size)
                for vi in sampled:
                    # Compute delta
                    delta = 0
                    for (c2, pos) in var_clauses[vi]:
                        old_val = assignment_r[vi]
                        if old_val == pos:
                            if sat_count[c2] == 1:
                                delta += 1
                        else:
                            if sat_count[c2] == 0:
                                delta -= 1
                    if delta < best_delta:
                        best_delta = delta
                        best_vi = vi
                vi = best_vi
            else:
                cv = clause_vars[ci]
                idx = rng.randrange(len(cv))
                vi = cv[idx][0]

        # Flip
        old_cur_unsat = cur_unsat
        new_delta = flip_variable(vi, assignment_r, sat_count, false_vars)
        cur_unsat += new_delta

        # WalkSAT: if increased unsat significantly, consider reverting (less aggressive)
        if new_delta > 1:
            if rng.random() > 0.7:
                flip_variable(vi, assignment_r, sat_count, false_vars)
                cur_unsat = old_cur_unsat
        elif new_delta == 1:
            if rng.random() > 0.85:
                flip_variable(vi, assignment_r, sat_count, false_vars)
                cur_unsat = old_cur_unsat

        steps_in_restart += 1

        # Adapt w: increase when making progress, decrease when stuck
        if steps_in_restart % 500 == 0:
            if cur_unsat < best_unsat:
                w = min(w_max, w * 0.95)  # More exploitation when improving
            else:
                w = max(w_min, w * 1.02)  # More exploration when stuck

    # Use best assignment found
    if best_unsat < cur_unsat:
        assignment_r = best_assignment_r

    # Final polish: local search to fix remaining unsatisfied clauses
    if best_unsat > 0:
        sat_count, false_vars = init_state(assignment_r)
        cur_unsat = count_unsat(sat_count)
        for _ in range(2000):
            if cur_unsat == 0 or time.time() > deadline - 3.0:
                break
            ci = find_unsat_clause(sat_count)
            if ci == -1:
                break
            fv = false_vars[ci]
            if not fv:
                continue
            best_vi = -1
            best_delta = float('inf')
            for vi in fv:
                delta = 0
                for (c2, pos) in var_clauses[vi]:
                    old_val = assignment_r[vi]
                    if old_val == pos:
                        if sat_count[c2] == 1:
                            delta += 1
                    else:
                        if sat_count[c2] == 0:
                            delta -= 1
                if delta < best_delta:
                    best_delta = delta
                    best_vi = vi
            if best_vi != -1 and best_delta < 0:
                flip_variable(best_vi, assignment_r, sat_count, false_vars)
                cur_unsat += best_delta
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment_r = assignment_r[:]
            elif best_vi != -1:
                # No improving move, try random flip
                vi = rng.choice(list(fv))
                d = flip_variable(vi, assignment_r, sat_count, false_vars)
                cur_unsat += d
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment_r = assignment_r[:]
        if best_unsat < cur_unsat:
            assignment_r = best_assignment_r

    # Map back to original assignment
    for i, vi in enumerate(unassigned_vars):
        assignment[vi] = assignment_r[i]

    # Verify final solution
    for c in clauses:
        sat = False
        for l in c:
            vi = abs(l) - 1
            if assignment[vi] == (l > 0):
                sat = True
                break
        if not sat:
            # Try more random restarts with different seeds
            for attempt in range(50):
                if time.time() > deadline - 2.0:
                    break
                rng2 = random.Random(12345 + attempt * 7919)
                assignment_r2 = [rng2.random() < 0.5 for _ in range(r)]
                sat_count2, false_vars2 = init_state(assignment_r2)
                cur_unsat2 = count_unsat(sat_count2)
                if cur_unsat2 == 0:
                    for i, vi in enumerate(unassigned_vars):
                        assignment[vi] = assignment_r2[i]
                    return assignment
                # Quick local search from this start
                for _ in range(500):
                    if cur_unsat2 == 0:
                        break
                    ci2 = find_unsat_clause(sat_count2)
                    if ci2 == -1:
                        break
                    fv2 = false_vars2[ci2]
                    if not fv2:
                        continue
                    best_vi2 = -1
                    best_delta2 = float('inf')
                    for vi2 in list(fv2)[:5]:
                        delta2 = 0
                        for (c2, pos) in var_clauses[vi2]:
                            old_val2 = assignment_r2[vi2]
                            if old_val2 == pos:
                                if sat_count2[c2] == 1:
                                    delta2 += 1
                            else:
                                if sat_count2[c2] == 0:
                                    delta2 -= 1
                        if delta2 < best_delta2:
                            best_delta2 = delta2
                            best_vi2 = vi2
                    if best_vi2 != -1 and best_delta2 < 0:
                        flip_variable(best_vi2, assignment_r2, sat_count2, false_vars2)
                        cur_unsat2 += best_delta2
                if cur_unsat2 == 0:
                    for i, vi in enumerate(unassigned_vars):
                        assignment[vi] = assignment_r2[i]
                    return assignment

    return assignment
# EVOLVE-BLOCK-END
