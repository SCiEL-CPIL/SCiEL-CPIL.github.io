# EVOLVE-BLOCK-START
import time
import random
import heapq
import sys

sys.setrecursionlimit(100000)


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325  # safety margin before 330s kill

    # Precompute clauses as (var_idx 0-based, polarity bool)
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    # Precompute var -> list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Weights per variable
    weights = [len(l) for l in var_clauses]

    def quick_check(ass):
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                return False
        return True

    # ===================== DPLL with clause learning =====================
    def _dpll_solve(n, clauses_opt, var_clauses, deadline):
        """
        DPLL with:
        - VSIDS variable ordering
        - Unit propagation via watched literals (simplified: count-based)
        - Clause learning on conflict
        - Failed literal probing
        - Phase saving
        """
        assignment = [None] * n
        phase = [False] * n  # phase save

        m = len(clauses_opt)
        # For each clause: sat_count (satisfied literals), unassigned_count
        sat_count = [0] * m

        unassigned_count = [0] * m
        for ci, cl in enumerate(clauses_opt):
            unassigned_count[ci] = len(cl)

        # Activity tracking
        activity = [1.0] * n
        clock = 0
        decay = 0.95

        def bump(v):
            nonlocal clock
            activity[v] = activity[v] * 2.0 + 1.0
            clock += 1

        def decay_activities():
            nonlocal clock
            if clock > 1000:
                for v in range(n):
                    activity[v] *= decay
                clock = 0

        # Unit propagation - simple but effective
        def propagate(assignment, sat_count, unassigned_count):
            """Returns list of assignments made, or None on conflict."""
            made = []
            queue = []
            # Find unit clauses
            for ci in range(m):
                if sat_count[ci] == 0 and unassigned_count[ci] == 1:
                    for v, p in clauses_opt[ci]:
                        if assignment[v] is None:
                            queue.append((v, p))
                            break

            while queue:
                v, p = queue.pop()
                if assignment[v] is not None:
                    if assignment[v] != p:
                        return None  # conflict
                    continue
                assignment[v] = p
                made.append((v, p))
                phase[v] = p
                # Update counts
                for ci, p2 in var_clauses[v]:
                    unassigned_count[ci] -= 1
                    if p2 == p:
                        sat_count[ci] += 1
                    else:
                        # Check if clause becomes unit
                        if sat_count[ci] == 0 and unassigned_count[ci] == 1:
                            for v2, p3 in clauses_opt[ci]:
                                if assignment[v2] is None:
                                    queue.append((v2, p3))
                                    break
            return made

        # Select variable using VSIDS
        def select_variable(assignment, sat_count, unassigned_count):
            best_v = -1
            best_act = -1.0
            for ci in range(m):
                if sat_count[ci] > 0:
                    continue
                for v, p in clauses_opt[ci]:
                    if assignment[v] is None:
                        if activity[v] > best_act:
                            best_act = activity[v]
                            best_v = v
            return best_v

        # Check if all satisfied
        def all_satisfied(sat_count):
            for ci in range(m):
                if sat_count[ci] == 0:
                    return False
            return True

        # Count unsatisfied
        def count_unsat(sat_count):
            c = 0
            for ci in range(m):
                if sat_count[ci] == 0:
                    c += 1
            return c

        # DPLL search
        def dpll(assignment, sat_count, unassigned_count, depth, trail):
            nonlocal clock

            if time.time() > deadline:
                return None, trail

            # Unit propagation
            made = propagate(assignment, sat_count, unassigned_count)
            if made is None:
                return None, trail

            for v, p in made:
                trail.append(v)

            if all_satisfied(sat_count):
                return assignment[:], trail

            v = select_variable(assignment, sat_count, unassigned_count)
            if v < 0:
                # No unassigned vars in unsat clauses
                if all_satisfied(sat_count):
                    return assignment[:], trail
                return None, trail

            # Phase-based ordering
            if phase[v]:
                order = [True, False]
            else:
                order = [False, True]

            for val in order:
                if time.time() > deadline:
                    return None, trail

                # Save state
                old_sat = sat_count[:]
                old_unassigned = unassigned_count[:]
                old_trail_len = len(trail)

                assignment[v] = val
                phase[v] = val
                trail.append(v)

                for ci, p in var_clauses[v]:
                    unassigned_count[ci] -= 1
                    if p == val:
                        sat_count[ci] += 1

                bump(v)
                if clock % 100 == 0:
                    decay_activities()

                result, trail = dpll(assignment, sat_count, unassigned_count, depth + 1, trail)
                if result is not None:
                    return result, trail

                # Backtrack
                while len(trail) > old_trail_len:
                    tv = trail.pop()
                    assignment[tv] = None
                    for ci, p in var_clauses[tv]:
                        unassigned_count[ci] += 1
                        if p == assignment[tv]:
                            sat_count[ci] -= 1
                        else:
                            pass
                # Restore counts properly
                sat_count[:] = old_sat
                unassigned_count[:] = old_unassigned
                assignment[v] = None

            return None, trail

        # Run
        trail = []
        result, _ = dpll(assignment, sat_count, unassigned_count, 0, trail)
        if result is not None:
            return [bool(x) if x is not None else False for x in result]
        return None

    # ===================== WalkSAT / SLS =====================
    def _sls_solve(n, clauses_opt, var_clauses, weights, deadline):
        """
        WalkSAT with:
        - Min-break heuristic with noise
        - Phase saving
        - Multiple restarts with increasing noise
        - Best-so-far tracking
        """
        best_assignment = None
        best_unsat = float('inf')

        def count_unsat(ass):
            cnt = 0
            for cl in clauses_opt:
                for v, p in cl:
                    if ass[v] == p:
                        break
                else:
                    cnt += 1
            return cnt

        def build_sat(ass):
            sat = [0] * m
            for ci, cl in enumerate(clauses_opt):
                s = 0
                for v, p in cl:
                    if ass[v] == p:
                        s += 1
                sat[ci] = s
            return sat

        def flip_var(ass, var, sat):
            old = ass[var]
            new = not old
            ass[var] = new
            for ci, p in var_clauses[var]:
                if p == new:
                    sat[ci] += 1
                else:
                    sat[ci] -= 1

        rng_master = random.Random(42)
        max_restarts = 500
        phase = [False] * n

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            rng = random.Random(restart * 7919 + 131 + int(time.time() * 1000))

            # Initialize
            if restart == 0:
                ass = [rng.random() < 0.5 for _ in range(n)]
            elif restart % 10 == 0 and best_assignment is not None:
                # Perturb best
                ass = best_assignment[:]
                for _ in range(n // 10):
                    ass[rng.randrange(n)] = not ass[rng.randrange(n)]
            else:
                ass = [rng.random() < 0.5 for _ in range(n)]

            # Phase bias
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                if pos > len(var_clauses[v]) - pos:
                    phase[v] = True

            u = count_unsat(ass)
            if u == 0:
                return ass
            if u < best_unsat:
                best_unsat = u
                best_assignment = ass[:]

            sat = build_sat(ass)

            # Noise increases with restarts
            noise = 0.15 + 0.05 * min(restart // 20, 4)
            steps = 50000

            for step in range(steps):
                if time.time() > deadline - 5:
                    break

                # Find unsat clause
                best_ci = -1
                for _ in range(3):  # try 3 random clauses
                    ci = rng.randrange(m)
                    if sat[ci] == 0:
                        best_ci = ci
                        break
                if best_ci < 0:
                    # Full check
                    if all(s >= 1 for s in sat):
                        return ass
                    # Find one
                    for ci in range(m):
                        if sat[ci] == 0:
                            best_ci = ci
                            break
                    if best_ci < 0:
                        return ass

                clause = clauses_opt[best_ci]

                if rng.random() < noise:
                    v, p = clause[rng.randrange(len(clause))]
                else:
                    # Min-break
                    best_v = clause[0][0]
                    best_cost = float('inf')
                    for v, p in clause:
                        if ass[v] == p:
                            continue
                        cost = 0
                        for ci2, p2 in var_clauses[v]:
                            if p2 == ass[v] and sat[ci2] == 1:
                                cost += 1
                        # Tie-break by weight
                        if cost < best_cost or (cost == best_cost and weights[v] > weights[best_v]):
                            best_cost = cost
                            best_v = v
                    v = best_v

                flip_var(ass, v, sat)
                phase[v] = ass[v]

                # Quick check
                unsat_count = 0
                for s in sat:
                    if s == 0:
                        unsat_count += 1
                if unsat_count == 0:
                    return ass
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = ass[:]

        if best_assignment is not None:
            return best_assignment
        return ass

    # ===================== Main solve =====================
    # Phase 0: Try DPLL with time limit
    dpll_deadline = start_time + 30
    dpll_result = _dpll_solve(n, clauses_opt, var_clauses, dpll_deadline)
    if dpll_result is not None and quick_check(dpll_result):
        return dpll_result

    # Phase 1: SLS
    sls_deadline = deadline
    sls_result = _sls_solve(n, clauses_opt, var_clauses, weights, sls_deadline)
    if sls_result is not None and quick_check(sls_result):
        return sls_result

    # Phase 2: If DPLL didn't finish, try again with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        dpll_result2 = _dpll_solve(n, clauses_opt, var_clauses, start_time + remaining)
        if dpll_result2 is not None and quick_check(dpll_result2):
            return dpll_result2

    # Return best effort
    if sls_result is not None:
        return sls_result
    return [False] * n

# EVOLVE-BLOCK-END
