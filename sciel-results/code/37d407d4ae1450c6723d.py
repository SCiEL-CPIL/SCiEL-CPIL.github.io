"""Planted 3-SAT solver: incremental-tracking WalkSAT with adaptive restarts and a time deadline."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    # ---- deadline guard (300 s hard limit, 270 s safety margin) ----
    start = time.time()
    DEADLINE = start + 270.0

    # ---- incremental clause tracking ----
    # For each clause store its (var, sign) pairs.
    # var_clauses[v] = list of clause indices containing v.
    # var_sign_in_clause[(v, c)] = sign of v in clause c (1 for +v, -1 for -v).
    clause_vars = []
    var_clauses = [[] for _ in range(n)]
    var_sign = {}
    for c, clause in enumerate(clauses):
        pairs = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pairs.append(v)
            var_clauses[v].append(c)
            var_sign[(v, c)] = sign
        clause_vars.append(pairs)

    m = len(clauses)

    def rebuild_sat_count(assignment):
        sc = [0] * m
        for c in range(m):
            cnt = 0
            for v in clause_vars[c]:
                if assignment[v] == (var_sign[(v, c)] == 1):
                    cnt += 1
            sc[c] = cnt
        return sc

    def update_flip(v, assignment, sat_count):
        """Flip variable v; update sat_count for clauses containing v.
        Returns list of (clause_idx, is_now_sat) for clauses whose status changed.
        """
        new_val = not assignment[v]
        assignment[v] = new_val
        changed = []
        for c in var_clauses[v]:
            sgn = var_sign[(v, c)]
            was_sat = sat_count[c] > 0
            if sgn == 1:
                if new_val:
                    sat_count[c] += 1
                else:
                    sat_count[c] -= 1
            else:
                if new_val:
                    sat_count[c] -= 1
                else:
                    sat_count[c] += 1
            is_sat = sat_count[c] > 0
            if was_sat != is_sat:
                changed.append((c, is_sat))
        return changed

    # ---- search ----
    best_unsat = None
    best_assignment = None

    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    sat_count = rebuild_sat_count(assignment)
    unsat_set = set(c for c in range(m) if sat_count[c] == 0)
    if not unsat_set:
        return assignment

    restart_steps = 2000
    max_restart = 20000
    noise = 0.45
    min_unsat_this_attempt = len(unsat_set)
    steps_since_improve = 0

    while time.time() < DEADLINE:
        if not unsat_set:
            best_unsat = 0
            best_assignment = assignment[:]
            break

        clause_idx = rng.choice(list(unsat_set))
        vars_in = clause_vars[clause_idx]

        if rng.random() < noise:
            v = rng.choice(vars_in)
        else:
            # greedy: flip the var that satisfies the most other currently-unsat clauses
            best_score = -1
            best_v = vars_in[0]
            for v in vars_in:
                new_val = (var_sign[(v, clause_idx)] == 1)  # value that satisfies this clause
                score = 0
                for c2 in var_clauses[v]:
                    if c2 == clause_idx:
                        continue
                    s2 = var_sign[(v, c2)]
                    cur_val = assignment[v]
                    lit_now = (cur_val == (s2 == 1))
                    lit_after = (new_val == (s2 == 1))
                    if not lit_now and lit_after:
                        score += 1
                    elif lit_now and not lit_after:
                        score -= 1
                if score > best_score:
                    best_score = score
                    best_v = v
            v = best_v

        changed = update_flip(v, assignment, sat_count)
        for (c, is_sat) in changed:
            if is_sat:
                unsat_set.discard(c)
            else:
                unsat_set.add(c)

        cur_unsat = len(unsat_set)
        if cur_unsat < min_unsat_this_attempt:
            min_unsat_this_attempt = cur_unsat
            steps_since_improve = 0
            if best_unsat is None or cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
                if cur_unsat == 0:
                    break
        else:
            steps_since_improve += 1

        if steps_since_improve >= restart_steps:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            sat_count = rebuild_sat_count(assignment)
            unsat_set = set(c for c in range(m) if sat_count[c] == 0)
            min_unsat_this_attempt = len(unsat_set)
            steps_since_improve = 0
            restart_steps = int(restart_steps * 1.3)
            if restart_steps > max_restart:
                restart_steps = max_restart
            if rng.random() < 0.3:
                noise = rng.uniform(0.2, 0.7)

    if best_unsat == 0 and best_assignment is not None:
        return best_assignment
    if best_assignment is not None:
        return best_assignment
    return assignment
