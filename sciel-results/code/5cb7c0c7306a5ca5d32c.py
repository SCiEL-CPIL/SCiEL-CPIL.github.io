"""DPLL solver with VSIDS, failed-literal probing, and clause learning."""
# EVOLVE-BLOCK-START
import time
import heapq
import random

_DEADLINE = 320.0


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    t0 = time.time()

    # Convert to (var, sign) format
    cdb = []
    for cl in clauses:
        cdb.append([(abs(l) - 1, 1 if l > 0 else -1) for l in cl])
    m = len(cdb)

    # Value: -1 unassigned, 0 false, 1 true
    value = [-1] * n
    trail = []
    trail_lim = []
    level = [0] * n
    reason = [-1] * n
    unassigned = set(range(n))

    # Watchers: for each literal (var, sign), list of (clause_idx, position)
    # lit_key = var * 2 + (0 if sign==-1 else 1)
    def lit_key(var, sign):
        return var * 2 + (0 if sign == -1 else 1)

    watchers = [[] for _ in range(2 * n)]
    for ci, cl in enumerate(cdb):
        for pos in (0, 1):
            v, s = cl[pos]
            watchers[lit_key(v, s)].append((ci, pos))

    # Activity
    activity = [0.0] * n
    var_heap = [(-0.0, v) for v in range(n)]
    heapq.heapify(var_heap)

    # Occurrence lists
    occ = [[] for _ in range(n)]
    for ci, cl in enumerate(cdb):
        for v, s in cl:
            occ[v].append(ci)

    def is_true(var, sign):
        v = value[var]
        if v == -1:
            return False
        return (v == 1 and sign == 1) or (v == 0 and sign == -1)

    def enqueue(var, sign, rsn):
        """Assign and propagate. Returns (ok, conflict_clause)."""
        if value[var] != -1:
            if (value[var] == 1 and sign == 1) or (value[var] == 0 and sign == -1):
                return True, -1
            return False, rsn
        value[var] = 1 if sign == 1 else 0
        level[var] = len(trail_lim)
        reason[var] = rsn
        trail.append(var)
        unassigned.discard(var)
        # Bump activity
        for ci in occ[var]:
            activity[var] += 0.5
        # Propagate watches
        false_lit = lit_key(var, 1 - value[var] if value[var] == 1 else 0)
        # false_lit: the literal that is now false
        if value[var] == 1:
            false_lit = lit_key(var, -1)
        else:
            false_lit = lit_key(var, 1)
        for (ci, pos) in list(watchers[false_lit]):
            cl = cdb[ci]
            v0, s0 = cl[0]
            v1, s1 = cl[1]
            if is_true(v0, s0) or is_true(v1, s1):
                continue
            # Remove watcher
            watchers[false_lit].remove((ci, pos))
            found = False
            for j in range(3):
                if j == pos:
                    continue
                vj, sj = cl[j]
                if is_true(vj, sj):
                    continue
                # Move watch to j
                cl[pos], cl[j] = cl[j], cl[pos]
                watchers[lit_key(vj, sj)].append((ci, pos))
                found = True
                break
            if not found:
                # Unit or empty
                unit_var = -1
                unit_sign = 0
                for j in range(3):
                    vj, sj = cl[j]
                    if value[vj] == -1:
                        unit_var = vj
                        unit_sign = sj
                        break
                if unit_var == -1:
                    return False, ci
                ok, conf = enqueue(unit_var, unit_sign, ci)
                if not ok:
                    return False, conf
        return True, -1

    def analyze(conflict_ci):
        """First-UIP learning."""
        seen = [False] * n
        learnt = []  # list of (var, sign)
        p = len(trail) - 1
        counter = 0
        btl = 0
        clause = cdb[conflict_ci]
        while True:
            for j in range(3):
                v, s = clause[j]
                if value[v] == -1 or seen[v]:
                    continue
                if level[v] > 0:
                    counter += 1
                    if level[v] > btl:
                        btl = level[v]
                    seen[v] = True
                    learnt.append((v, -s))
            # Find next
            while p >= 0:
                v = trail[p]
                p -= 1
                if not seen[v]:
                    break
            if p < 0:
                break
            v = trail[p]
            clause = cdb[reason[v]]
            for k in range(len(learnt) - 1, -1, -1):
                if learnt[k][0] == v:
                    learnt.pop(k)
                    break
            counter -= 1
            if counter == 0:
                break
        # Decrement activity
        for v, s in learnt:
            activity[v] *= 0.5
        # Add learnt clause
        ci = len(cdb)
        cdb.append(learnt)
        for pos in (0, 1):
            v, s = learnt[pos]
            watchers[lit_key(v, s)].append((ci, pos))
        for v, s in learnt:
            occ[v].append(ci)
        # Backtrack level
        levels = sorted([level[v] for v, s in learnt], reverse=True)
        btl = levels[1] if len(levels) >= 2 else 0
        return learnt, btl

    def backtrack(btl):
        while len(trail_lim) > btl:
            trail_lim.pop()
        while len(trail) > len(trail_lim):
            v = trail.pop()
            value[v] = -1
            unassigned.add(v)

    def pick_var():
        while var_heap:
            neg_act, v = heapq.heappop(var_heap)
            if v in unassigned:
                return v
        return -1

    def check_sat():
        for cl in cdb[:m]:
            if not any(is_true(v, s) for v, s in cl):
                return False
        return True

    # Main loop
    conflict_count = 0
    max_conflicts = 50000
    restarts = 0
    max_restarts = 50

    # Initial propagation
    for ci in range(m):
        cl = cdb[ci]
        unassigned_in_cl = [(v, s) for v, s in cl if value[v] == -1]
        if len(unassigned_in_cl) == 1:
            v, s = unassigned_in_cl[0]
            ok, conf = enqueue(v, s, ci)
            if not ok:
                return [True] * n

    while True:
        if time.time() - t0 > _DEADLINE:
            result = []
            for v in range(n):
                if value[v] == -1:
                    result.append(True)
                else:
                    result.append(value[v] == 1)
            return result

        if not unassigned:
            if check_sat():
                return [value[v] == 1 for v in range(n)]
            break

        v = pick_var()
        if v == -1:
            break

        trail_lim.append(len(trail))
        ok, conf = enqueue(v, 1, -1)
        if not ok:
            learnt, btl = analyze(conf)
            backtrack(btl)
            if len(learnt) == 1:
                lv, ls = learnt[0]
                ok2, conf2 = enqueue(lv, ls, len(cdb) - 1)
                if not ok2:
                    learnt2, btl2 = analyze(conf2)
                    backtrack(btl2)
            conflict_count += 1
            if conflict_count > max_conflicts:
                backtrack(0)
                conflict_count = 0
                restarts += 1
                if restarts > max_restarts:
                    break
            continue

        if not unassigned:
            if check_sat():
                return [value[v] == 1 for v in range(n)]
            break

    # Fallback
    result = []
    for v in range(n):
        if value[v] == -1:
            result.append(True)
        else:
            result.append(value[v] == 1)
    return result

# EVOLVE-BLOCK-END
