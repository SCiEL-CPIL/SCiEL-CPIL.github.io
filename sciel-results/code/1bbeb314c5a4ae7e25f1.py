# EVOLVE-BLOCK-START
import time
import random
import heapq


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(clauses_opt, var_clauses, n, m, deadline):
    if n == 0:
        return []
    assign = [-1] * n
    trail = []
    level = [0] * n
    learnt = []

    activity = [1.0] * n
    var_heap = [(-activity[v], v) for v in range(n)]
    heapq.heapify(var_heap)
    phase = [True] * n

    def propagate(queue):
        while queue:
            ci, is_learnt = queue.pop()
            if is_learnt:
                if ci >= len(learnt):
                    continue
                cl = learnt[ci]
            else:
                if ci >= m:
                    continue
                cl = clauses_opt[ci]

            satisfied = False
            unfixed = []
            for v, p in cl:
                if assign[v] != -1:
                    if assign[v] == (1 if p else 0):
                        satisfied = True
                        break
                else:
                    unfixed.append((v, p))

            if satisfied:
                continue
            if not unfixed:
                return False, (ci, is_learnt)
            if len(unfixed) == 1:
                v, p = unfixed[0]
                if assign[v] == -1:
                    assign[v] = 1 if p else 0
                    trail.append(v)
                    level[v] = len(trail)
                    for ci2, p2 in var_clauses[v]:
                        queue.append((ci2, False))
                    for li in range(len(learnt)):
                        for v2, p2 in learnt[li]:
                            if v2 == v:
                                queue.append((li, True))
                                break
        return True, None

    def analyze_conflict(conf_clause):
        ci, is_learnt = conf_clause
        if is_learnt:
            cl = learnt[ci]
        else:
            cl = clauses_opt[ci]

        learnt_clause = []
        max_level = 0
        for v, p in cl:
            if assign[v] != -1:
                learnt_clause.append((v, p))
                if level[v] > max_level:
                    max_level = level[v]

        if len(learnt_clause) > 1:
            seen = set()
            unique = []
            for v, p in learnt_clause:
                if (v, p) not in seen:
                    seen.add((v, p))
                    unique.append((v, p))
            learnt.append(unique)
            for v, p in unique:
                activity[v] += 1.0
            return unique, max_level
        return None, 0

    q = [(ci, False) for ci in range(m)]
    ok, conf = propagate(q)
    if not ok:
        return None

    while time.time() < deadline:
        v = -1
        while var_heap:
            neg_act, cand = var_heap[0]
            if assign[cand] == -1:
                v = cand
                heapq.heappop(var_heap)
                break
            else:
                heapq.heappop(var_heap)

        if v == -1:
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    return None
            return [bool(assign[i]) for i in range(n)]

        def try_branch(val):
            assign[v] = val
            trail.append(v)
            level[v] = len(trail)

            q2 = [(ci2, False) for ci2, p2 in var_clauses[v]]
            for li in range(len(learnt)):
                for v2, p2 in learnt[li]:
                    if v2 == v:
                        q2.append((li, True))
                        break

            ok2, conf2 = propagate(q2)
            if ok2:
                for ci in range(m):
                    cl = clauses_opt[ci]
                    for vi, pi in cl:
                        if assign[vi] == (1 if pi else 0):
                            break
                    else:
                        break
                else:
                    return [bool(assign[i]) for i in range(n)]
            else:
                learnt_clause, bt_level = analyze_conflict(conf2)
                while trail and level[trail[-1]] >= bt_level:
                    vv = trail.pop()
                    assign[vv] = -1
                    level[vv] = 0
                return None

        val = 1 if phase[v] else 0
        res = try_branch(val)
        if res is not None:
            return res
        if not trail:
            return None

        val2 = 1 - val
        assign[v] = val2
        trail.append(v)
        level[v] = len(trail)
        phase[v] = not phase[v]

        q2 = [(ci2, False) for ci2, p2 in var_clauses[v]]
        for li in range(len(learnt)):
            for v2, p2 in learnt[li]:
                if v2 == v:
                    q2.append((li, True))
                    break

        ok2, conf2 = propagate(q2)
        if ok2:
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    break
            else:
                return [bool(assign[i]) for i in range(n)]
        else:
            learnt_clause, bt_level = analyze_conflict(conf2)
            while trail and level[trail[-1]] >= bt_level:
                vv = trail.pop()
                assign[vv] = -1
                level[vv] = 0
            if not trail:
                return None

    return None


def _sls_solve(clauses_opt, var_clauses, n, m, deadline, seed=0):
    def build_state(ass):
        sat = [0] * m
        for ci, cl in enumerate(clauses_opt):
            s = 0
            for v, p in cl:
                if ass[v] == p:
                    s += 1
            sat[ci] = s
        return sat

    def flip_var(ass, var, sat):
        new = not ass[var]
        ass[var] = new
        for ci, p in var_clauses[var]:
            if p == new:
                sat[ci] += 1
            else:
                sat[ci] -= 1

    def unit_propagate(ass, remaining_vars, n_remaining, sat):
        changed = True
        while changed and n_remaining > 0:
            changed = False
            for ci in range(m):
                if sat[ci] > 0:
                    continue
                cl = clauses_opt[ci]
                unfixed = []
                for v, p in cl:
                    if v not in remaining_vars:
                        continue
                    unfixed.append((v, p))
                if len(unfixed) == 1:
                    v, p = unfixed[0]
                    if ass[v] != p:
                        flip_var(ass, v, sat)
                    remaining_vars.discard(v)
                    n_remaining -= 1
                    changed = True
        return True

    weights = [len(l) for l in var_clauses]
    best_assignment = None
    best_unsat = float('inf')

    def track(ass, sat):
        nonlocal best_assignment, best_unsat
        u = sum(1 for s in sat if s == 0)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        return u

    max_restarts = 120
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(seed * 1000003 + restart * 7919 + 131)
        ass = [rng.random() < 0.5 for _ in range(n)]

        if restart % 5 == 0:
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                ass[v] = (pos >= len(var_clauses[v]) - pos) if len(var_clauses[v]) else False

        sat = build_state(ass)
        if all(s > 0 for s in sat):
            return ass

        remaining_vars = set(range(n))
        unit_propagate(ass, remaining_vars, n, sat)
        if all(s > 0 for s in sat):
            return ass

        track(ass, sat)
        mode = restart % 3
        noise = 0.2 + 0.1 * (restart % 4)

        steps = 12000
        for step in range(steps):
            if time.time() > deadline:
                break

            best_ci = -1
            best_len = 0
            for ci in range(m):
                if sat[ci] == 0:
                    ln = len(clauses_opt[ci])
                    if ln > best_len:
                        best_len = ln
                        best_ci = ci
                        if ln >= 3:
                            break
            if best_ci < 0:
                return ass

            clause = clauses_opt[best_ci]

            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                best_v = clause[0][0]
                best_score = None
                for v, p in clause:
                    if ass[v] == p:
                        continue
                    cost = 0
                    for ci2, p2 in var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    if mode == 0:
                        score = (cost, -weights[v])
                    elif mode == 1:
                        pos = sum(1 for ci2, p2 in var_clauses[v] if p2)
                        phase_match = 1 if (p == (pos >= len(var_clauses[v]) - pos)) else 0
                        score = (cost, -phase_match, -weights[v])
                    else:
                        score = (cost, -weights[v])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_v = v
                v = best_v
            flip_var(ass, v, sat)

            if all(s > 0 for s in sat):
                return ass

        track(ass, sat)

    if best_assignment is not None:
        return best_assignment
    return ass


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325

    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Find connected components via union-find over variables sharing clauses
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cl in clauses_opt:
        if len(cl) > 1:
            first = cl[0][0]
            for v, p in cl[1:]:
                union(first, v)

    comps = {}
    for v in range(n):
        r = find(v)
        comps.setdefault(r, []).append(v)

    comp_list = list(comps.values())

    # Sort components: try small ones first (cheap wins), but keep big ones for DPLL
    comp_list.sort(key=len)

    full_assign = [False] * n

    # For each component, solve independently
    comp_deadline = start_time + 300
    per_comp_time = (comp_deadline - start_time) / max(1, len(comp_list))

    # Solve each component
    for ci, comp_vars in enumerate(comp_list):
        if time.time() > comp_deadline:
            break
        k = len(comp_vars)
        # Map global -> local
        v2l = {v: i for i, v in enumerate(comp_vars)}
        l2v = comp_vars

        # Local clauses
        local_clauses = []
        for cl in clauses_opt:
            if any(v in v2l for v, p in cl):
                local_clauses.append([(v2l[v], p) for v, p in cl])
        lm = len(local_clauses)

        local_var_clauses = [[] for _ in range(k)]
        for lci, cl in enumerate(local_clauses):
            for v, p in cl:
                local_var_clauses[v].append((lci, p))

        # Allocate time: bigger components get more
        budget = min(per_comp_time, 300 - (time.time() - start_time))
        if k <= 1:
            # Trivial: assign to satisfy
            if local_clauses:
                for cl in local_clauses:
                    full_assign[l2v[cl[0][0]]] = cl[0][1]
            continue

        # Try DPLL for small components
        if k <= 60:
            dpll_dead = start_time + min(15, budget * 0.5)
            res = _dpll_solve(local_clauses, local_var_clauses, k, lm, dpll_dead)
            if res is not None:
                for i, v in enumerate(l2v):
                    full_assign[v] = res[i]
                continue

        # SLS for this component
        sls_dead = start_time + min(60, budget)
        res = _sls_solve(local_clauses, local_var_clauses, k, lm, sls_dead, seed=ci)
        if res is not None:
            for i, v in enumerate(l2v):
                full_assign[v] = res[i]

    # Verify full assignment
    for cl in clauses_opt:
        for v, p in cl:
            if full_assign[v] == p:
                break
        else:
            # Fallback: full-instance SLS
            if time.time() < deadline:
                res = _sls_solve(clauses_opt, var_clauses, n, m, deadline, seed=42)
                if res is not None:
                    return res
            return full_assign
    return full_assign
# EVOLVE-BLOCK-END
