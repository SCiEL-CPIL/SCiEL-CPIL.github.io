"""Planted 3-SAT solver: component decomposition + CDCL (DPLL + clause learning + VSIDS)."""
# EVOLVE-BLOCK-START
import time
import random
import heapq


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 280.0

    # Convert to 0-based signed literals: (var, sign)
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Build variable-clause incidence for component detection
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, sign in cv:
            var_to_clauses[var].append(ci)

    # Find connected components via BFS on variable-clause bipartite graph
    visited_vars = [False] * n
    visited_clauses = [False] * m
    components = []

    for v in range(n):
        if visited_vars[v]:
            continue
        comp_vars = set()
        comp_clauses = set()
        stack = [(v, 'var')]
        visited_vars[v] = True

        while stack:
            node, node_type = stack.pop()
            if node_type == 'var':
                comp_vars.add(node)
                for ci in var_to_clauses[node]:
                    if not visited_clauses[ci]:
                        visited_clauses[ci] = True
                        stack.append((ci, 'clause'))
            else:
                comp_clauses.add(node)
                for var, sign in clause_vars[node]:
                    if not visited_vars[var]:
                        visited_vars[var] = True
                        stack.append((var, 'var'))

        components.append((list(comp_vars), list(comp_clauses)))

    # Sort components by size (largest first)
    components.sort(key=lambda x: -len(x[0]))

    assignment = [False] * n
    rng = random.Random(42)

    for comp_idx, (comp_vars, comp_clauses) in enumerate(components):
        if time.time() > deadline - 3.0:
            break

        num_vars = len(comp_vars)
        num_clauses = len(comp_clauses)
        time_budget = min(60.0, max(0.3, 150.0 * num_vars / n))
        comp_start = time.time()

        # Map component variables to local indices
        var_map = {v: i for i, v in enumerate(comp_vars)}
        local_n = num_vars

        # Local clauses as lists of (var, sign)
        local_clauses = []
        for ci in comp_clauses:
            local_c = []
            for var, sign in clause_vars[ci]:
                local_c.append((var_map[var], sign))
            local_clauses.append(local_c)

        # Local variable-to-clause mapping
        local_var_to_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for var, sign in lc:
                local_var_to_clauses[var].append(ci)

        comp_assignment = _solve_cdcl(
            local_n, local_clauses, local_var_to_clauses,
            time_budget, rng, comp_start
        )

        for i, v in enumerate(comp_vars):
            assignment[v] = comp_assignment[i]

    return assignment


def _solve_cdcl(n, clauses, var_to_clauses, time_budget, rng, start_time):
    """CDCL solver. Returns list of bools of length n."""
    deadline = start_time + time_budget
    m_orig = len(clauses)

    # State
    val = [0] * n            # 0=unassigned, 1=True, -1=False
    trail = []
    trail_lim = []
    reason = [-1] * n
    level = [0] * n

    # Clause DB
    db = [list(c) for c in clauses]
    # Watches: watches[var] = list of (clause_idx, other_var, other_sign)
    # meaning clause watches (var, sign_of_var_in_clause) and (other_var, other_sign)
    # We store for each watch entry: (clause_idx, polarity_of_this_var, other_var, polarity_of_other)
    watches = [[] for _ in range(n)]

    def clause_polarity(ci, v):
        for (wv, ws) in db[ci]:
            if wv == v:
                return ws
        return None

    def add_watches(ci):
        c = db[ci]
        if len(c) >= 2:
            (v1, s1), (v2, s2) = c[0], c[1]
            watches[v1].append((ci, s1, v2, s2))
            watches[v2].append((ci, s2, v1, s1))
        elif len(c) == 1:
            (v1, s1) = c[0]
            watches[v1].append((ci, s1, -1, False))

    for i in range(len(db)):
        add_watches(i)

    # VSIDS
    activity = [1.0] * n
    var_inc = 1.0
    var_decay = 0.95
    heap = [(-activity[i], i) for i in range(n)]
    heapq.heapify(heap)

    clause_act = [0.0] * len(db)
    clause_inc = 1.0

    conflicts = 0
    next_restart = 100
    cur_lvl = 0

    def bump_var(v):
        nonlocal var_inc
        activity[v] += var_inc
        if activity[v] > 1e100:
            for i in range(n):
                activity[i] *= 1e-100
            var_inc *= 1e-100
        heapq.heappush(heap, (-activity[v], v))

    def bump_clause(ci):
        nonlocal clause_inc
        clause_act[ci] += clause_inc
        if clause_act[ci] > 1e100:
            for i in range(len(clause_act)):
                clause_act[i] *= 1e-100
            clause_inc *= 1e-100

    def is_true(v, s):
        return val[v] != 0 and (val[v] == 1) == s

    def is_false(v, s):
        return val[v] != 0 and (val[v] == 1) != s

    def assign(v, s, lvl, rsn):
        val[v] = 1 if s else -1
        trail.append(v)
        trail_lim.append(lvl)
        level[v] = lvl
        reason[v] = rsn

    def unassign_to(lvl):
        while trail and trail_lim[-1] > lvl:
            v = trail.pop()
            trail_lim.pop()
            val[v] = 0
            reason[v] = -1

    def propagate():
        """Returns None on success, or conflict clause index."""
        while True:
            conflict = None
            # Scan for unit clauses via watches
            for v in range(n):
                wlist = watches[v]
                i = 0
                while i < len(wlist):
                    ci, s_w, other_v, s_other = wlist[i]
                    if db[ci] is None:
                        wlist.pop(i)
                        continue
                    if is_false(v, s_w):
                        # Find new watch
                        c = db[ci]
                        new_watch = None
                        for (wv, ws) in c:
                            if wv == v or wv == other_v:
                                continue
                            if val[wv] == 0:
                                new_watch = (wv, ws)
                                break
                        if new_watch is None:
                            # Unit or conflict
                            if other_v < 0:
                                # Clause had length 1 and its only literal is
                                # false -> conflict
                                conflict = ci
                                wlist.pop(i)
                                break
                            if val[other_v] == 0:
                                assign(other_v, s_other, cur_lvl, ci)
                                wlist.pop(i)
                                # Remove the other watch entry for this clause
                                ow = watches[other_v]
                                ow[:] = [e for e in ow if e[0] != ci]
                                i -= 1  # because we popped, re-examine same index
                                # Actually after popping, i points to next; but we
                                # want to continue scanning. Just continue.
                                continue
                            else:
                                conflict = ci
                                wlist.pop(i)
                                break
                        else:
                            nv, ns = new_watch
                            wlist.pop(i)
                            watches[nv].append((ci, ns, other_v, s_other))
                            continue
                    else:
                        i += 1
                if conflict is not None:
                    return conflict
            return None

    def analyze(ci):
        """Returns (learned_clause, backjump_lvl)."""
        seen = set()
        learnt = []
        p = None
        max_lvl = 0
        first = True

        while True:
            c = db[ci]
            # Collect literals
            for (v, s) in c:
                if p is not None and v == p:
                    continue
                if first and val[v] == 0:
                    # This is the asserting literal candidate at current level
                    # but we haven't started yet; skip for now
                    pass
                if val[v] == 0:
                    # Unassigned -> this is the asserting literal (at cur_lvl)
                    learnt.append((v, s))
                    break
                if v in seen:
                    continue
                seen.add(v)
                learnt.append((v, s))
                if level[v] > max_lvl:
                    max_lvl = level[v]
                bump_clause(ci)
                for (wv, ws) in c:
                    if wv == p or wv in seen or val[wv] == 0:
                        continue
                    bump_var(wv)
                p = v
                ci = reason[v]
                first = False
                break
            else:
                # No unassigned literal found in this clause
                if p is None:
                    # Shouldn't happen
                    break
                # Move to next clause
                continue

        # Simplify: remove literals implied by lower-level reasons
        simplified = []
        for (v, s) in learnt:
            if val[v] == 0:
                simplified.append((v, s))
            else:
                if level[v] == max_lvl:
                    simplified.append((v, s))

        if len(simplified) == 0:
            return None, 0

        # Find asserting literal (at max_lvl)
        assert_lit = None
        for (v, s) in simplified:
            if level[v] == max_lvl:
                assert_lit = (v, s)
                break
        if assert_lit is None:
            assert_lit = simplified[-1]

        # Build learned clause: negate asserting literal, keep others
        learnt_clause = []
        for (v, s) in simplified:
            if (v, s) == assert_lit:
                learnt_clause.append((v, not s))
            else:
                learnt_clause.append((v, s))

        return learnt_clause, max_lvl

    # Main loop
    solution = None
    while time.time() < deadline:
        # Restart
        if conflicts >= next_restart and cur_lvl > 0:
            for i in range(n):
                activity[i] *= var_decay
            for i in range(len(clause_act)):
                clause_act[i] *= 0.999
            var_inc /= var_decay
            heap = [(-activity[i], i) for i in range(n)]
            heapq.heapify(heap)
            # Delete weak learned clauses
            for ci in range(m_orig, len(db)):
                if db[ci] is not None and len(db[ci]) > 2 and clause_act[ci] < 0.5:
                    c = db[ci]
                    for w in watches:
                        w[:] = [e for e in w if e[0] != ci]
                    db[ci] = None
            next_restart = int(next_restart * 1.5) + 50
            conflicts = 0
            unassign_to(0)
            cur_lvl = 0
            continue

        # Check if all assigned
        if len(trail) == n:
            ok = True
            for c in clauses:
                sat = False
                for (v, s) in c:
                    if val[v] != 0 and (val[v] == 1) == s:
                        sat = True
                        break
                if not sat:
                    ok = False
                    break
            if ok:
                solution = [val[i] == 1 for i in range(n)]
                break
            break

        # Propagate
        conflict_ci = propagate()
        if conflict_ci is not None:
            learnt, backjump_lvl = analyze(conflict_ci)
            if learnt is None:
                break
            db.append(learnt)
            add_watches(len(db) - 1)
            bump_clause(len(db) - 1)
            unassign_to(backjump_lvl)
            cur_lvl = backjump_lvl
            # Assert
            for (v, s) in learnt:
                if level[v] == backjump_lvl and val[v] == 0:
                    assign(v, s, backjump_lvl, len(db) - 1)
                    break
            conflicts += 1
            continue

        # Decision
        while True:
            a, v = heapq.heappop(heap)
            if val[v] == 0:
                break
        # Simple heuristic
        true_c = 0
        false_c = 0
        for ci in var_to_clauses[v]:
            c = clauses[ci]
            sat = False
            for (wv, ws) in c:
                if wv == v:
                    continue
                if val[wv] != 0 and (val[wv] == 1) == ws:
                    sat = True
                    break
            if sat:
                continue
            for (wv, ws) in c:
                if wv == v:
                    if ws:
                        true_c += 1
                    else:
                        false_c += 1
                    break
        if true_c > false_c:
            s = True
        elif false_c > true_c:
            s = False
        else:
            s = rng.random() < 0.5
        cur_lvl += 1
        assign(v, s, cur_lvl, -1)

    if solution is not None:
        return solution

    # Fallback
    result = []
    for i in range(n):
        if val[i] != 0:
            result.append(val[i] == 1)
        else:
            result.append(rng.random() < 0.5)
    return result
# EVOLVE-BLOCK-END
