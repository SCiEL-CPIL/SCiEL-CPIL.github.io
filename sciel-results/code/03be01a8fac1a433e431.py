# EVOLVE-BLOCK-START
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    rng = random.Random(0)
    m = len(clauses)

    # ---------- State ----------
    assignment = [None] * n  # None, 1, 0
    trail = []
    trail_lim = []
    level = [0] * n
    reason = [-1] * n

    # ---------- Watched literals ----------
    # watches[var][val] = list of clause indices
    watches = [[[] for _ in range(2)] for _ in range(n)]
    clause_watches = [[0, 0] for _ in range(m)]

    # Initialize watches for initial clauses
    for ci in range(m):
        clause = clauses[ci]
        # Pick two literals to watch
        l0, l1 = clause[0], clause[1]
        clause_watches[ci] = [l0, l1]
        v0 = abs(l0) - 1
        val0 = 1 if l0 > 0 else 0
        watches[v0][val0].append(ci)
        v1 = abs(l1) - 1
        val1 = 1 if l1 > 0 else 0
        watches[v1][val1].append(ci)

    # ---------- VSIDS ----------
    var_inc = 1.0 / 100.0
    activity = [1.0] * n

    def bump(v):
        nonlocal var_inc
        activity[v] += var_inc
        if activity[v] > 1e100:
            for i in range(n):
                activity[i] /= 1e100
            var_inc /= 1e100

    def decay():
        nonlocal var_inc
        var_inc *= 0.95

    # ---------- Clause DB ----------
    clause_db = [list(c) for c in clauses]

    # ---------- Assignment with watched literals ----------
    def assign_var(v, val, reason_ci):
        assignment[v] = val
        trail.append(v)
        level[v] = len(trail_lim)
        reason[v] = reason_ci
        bump(v)
        # Process watches
        false_val = 1 - val
        for ci in list(watches[v][false_val]):
            l0 = clause_watches[ci][0]
            l1 = clause_watches[ci][1]
            # Which watch is on v with false_val?
            if abs(l0) - 1 == v and (1 if l0 > 0 else 0) == false_val:
                other = l1
                watches[v][false_val].remove(ci)
                ov = abs(other) - 1
                oval = 1 if other > 0 else 0
                if assignment[ov] == 1 - oval:
                    return ci  # conflict
                elif assignment[ov] is None:
                    watches[ov][oval].append(ci)
                    clause_watches[ci][0] = other
            else:
                other = l0
                watches[v][false_val].remove(ci)
                ov = abs(other) - 1
                oval = 1 if other > 0 else 0
                if assignment[ov] == 1 - oval:
                    return ci  # conflict
                elif assignment[ov] is None:
                    watches[ov][oval].append(ci)
                    clause_watches[ci][1] = other
        return -1

    def propagate():
        # Find initial unit clauses
        queue = []
        for ci in range(len(clause_db)):
            clause = clause_db[ci]
            if len(clause) == 0:
                return ci
            unassigned = 0
            false_count = 0
            for lit in clause:
                v = abs(lit) - 1
                val = 1 if lit > 0 else 0
                if assignment[v] is None:
                    unassigned += 1
                elif assignment[v] == 1 - val:
                    false_count += 1
            if unassigned == 1 and false_count == len(clause) - 1:
                for lit in clause:
                    v = abs(lit) - 1
                    val = 1 if lit > 0 else 0
                    if assignment[v] is None:
                        queue.append((v, val, ci))
                        break
        while queue:
            v, val, ci = queue.pop()
            if assignment[v] is not None:
                continue
            conflict_ci = assign_var(v, val, ci)
            if conflict_ci != -1:
                return conflict_ci
            # Find new unit clauses
            false_val = 1 - val
            for ci2 in list(watches[v][false_val]):
                l0 = clause_watches[ci2][0]
                l1 = clause_watches[ci2][1]
                if abs(l0) - 1 == v and (1 if l0 > 0 else 0) == false_val:
                    other = l1
                else:
                    other = l0
                ov = abs(other) - 1
                oval = 1 if other > 0 else 0
                if assignment[ov] is None:
                    queue.append((ov, oval, ci2))
        return -1

    def backtrack_to(old_trail_len):
        while len(trail) > old_trail_len:
            v = trail.pop()
            assignment[v] = None
            level[v] = 0
            reason[v] = -1

    # ---------- Failed literal probing ----------
    def probe_failed_literals():
        decisions = []
        for v in range(n):
            if assignment[v] is not None:
                continue
            for val in (0, 1):
                old_len = len(trail)
                conflict = assign_var(v, val, -1)
                if conflict == -1:
                    conflict = propagate()
                backtrack_to(old_len)
                if conflict != -1:
                    decisions.append((v, 1 - val))
                    break
        for v, val in decisions:
            if assignment[v] is not None:
                continue
            conflict = assign_var(v, val, -1)
            if conflict == -1:
                conflict = propagate()
            if conflict != -1:
                return False
        return True

    # ---------- Decision ----------
    def pick_var():
        best = -1
        best_act = -1
        for v in range(n):
            if assignment[v] is None:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best = v
        return best

    def decide():
        v = pick_var()
        if v == -1:
            return None
        true_count = 0
        false_count = 0
        for clause in clause_db:
            for lit in clause:
                if abs(lit) - 1 == v:
                    if lit > 0:
                        true_count += 1
                    else:
                        false_count += 1
                    break
        val = 1 if true_count >= false_count else 0
        return (v, val)

    # ---------- Conflict analysis (simplified) ----------
    def analyze(conflict_ci):
        clause = clause_db[conflict_ci]
        levels = []
        for lit in clause:
            v = abs(lit) - 1
            if assignment[v] is not None:
                levels.append(level[v])
        if not levels:
            return None, 0
        max_level = max(levels)
        second = 0
        for l in levels:
            if l < max_level and l > second:
                second = l
        return list(clause), second

    # ---------- DPLL ----------
    def dpll():
        conflict = propagate()
        if conflict != -1:
            return None
        if not probe_failed_literals():
            return None
        conflict = propagate()
        if conflict != -1:
            return None
        if all(assignment[v] is not None for v in range(n)):
            return assignment[:]
        while True:
            if all(assignment[v] is not None for v in range(n)):
                return assignment[:]
            decision = decide()
            if decision is None:
                return None
            v, val = decision
            trail_lim.append(len(trail))
            conflict = assign_var(v, val, -1)
            if conflict == -1:
                conflict = propagate()
            if conflict == -1:
                decay()
                continue
            learned, bt_level = analyze(conflict)
            if learned is None:
                return None
            clause_db.append(learned)
            # Add watches for learned clause
            ci_new = len(clause_db) - 1
            if len(learned) >= 2:
                l0, l1 = learned[0], learned[1]
                clause_watches.append([l0, l1])
                v0 = abs(l0) - 1
                val0 = 1 if l0 > 0 else 0
                watches[v0][val0].append(ci_new)
                v1 = abs(l1) - 1
                val1 = 1 if l1 > 0 else 0
                watches[v1][val1].append(ci_new)
            else:
                clause_watches.append([learned[0], learned[0]])
                v0 = abs(learned[0]) - 1
                val0 = 1 if learned[0] > 0 else 0
                watches[v0][val0].append(ci_new)
                watches[v0][val0].append(ci_new)
            # Backtrack
            while len(trail_lim) > bt_level + 1:
                old_len = trail_lim.pop()
                backtrack_to(old_len)
            # Assign the unit literal of the learned clause
            for lit in learned:
                vv = abs(lit) - 1
                oval = 1 if lit > 0 else 0
                if assignment[vv] is None:
                    conflict = assign_var(vv, oval, ci_new)
                    if conflict == -1:
                        conflict = propagate()
                    break
            if conflict != -1:
                return None

    # ---------- Verify ----------
    def verify(asm):
        for clause in clauses:
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                val = 1 if lit > 0 else 0
                if asm[v] == val:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # ---------- Try DPLL ----------
    result = dpll()
    if result is not None and verify(result):
        return result

    # ---------- Fallback: WalkSAT ----------
    assignment = [rng.random() < 0.5 for _ in range(n)]
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    for _ in range(50000):
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        best_lit = None
        best_gain = -1
        for lit in clause:
            v = abs(lit) - 1
            val = 1 if lit > 0 else 0
            if assignment[v] == val:
                continue
            gain = 0
            for c in clauses:
                if not satisfied(c):
                    for l2 in c:
                        if abs(l2) - 1 == v and (1 if l2 > 0 else 0) == val:
                            gain += 1
                            break
            if gain > best_gain:
                best_gain = gain
                best_lit = lit
        if best_lit is None:
            best_lit = rng.choice(clause)
        v = abs(best_lit) - 1
        assignment[v] = not assignment[v]
    if verify(assignment):
        return assignment

    # ---------- Final fallback ----------
    for _ in range(1000):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        if verify(assignment):
            return assignment
    return assignment
# EVOLVE-BLOCK-END
