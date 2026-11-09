"""Synthetic planted 3-SAT solver (50 vars) with DPLL pre-pass + annealed WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 300.0
_MARGIN = 25.0


def _count_unsat(assignment, clauses):
    cnt = 0
    for c in clauses:
        sat = False
        for l in c:
            if assignment[abs(l) - 1] == (l > 0):
                sat = True
                break
        if not sat:
            cnt += 1
    return cnt


def _check(assignment, clauses):
    return _count_unsat(assignment, clauses) == 0


def _dpll(instance):
    n, clauses = instance["n"], instance["clauses"]
    clauses = [list(c) for c in clauses]
    assignment = [None] * n

    def unit_propagate(clauses, assignment):
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                unassigned = [l for l in c if assignment[abs(l) - 1] is None]
                sat = False
                for l in c:
                    a = assignment[abs(l) - 1]
                    if a is not None and a == (l > 0):
                        sat = True
                        break
                if sat:
                    continue
                if len(unassigned) == 0:
                    return clauses, assignment, True
                if len(unassigned) == 1:
                    l = unassigned[0]
                    var = abs(l)
                    val = (l > 0)
                    if assignment[var - 1] is None:
                        assignment[var - 1] = val
                        changed = True
                new_clauses.append(c)
            clauses = new_clauses
        return clauses, assignment, False

    def pure_literal(clauses, assignment):
        pos = [False] * n
        neg = [False] * n
        for c in clauses:
            for l in c:
                var = abs(l) - 1
                if assignment[var] is not None:
                    continue
                if l > 0:
                    pos[var] = True
                else:
                    neg[var] = True
        for var in range(n):
            if assignment[var] is not None:
                continue
            if pos[var] and not neg[var]:
                return var, True
            if neg[var] and not pos[var]:
                return var, False
        return None, None

    clauses, assignment, contradiction = unit_propagate(clauses, assignment)
    if contradiction:
        return None
    while True:
        var, val = pure_literal(clauses, assignment)
        if var is None:
            break
        assignment[var] = val
        clauses, assignment, contradiction = unit_propagate(clauses, assignment)
        if contradiction:
            return None

    if all(a is not None for a in assignment):
        return assignment if _check(assignment, instance["clauses"]) else None
    return None


def _walksat(instance, seed, max_steps):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(seed)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Precompute for each variable, list of (clause_index, literal_sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append((ci, l))

    def is_sat(ci, assignment):
        c = clauses[ci]
        for l in c:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    cur_unsat = _count_unsat(assignment, clauses)
    best_assign = assignment[:]
    best_unsat = cur_unsat
    if best_unsat == 0:
        return best_assign, 0

    # Track which clauses are unsat as a set for efficiency
    unsat_set = set()
    for ci, c in enumerate(clauses):
        if not is_sat(ci, assignment):
            unsat_set.add(ci)

    T0 = 1.5
    Tend = 0.05
    for step in range(max_steps):
        if not unsat_set:
            return assignment, 0
        # Pick a random unsat clause
        clause_idx = rng.choice(list(unsat_set))
        clause = clauses[clause_idx]
        # For each literal in the clause, compute gain
        candidates = []
        for l in clause:
            var = abs(l) - 1
            new_val = not assignment[var]
            gain = 0
            # Count how many currently-unsat clauses containing var would become sat
            for (ci, _) in var_to_clauses[var]:
                if ci in unsat_set:
                    c = clauses[ci]
                    if any((abs(x) - 1 == var) and (new_val == (x > 0)) for x in c):
                        gain += 1
            candidates.append((gain, var, l))
        # Softmax selection
        gains = [g for g, v, l in candidates]
        max_g = max(gains) if gains else 0
        # Temperature anneals
        frac = step / max(1, max_steps)
        T = T0 * (Tend / T0) ** frac
        weights = [math.exp((g - max_g) / max(T, 1e-6)) for g in gains]
        total_w = sum(weights)
        r = rng.random() * total_w
        acc = 0
        chosen_var = candidates[0][1]
        for i, (g, v, l) in enumerate(candidates):
            acc += weights[i]
            if r <= acc:
                chosen_var = v
                break
        var = chosen_var
        old_val = assignment[var]
        assignment[var] = not old_val
        # Update unsat_set incrementally
        for (ci, _) in var_to_clauses[var]:
            was_sat = is_sat(ci, assignment)
            if was_sat:
                if ci in unsat_set:
                    unsat_set.remove(ci)
            else:
                if ci not in unsat_set:
                    unsat_set.add(ci)
        cur_unsat = len(unsat_set)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = assignment[:]
            if best_unsat == 0:
                return best_assign, 0
    return best_assign, best_unsat


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN

    # 1) DPLL pre-pass
    dpll_result = _dpll(instance)
    if dpll_result is not None:
        return dpll_result

    # 2) Annealed WalkSAT with restarts
    best_assign = None
    best_unsat = len(clauses)
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 60:
            steps = 60000
        elif remaining > 20:
            steps = 30000
        elif remaining > 5:
            steps = 10000
        else:
            steps = 3000
        assign, unsat = _walksat(instance, seed, steps)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assign
            if best_unsat == 0:
                return best_assign
        seed += 1

    if best_assign is None:
        rng = random.Random(999)
        best_assign = [rng.random() < 0.5 for _ in range(n)]
    return best_assign
# EVOLVE-BLOCK-END
