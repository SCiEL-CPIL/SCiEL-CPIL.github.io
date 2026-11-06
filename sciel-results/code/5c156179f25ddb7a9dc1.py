"""
Reciprocal-weight maximization for 3-AP-free subsets of [1,256].

Model: MAX-WT SAT.  Variable i (1..256) <=> i is in the set.  One clause per
3-AP (a,b,c): (¬a ∨ ¬b ∨ ¬c).  Objective: maximize sum_{i in S} 1/i.

Solver: from-scratch CDCL SAT with watched literals, clause learning (1-UIP),
VSIDS activity, phase saving, and Luby restarts.  A time-bounded maximizing
wrapper walks the implication trail from the last conflict to push the
assignment toward higher reciprocal weight.

The reciprocal weight is computed with fractions.Fraction for an exact
numerator/denominator (the evaluator serializes it); floats are used only for
ordering inside the search.
"""

import time
import random
from fractions import Fraction

N = 256


def _build_aps():
    aps = []
    for b in range(2, N + 1):
        lo = max(1, 2 * b - N)
        for a in range(lo, b):
            c = 2 * b - a
            aps.append((a, b, c))
    return aps


def _is_valid(S):
    Sset = set(S)
    for b in range(2, N + 1):
        if b not in Sset:
            continue
        lo = max(1, 2 * b - N)
        for a in range(lo, b):
            if a in Sset and (2 * b - a) in Sset:
                return False
    return True


def _exact_weight(S):
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total


def _luby(seq, i):
    k = 1
    while (1 << (k + 1)) - 1 <= i:
        k += 1
    k = (1 << k) - 1
    i = i - k - 1
    if i < 0:
        return seq[0]
    return seq[min(k, len(seq) - 1)]


class CDCL:
    __slots__ = (
        "nvars", "clauses", "watched", "activity", "var_inc", "phase",
        "assign", "level", "reason", "trail", "qhead", "learnts",
        "learnts_watched", "learnts_activity", "learnts_var_inc",
        "learnts_count", "conflicts", "decision_level", "objective",
    )

    def __init__(self, nvars, clauses, objective):
        self.nvars = nvars
        self.clauses = clauses
        self.objective = objective
        self.watched = [[] for _ in range(2 * nvars)]
        self.activity = [0.0] * (nvars + 1)
        self.var_inc = 1.0
        self.phase = [True] * (nvars + 1)
        self.assign = [None] * (nvars + 1)
        self.level = [0] * (nvars + 1)
        self.reason = [None] * (nvars + 1)
        self.trail = []
        self.qhead = 0
        self.learnts = []
        self.learnts_watched = [[] for _ in range(2 * nvars)]
        self.learnts_activity = []
        self.learnts_var_inc = 1.0
        self.learnts_count = 0
        self.conflicts = 0
        self.decision_level = 0
        for idx, cl in enumerate(clauses):
            for lit in (cl[0], cl[1]):
                self.watched[2 * (abs(lit) - 1) - (1 if lit < 0 else 0)].append(idx)

    def _bump(self, var, arr, inc):
        arr[var] += inc
        if arr[var] > 1e100:
            for i in range(1, len(arr)):
                arr[i] *= 1e-100
            inc *= 1e-100
        return inc

    def _bump_l(self, var):
        self.var_inc = self._bump(var, self.activity, self.var_inc)

    def _bump_l_learnt(self, var):
        self.learnts_var_inc = self._bump(var, self.learnts_activity, self.learnts_var_inc)

    def _update_watched(self, idx, new_lit):
        cl = self.clauses[idx]
        w = self.watched
        for wlit in (cl[0], cl[1]):
            lst = w[2 * (abs(wlit) - 1) - (1 if wlit < 0 else 0)]
            j = lst.index(idx)
            lst[j] = lst[-1]
            lst.pop()
        w[2 * (abs(new_lit) - 1) - (1 if new_lit < 0 else 0)].append(idx)
        other = cl[1] if cl[0] == new_lit else cl[0]
        w[2 * (abs(other) - 1) - (1 if other < 0 else 0)].append(idx)
        cl[0] = new_lit
        cl[1] = other

    def _propagate(self):
        assign = self.assign
        clauses = self.clauses
        watched = self.watched
        trail = self.trail
        qhead = self.qhead
        while qhead < len(trail):
            p = trail[qhead]
            qhead += 1
            lit = -p
            var = abs(p) - 1
            wlits = watched[2 * var - (1 if p < 0 else 0)]
            i = 0
            m = len(wlits)
            while i < m:
                idx = wlits[i]
                cl = clauses[idx]
                if cl[0] == lit:
                    cl[0], cl[1] = cl[1], lit
                p1 = cl[0]
                if assign[abs(p1) - 1] is True and p1 < 0:
                    found = False
                    for j in range(2, len(cl)):
                        pj = cl[j]
                        val = assign[abs(pj) - 1]
                        if val is None or (val is True and pj > 0):
                            self._update_watched(idx, pj)
                            found = True
                            break
                    if found:
                        i += 1
                        m = len(wlits)
                        continue
                    if cl[1] != lit and assign[abs(cl[1]) - 1] is None:
                        trail.append(cl[1])
                        self.level[abs(cl[1]) - 1] = self.decision_level
                        self.reason[abs(cl[1]) - 1] = idx
                    else:
                        return idx
                    continue
                i += 1
        self.qhead = qhead
        return -1

    def _analyze(self, conflict_idx):
        trail = self.trail
        learnt = [0]
        seen = [False] * (self.nvars + 1)
        counter = 0
        p = 0
        idx = conflict_idx
        while True:
            for q in trail[p:]:
                v = abs(q) - 1
                if not seen[v]:
                    self._bump_l(v)
                    seen[v] = True
                    if self.level[v] > 0:
                        learnt.append(q)
                        counter += 1
            p = self.level[abs(trail[p]) - 1]
            counter -= 1
            if counter == 0:
                break
            idx = self.reason[abs(trail[p]) - 1]
            if idx >= self.nvars:
                cl = self.learnts[idx - self.nvars]
                for q in cl:
                    v = abs(q) - 1
                    if not seen[v]:
                        self._bump_l(v)
                        seen[v] = True
                        if self.level[v] > 0:
                            learnt.append(q)
                            counter += 1
            else:
                cl = self.clauses[idx]
                for q in cl[2:]:
                    v = abs(q) - 1
                    if not seen[v]:
                        self._bump_l(v)
                        seen[v] = True
                        if self.level[v] > 0:
                            learnt.append(q)
                            counter += 1
        for q in learnt:
            self.phase[abs(q) - 1] = False
        learnt.sort(key=lambda q: -self.level[abs(q) - 1])
        learnt = learnt[1:]
        self._bump_l_learnt(abs(learnt[0]) - 1)
        self.learnts.append(learnt)
        self.learnts_activity.append(0.0)
        self.learnts_count += 1
        for wlit in (learnt[0], learnt[1]):
            self.learnts_watched[2 * (abs(wlit) - 1) - (1 if wlit < 0 else 0)].append(self.learnts_count - 1)
        return learnt

    def _cancel_until(self, level):
        assign = self.assign
        trail = self.trail
        while len(trail) > 0 and self.level[abs(trail[-1]) - 1] > level:
            v = abs(trail[-1]) - 1
            assign[v] = None
            trail.pop()
        self.qhead = len(trail)
        self.decision_level = level

    def _select_var(self):
        assign = self.assign
        phase = self.phase
        objective = self.objective
        activity = self.activity
        best = -1
        best_act = -1.0
        for i in range(1, self.nvars + 1):
            if assign[i - 1] is None:
                act = activity[i]
                if objective is not None:
                    act += objective[i - 1] * 1e-6
                if act > best_act:
                    best_act = act
                    best = i
        if best < 0:
            return None
        return best if phase[best - 1] else -best

    def _is_solved(self):
        return len(self.trail) == self.nvars

    def search(self, time_limit, max_conflicts=None):
        deadline = time.time() + time_limit
        luby_i = 0
        cur_conflicts = 0
        while True:
            if time.time() > deadline:
                return None
            if max_conflicts is not None and self.conflicts >= max_conflicts:
                return None
            conflict = self._propagate()
            if conflict >= 0:
                self.conflicts += 1
                cur_conflicts += 1
                learnt = self._analyze(conflict)
                self._bump_l_learnt(abs(learnt[0]) - 1)
                if len(learnt) == 1:
                    self._cancel_until(0)
                    trail = self.trail
                    trail.append(learnt[0])
                    assign = self.assign
                    assign[abs(learnt[0]) - 1] = (learnt[0] > 0)
                    self.level[abs(learnt[0]) - 1] = 0
                    self.reason[abs(learnt[0]) - 1] = self.nvars + self.learnts_count - 1
                else:
                    back = self.level[abs(learnt[1]) - 1]
                    self._cancel_until(back)
                    trail = self.trail
                    trail.append(learnt[0])
                    assign = self.assign
                    assign[abs(learnt[0]) - 1] = (learnt[0] > 0)
                    self.level[abs(learnt[0]) - 1] = self.decision_level
                    self.reason[abs(learnt[0]) - 1] = self.nvars + self.learnts_count - 1
                continue
            if self._is_solved():
                return [(i + 1, self.assign[i] is True) for i in range(self.nvars)]
            var = self._select_var()
            if var is None:
                return [(i + 1, self.assign[i] is True) for i in range(self.nvars)]
            self.decision_level += 1
            self.trail.append(var)
            self.assign[abs(var) - 1] = (var > 0)
            self.level[abs(var) - 1] = self.decision_level
            self.reason[abs(var) - 1] = None
            self.phase[abs(var) - 1] = (var > 0)
            if cur_conflicts > 100:
                luby_i += 1
                target = 100 * _luby([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], luby_i)
                if cur_conflicts >= target:
                    self._cancel_until(0)
                    cur_conflicts = 0
        return None


def _maximize_weighted(solver, nvars, objective, time_limit):
    """Walk the implication trail to push toward higher reciprocal weight."""
    deadline = time.time() + time_limit
    assign = solver.assign
    trail = solver.trail

    def current_weight():
        w = 0.0
        for i in range(nvars):
            if assign[i] is True:
                w += objective[i]
        return w

    best_S = {i + 1 for i in range(nvars) if assign[i] is True}
    best_w = current_weight()

    decision_vars = []
    for lit in trail:
        v = abs(lit) - 1
        if solver.reason[v] is None and solver.level[v] > 0:
            decision_vars.append(v)

    attempts = 0
    max_attempts = 150
    for v in decision_vars:
        if time.time() > deadline or attempts >= max_attempts:
            break
        attempts += 1
        cur_val = assign[v]
        if cur_val is None:
            continue
        lvl = solver.level[v]
        solver._cancel_until(lvl - 1)
        new_val = not cur_val
        solver.assign[v] = new_val
        solver.level[v] = lvl
        solver.reason[v] = None
        solver.phase[v] = new_val
        solver.trail.append((v + 1) if new_val else -(v + 1))
        conflict = solver._propagate()
        if conflict >= 0:
            solver._cancel_until(lvl - 1)
            solver.assign[v] = cur_val
            solver.level[v] = lvl
            solver.reason[v] = None
            solver.trail.append((v + 1) if cur_val else -(v + 1))
            solver._propagate()
            continue
        w = current_weight()
        if w > best_w:
            best_w = w
            best_S = {i + 1 for i in range(nvars) if assign[i] is True}
    return best_S, best_w


def _greedy():
    S = set()
    for x in range(1, N + 1):
        ok = True
        lo = max(1, 2 * x - N)
        for a in range(lo, x):
            if a in S and (2 * x - a) in S:
                ok = False
                break
        if not ok:
            continue
        for a in range(1, x):
            b = 2 * x - a
            if b < x and a in S and b in S:
                ok = False
                break
        if not ok:
            continue
        S.add(x)
    return S


def _local_improve(S, max_passes=40):
    Sset = set(S)
    weights = {i: 1.0 / i for i in range(1, N + 1)}
    for _ in range(max_passes):
        improved = False
        for x in sorted(Sset, reverse=True):
            for y in range(1, x):
                if y in Sset:
                    continue
                S2 = (Sset - {x})
                S2.add(y)
                if _is_valid(S2) and weights[y] > weights[x]:
                    Sset = S2
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return Sset


def construct():
    start = time.time()
    deadline = start + 80.0

    aps = _build_aps()
    clauses = [[-a, -b, -c] for (a, b, c) in aps]
    objective = [1.0 / i for i in range(1, N + 1)]

    solver = CDCL(N, clauses, objective)
    for i in range(N):
        solver.phase[i] = True

    sol = solver.search(time_limit=25.0, max_conflicts=3000000)
    if sol is None:
        S = _greedy()
        if not S:
            S = {1}
        return sorted(S)

    S, w = _maximize_weighted(solver, N, objective, time_limit=35.0)

    if not _is_valid(S):
        S = _greedy()
    if not S:
        S = {1}

    if time.time() < deadline - 5.0:
        S = _local_improve(S)

    return sorted(S)
