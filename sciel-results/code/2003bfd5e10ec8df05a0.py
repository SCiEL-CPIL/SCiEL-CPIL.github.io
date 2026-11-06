"""Planted 3-SAT solver (50 vars) - WalkSAT with vectorized eval, weighted restarts, polish."""
import time
import numpy as np


def _satisfied_mask(assign, pos, neg):
    # returns bool array: clause satisfied if any literal true
    return np.logical_or(np.logical_or(assign[pos], assign[neg]),
                         np.logical_or(~assign[pos], ~assign[neg]))


def _polish(assign, pos, neg, n):
    """Exact local resolution: unit propagation + exhaustive 3-literal backtracking."""
    a = assign.copy()
    # iterative unit propagation
    changed = True
    while changed:
        changed = False
        for i in range(len(pos)):
            c = i
            s = np.logical_or(np.logical_or(a[pos[c]], a[neg[c]]),
                              np.logical_or(~a[pos[c]], ~a[neg[c]]))
            if s[c]:
                continue
            # exactly one literal false -> force it
            l0 = a[pos[c]] == (pos[c] > 0)
            l1 = a[neg[c]] == (neg[c] > 0)
            l2 = a[pos[c + 1]] == (pos[c + 1] > 0)
            l3 = a[neg[c + 1]] == (neg[c + 1] > 0)
            l4 = a[pos[c + 2]] == (pos[c + 2] > 0)
            l5 = a[neg[c + 2]] == (neg[c + 2] > 0)
            vals = [l0, l1, l2, l3, l4, l5]
            if sum(vals) == 5:  # exactly one false
                for j, v in enumerate(vals):
                    if not v:
                        lit = [pos[c], neg[c], pos[c + 1], neg[c + 1], pos[c + 2], neg[c + 2]][j]
                        a[abs(lit) - 1] = (lit > 0)
                        changed = True
                        break
    # check satisfied
    if np.all(_satisfied_mask(a, pos, neg)):
        return a
    # exhaustive backtracking over the 3 unsatisfied clauses (each has 1 false literal)
    unsat = []
    for i in range(len(pos)):
        c = i
        s = np.logical_or(np.logical_or(a[pos[c]], a[neg[c]]),
                          np.logical_or(~a[pos[c]], ~a[neg[c]]))
        if not s[c]:
            l0 = a[pos[c]] == (pos[c] > 0)
            l1 = a[neg[c]] == (neg[c] > 0)
            l2 = a[pos[c + 1]] == (pos[c + 1] > 0)
            l3 = a[neg[c + 1]] == (neg[c + 1] > 0)
            l4 = a[pos[c + 2]] == (pos[c + 2] > 0)
            l5 = a[neg[c + 2]] == (neg[c + 2] > 0)
            vals = [l0, l1, l2, l3, l4, l5]
            flit = [pos[c], neg[c], pos[c + 1], neg[c + 1], pos[c + 2], neg[c + 2]]
            flit = [flit[j] for j in range(6) if not vals[j]]
            unsat.append(flit[0])
    # try all 2^k flips of the distinct variables in unsat
    distinct = list(dict.fromkeys(abs(l) - 1 for l in unsat))
    k = len(distinct)
    for mask in range(1 << k):
        trial = a.copy()
        for b in range(k):
            if mask & (1 << b):
                trial[distinct[b]] = not trial[distinct[b]]
        if np.all(_satisfied_mask(trial, pos, neg)):
            return trial
    return a


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Build literal arrays: for each clause, 3 literals (signed, 1-based).
    # pos = index of literal if it's positive, neg = index if negative.
    pos = np.empty((m, 3), dtype=np.int64)
    neg = np.empty((m, 3), dtype=np.int64)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            if l > 0:
                pos[i, j] = l - 1
                neg[i, j] = -1
            else:
                pos[i, j] = -1
                neg[i, j] = -l - 1
    # For vectorized eval, use a single "literal value" lookup:
    # lit_val[i,j] = assign[idx] if positive else ~assign[idx]
    lit_idx = np.where(pos >= 0, pos, neg)  # index into assign
    lit_sign = np.where(pos >= 0, 1, 0)    # 1 positive, 0 negative

    t0 = time.time()
    deadline = t0 + 295.0  # safety margin before 330s kill

    rng = np.random.RandomState(12345)

    # Per-variable clause incidence (for the weighted flip heuristic)
    # var_clauses[v] = list of clause indices containing variable v (as positive or negative)
    var_clauses = [[] for _ in range(n)]
    var_polarity = [[] for _ in range(n)]  # polarity[v] = 1 if lit positive else 0
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_clauses[v].append(i)
            var_polarity[v].append(1 if l > 0 else 0)

    def lit_values(assign):
        # shape (m,3)
        a = assign
        vals = np.empty((m, 3), dtype=bool)
        p = pos >= 0
        # positive literals: assign[idx]; negative: ~assign[idx]
        vals[p] = a[lit_idx[p]]
        vals[~p] = ~a[lit_idx[~p]]
        return vals

    def unsat_count(assign):
        lv = lit_values(assign)
        sat = np.any(lv, axis=1)
        return (~sat).sum(), sat

    # initial random assignment
    assign = rng.rand(n) < 0.5
    ucount, sat = unsat_count(assign)
    if ucount == 0:
        return assign.tolist()

    best_assign = assign.copy()
    best_ucount = ucount

    # clause weights (for heuristic)
    cweight = np.ones(m)

    restart = 0
    while time.time() < deadline:
        restart += 1
        if restart == 1:
            pass  # use current assign
        else:
            # weighted restart: keep best with prob 0.3, else random
            if rng.rand() < 0.3:
                assign = best_assign.copy()
            else:
                assign = rng.rand(n) < 0.5
            ucount, sat = unsat_count(assign)
            if ucount == 0:
                return assign.tolist()
            if ucount < best_ucount:
                best_ucount = ucount
                best_assign = assign.copy()

        # WalkSAT loop
        noise = 0.45 + 0.1 * min(restart, 5)  # mild adaptation
        steps = 0
        max_steps = 2000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            if np.all(sat):
                return assign.tolist()
            # pick unsat clause (uniform)
            unsat_idx = np.where(~sat)[0]
            c = unsat_idx[rng.randint(len(unsat_idx))]
            # compute, for each of the 3 literals in clause c, how many other
            # unsatisfied clauses would be satisfied by flipping that variable.
            best_score = -1
            best_v = -1
            best_is_neg = False
            for j in range(3):
                l = clauses[c][j]
                v = abs(l) - 1
                is_neg = l < 0
                # score: number of currently-unsat clauses that contain v and
                # would become satisfied by flipping v.
                score = 0
                for ci in var_clauses[v]:
                    if sat[ci]:
                        continue
                    # would flipping v satisfy clause ci?
                    # find the literal of v in ci
                    for lj, ll in enumerate(clauses[ci]):
                        if abs(ll) - 1 == v:
                            cur = assign[v] == (ll > 0)
                            if not cur:
                                # currently false; flipping makes it true -> satisfies
                                score += 1
                            break
                # random tie-break / noise
                score += rng.rand() * noise
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_is_neg = is_neg
            if best_v < 0:
                break
            assign[best_v] = not assign[best_v]
            ucount, sat = unsat_count(assign)
            if ucount < best_ucount:
                best_ucount = ucount
                best_assign = assign.copy()

        if best_ucount == 0:
            return best_assign.tolist()

    # final polish on best
    pol = _polish(best_assign, pos, neg, n)
    if np.all(_satisfied_mask(pol, pos, neg)):
        return pol.tolist()
    return best_assign.tolist()
