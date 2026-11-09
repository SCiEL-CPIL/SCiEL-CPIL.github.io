# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DEADLINE_SECONDS = 295.0  # safety margin under the 300s evaluator limit


def _build(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    pos = np.empty((len(clauses), 3), dtype=np.int64)
    neg = np.empty((len(clauses), 3), dtype=np.int64)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            if l > 0:
                pos[i, j] = l - 1
                neg[i, j] = -1
            else:
                neg[i, j] = -l - 1
                pos[i, j] = -1
    return n, pos, neg


def _phase_score(x, pos, neg, m, n):
    # number of satisfied clauses under assignment x (0/1 array)
    true_pos = x[pos]
    true_neg = 1.0 - x[neg]
    val = np.where(pos >= 0, true_pos, true_neg)
    return float((val.max(axis=1) > 0).sum())


def _run_walksat(n, pos, neg, m, x, phase, rng, deadline):
    """WalkSAT with phase save/restore and adaptive noise. Returns (x, solved)."""
    best = x.copy()
    best_score = _phase_score(x, pos, neg, m, n)
    if best_score >= m:
        return best, True

    sat = np.zeros(m, dtype=np.int8)
    for i in range(m):
        v = np.where(pos[i] >= 0, x[pos[i]], 1.0 - x[neg[i]])
        sat[i] = (v.max() > 0)
    unsat_count = m - int(sat.sum())

    # initial phase score: prefer the assignment that satisfies more clauses
    phase0 = x.copy()
    s0 = best_score
    xflip = 1.0 - x
    s1 = _phase_score(xflip, pos, neg, m, n)
    if s1 > s0:
        x = xflip
        best = x.copy()
        best_score = s1
        phase0 = x.copy()
        unsat_count = m - s1
        sat = np.zeros(m, dtype=np.int8)
        for i in range(m):
            v = np.where(pos[i] >= 0, x[pos[i]], 1.0 - x[neg[i]])
            sat[i] = (v.max() > 0)
    else:
        x = phase0.copy()

    if best_score >= m:
        return best, True

    step = 0
    while time.time() < deadline:
        if unsat_count == 0:
            return x, True
        step += 1
        # pick an unsatisfied clause; bias toward the most violated (longest unsat)
        # for 3-SAT all unsat clauses have 0 satisfied literals, so pick randomly
        uidx = np.flatnonzero(sat == 0)
        ci = int(rng.choice(uidx))
        p = pos[ci]
        nn = neg[ci]
        # candidate variables: the literals in this clause
        cands = []
        for j in range(3):
            if p[j] >= 0:
                cands.append((p[j], 1))   # setting to True satisfies
            if nn[j] >= 0:
                cands.append((nn[j], 0))  # setting to False satisfies
        # noise: with probability decay with progress, flip a random variable
        progress = unsat_count / m
        noise_p = 0.25 * (0.5 + 0.5 * progress)  # more noise when far from solved
        if rng.random() < noise_p:
            var = rng.randrange(n)
            target = 1.0 if rng.random() < 0.5 else 0.0
        else:
            var, target = cands[rng.randrange(len(cands))]

        old = x[var]
        if old == target:
            continue
        # flip var
        x[var] = target
        # update sat for clauses containing var
        # find clauses where pos==var or neg==var
        affected = np.where((pos == var) | (neg == var))[0]
        for i in affected:
            v = np.where(pos[i] >= 0, x[pos[i]], 1.0 - x[neg[i]])
            new_sat = 1 if v.max() > 0 else 0
            if new_sat != sat[i]:
                unsat_count += (1 if new_sat else -1)
                sat[i] = new_sat

        if unsat_count == 0:
            return x, True

        cur_score = m - unsat_count
        if cur_score > best_score:
            best_score = cur_score
            best = x.copy()
        # phase restore: if we are much worse than the initial phase, reset to phase0
        # and re-randomize a bit (cheap local restart)
        if cur_score < s0 - 15 and step % 200 == 0:
            # local restart: keep best, but jump toward best
            x = best.copy()
            # recompute sat
            sat = np.zeros(m, dtype=np.int8)
            for i in range(m):
                v = np.where(pos[i] >= 0, x[pos[i]], 1.0 - x[neg[i]])
                sat[i] = (v.max() > 0)
            unsat_count = m - int(sat.sum())
            if unsat_count == 0:
                return x, True
            step = 0

    return best, False


def solve(instance):
    start = time.time()
    deadline = start + DEADLINE_SECONDS
    rng = random.Random(0)
    n, pos, neg = _build(instance)
    m = len(pos)

    best = None
    best_score = -1

    # multiple restarts with different random seeds
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = random.Random(attempt * 7919 + 13)
        x = np.array([1.0 if r.random() < 0.5 else 0.0 for _ in range(n)])
        x, solved = _run_walksat(n, pos, neg, m, x, None, r, deadline)
        sc = _phase_score(x, pos, neg, m, n)
        if sc > best_score:
            best_score = sc
            best = x.copy()
        if solved or best_score >= m:
            break
        # if we keep failing, the time budget naturally limits restarts

    if best is None:
        best = np.array([0.0] * n)
    return [bool(b) for b in best]
# EVOLVE-BLOCK-END
