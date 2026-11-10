"""Planted 3-SAT solver: vectorized WalkSAT with adaptive noise + restarts."""
import time
import numpy as np


def _is_satisfied(A, B, assign):
    # A: (m,3) literal signs in {+1,-1}; B: (m,3) 0-based var indices
    # returns boolean array of length m
    return np.any(A * (2.0 * assign[B] - 1.0) > 0.0, axis=1)


def _walksat(A, B, m, n, assign, rng, budget, t0, deadline, best, best_count):
    unsat = ~_is_satisfied(A, B, assign)
    unsat_count = int(unsat.sum())
    if unsat_count == 0:
        return assign, True, 0
    best = assign.copy()
    best_count = unsat_count
    for step in range(budget):
        if unsat_count == 0:
            return assign, True, step
        if time.monotonic() > deadline:
            return best, False, step
        idx = int(rng.choice(np.nonzero(unsat)[0]))
        c = A[idx]
        v = int(B[idx, int(rng.randint(0, 2))])
        flip = assign[v]
        assign[v] = not flip
        # delta update: clauses containing variable v
        rows = np.nonzero(B[:, 0] == v)[0]
        if rows.size:
            rows = np.nonzero(B[:, 1] == v)[0]
            rows = np.concatenate(rows, np.nonzero(B[:, 2] == v)[0])
        else:
            rows = np.concatenate(
                (np.nonzero(B[:, 1] == v)[0], np.nonzero(B[:, 2] == v)[0]))
        if rows.size:
            # literal values for variable v in those clauses
            litv = np.empty(rows.size, dtype=np.float64)
            r0 = rows[B[rows, 0] == v]
            r1 = rows[B[rows, 1] == v]
            r2 = rows[B[rows, 2] == v]
            litv[r0] = A[r0, 0]
            litv[r1] = A[r1, 1]
            litv[r2] = A[r2, 2]
            newval = 2.0 * (1.0 if assign[v] else 0.0) - 1.0
            oldval = 2.0 * (1.0 if flip else 0.0) - 1.0
            sat_now = litv * newval > 0.0
            sat_old = litv * oldval > 0.0
            other_sat = np.any(
                (A[rows] * (2.0 * assign[B[rows]] - 1.0) > 0.0), axis=1)
            other_sat[rows == r0] = np.any(
                (A[r0] * (2.0 * assign[B[r0]] - 1.0) > 0.0)
                & (np.arange(3) != 0), axis=1) if False else other_sat[rows == r0]
            # recompute "other" saturation robustly (cheap, m small)
            other_sat = np.any(
                (A[rows] * (2.0 * assign[B[rows]] - 1.0) > 0.0), axis=1)
            other_sat[other_sat & np.isin(rows, r0)] = np.any(
                (A[r0] * (2.0 * assign[B[r0]] - 1.0) > 0.0)
                & (np.arange(3) != 0), axis=1)
            # simpler robust path: recompute full clause sat for affected rows
            full = (A[rows] * (2.0 * assign[B[rows]] - 1.0) > 0.0)
            new_sat = np.any(full, axis=1)
            unsat[rows] = ~new_sat
            unsat_count = int(unsat.sum())
        if unsat_count < best_count:
            best = assign.copy()
            best_count = unsat_count
        # adaptive noise: high early, anneals as unsat_count shrinks
        if unsat_count <= 2:
            noise = 0.02
        elif unsat_count <= 5:
            noise = 0.08
        elif unsat_count <= 10:
            noise = 0.18
        elif unsat_count <= 20:
            noise = 0.30
        else:
            noise = 0.45
        if rng.random() < noise:
            # random flip
            v2 = int(rng.randint(0, n - 1))
            assign[v2] = not assign[v2]
            rows2 = np.concatenate(
                (np.nonzero(B[:, 0] == v2)[0],
                 np.nonzero(B[:, 1] == v2)[0],
                 np.nonzero(B[:, 2] == v2)[0]))
            if rows2.size:
                full2 = (A[rows2] * (2.0 * assign[B[rows2]] - 1.0) > 0.0)
                unsat[rows2] = ~np.any(full2, axis=1)
                unsat_count = int(unsat.sum())
            if unsat_count < best_count:
                best = assign.copy()
                best_count = unsat_count
    return best, False, budget


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    rng = np.random.default_rng(12345)
    t0 = time.monotonic()
    deadline = t0 + 280.0  # safety margin before 330s kill

    # precompute clause literal arrays
    A = np.zeros((m, 3), dtype=np.float64)
    B = np.zeros((m, 3), dtype=np.int64)
    for i, cl in enumerate(clauses):
        for j, lit in enumerate(cl):
            A[i, j] = 1.0 if lit > 0 else -1.0
            B[i, j] = abs(lit) - 1

    best = None
    best_count = m + 1
    start_budget = 2000
    max_budget = 60000
    restart = 0
    last_count = m + 1
    while time.monotonic() < deadline:
        # fresh random start each restart
        assign = (rng.random(n) < 0.5)
        budget = min(start_budget * (restart + 1), max_budget)
        assign, solved, steps = _walksat(
            A, B, m, n, assign, rng, budget, t0, deadline, best, best_count)
        cur = int((~_is_satisfied(A, B, assign)).sum())
        if cur < best_count:
            best_count = cur
            best = assign.copy()
        if cur == 0:
            return assign.tolist()
        # annealing restart: only restart if we made little progress
        if cur < last_count - 3:
            last_count = cur
            restart = 0
        else:
            restart += 1
            if restart > 40:
                restart = 40
    if best is not None:
        return best.tolist()
    return (rng.random(n) < 0.5).tolist()
