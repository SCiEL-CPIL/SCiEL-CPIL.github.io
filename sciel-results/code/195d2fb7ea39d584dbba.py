"""Improved planted 3-SAT solver: vectorized weighted WalkSAT with clause
learning, restarts, a deadline guard, and a final polish pass."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _build_arrays(clauses):
    # clauses: list of lists of signed 1-based ints
    m = len(clauses)
    L = np.array(clauses, dtype=np.int32)            # (m,3) signed
    idx = np.abs(L) - 1                              # (m,3) 0-based var index
    sign = np.where(L > 0, 1, -1).astype(np.int32)   # +1 if positive literal
    return L, idx, sign


def _eval_unsat(x, idx, sign):
    # x: (n,) bool. Returns boolean array (m,) True where clause is UNSAT.
    # A literal is true iff (x == (sign==+1)).
    lit_true = (x[None, :] == (sign == 1))           # (m,3)
    clause_sat = lit_true.any(axis=1)
    return ~clause_sat


def _polish(x, clauses, idx, sign, m, deadline, budget_steps=200000):
    """Deterministic-ish local improvement; returns (x, unsat_count)."""
    x = x.copy()
    unsat = _eval_unsat(x, idx, sign)
    c = unsat.sum()
    if c == 0:
        return x, 0
    rng = random.Random(12345)
    steps = 0
    while c > 0 and steps < budget_steps:
        if time.time() > deadline:
            break
        # pick a random unsat clause
        u = np.flatnonzero(unsat)
        ci = u[rng.randrange(len(u))]
        # try flipping each literal's variable, pick best delta
        best_j = -1
        best_c = c
        for j in range(3):
            v = idx[ci, j]
            x[v] = not x[v]
            new_unsat = _eval_unsat(x, idx, sign)
            nc = new_unsat.sum()
            if nc < best_c:
                best_c = nc
                best_j = j
                best_unsat = new_unsat
            x[v] = not x[v]
        if best_j >= 0:
            x[idx[ci, best_j]] = not x[idx[ci, best_j]]
            unsat = best_unsat
            c = best_c
        else:
            # random kick
            v = idx[ci, rng.randrange(3)]
            x[v] = not x[v]
            unsat = _eval_unsat(x, idx, sign)
            c = unsat.sum()
        steps += 1
    return x, c


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start = time.time()
    deadline = start + 320.0  # safety margin before 330s kill

    L, idx, sign = _build_arrays(clauses)

    # variable occurrence counts for initial weights
    var_count = np.zeros(n, dtype=np.int32)
    for l in L:
        var_count[np.abs(l) - 1] += 1

    best_x = None
    best_c = m + 1

    def consider(x, c):
        nonlocal best_x, best_c
        if c < best_c:
            best_c = c
            best_x = x.copy()
        if c == 0:
            return True
        return False

    rng = random.Random(0)
    restart = 0
    base_steps = 4000
    # clause weights
    weights = np.ones(m, dtype=np.float64)

    while time.time() < deadline:
        restart += 1
        # random initial assignment
        x = (np.random.RandomState(1000 + restart).rand(n) < 0.5)
        unsat = _eval_unsat(x, idx, sign)
        c = int(unsat.sum())
        if consider(x, c):
            return x.tolist()

        steps = base_steps * (1 + restart // 8)
        p_random = 0.05  # noise factor
        local = 0
        for s in range(steps):
            if time.time() > deadline:
                break
            if c == 0:
                return x.tolist()
            # pick a clause: weighted by unsat, with soft-max emphasis
            u = np.flatnonzero(unsat)
            if len(u) == 0:
                return x.tolist()
            # choose clause: with prob p_random random, else weighted by weights
            if rng.random() < p_random:
                ci = u[rng.randrange(len(u))]
            else:
                wu = weights[u]
                # log-sum-exp soft-max to pick heavily weighted unsat clause
                wmax = wu.max()
                ew = np.exp(wu - wmax)
                r = rng.random() * ew.sum()
                acc = 0.0
                ci = u[-1]
                for k in range(len(u)):
                    acc += ew[k]
                    if r <= acc:
                        ci = u[k]
                        break
            # choose literal to flip
            # heuristic: prefer flips that satisfy this clause and hurt few others
            # count how many clauses each of the 3 vars currently satisfies via this clause's literal
            best_j = -1
            best_score = None
            for j in range(3):
                v = idx[ci, j]
                # delta: flipping v changes satisfaction of all clauses containing v
                # approximate score = (# clauses v currently satisfies) - weight of ci
                # We precompute per-variable satisfied-clause count lazily via vectorized eval
                x[v] = not x[v]
                new_unsat = _eval_unsat(x, idx, sign)
                nc = new_unsat.sum()
                x[v] = not x[v]
                score = nc  # lower is better
                if best_score is None or score < best_score - 0.0 or (
                        score == best_score and rng.random() < 0.5):
                    best_score = score
                    best_j = j
            x[idx[ci, best_j]] = not x[idx[ci, best_j]]
            new_unsat = _eval_unsat(x, idx, sign)
            nc = int(new_unsat.sum())
            # clause learning: increase weight of still-unsat clauses, decay others
            weights[new_unsat] *= 1.1
            weights[~new_unsat] *= 0.99
            unsat = new_unsat
            if nc < c:
                local = 0
            else:
                local += 1
            c = nc
            if consider(x, c):
                if c == 0:
                    return x.tolist()
        # after a walk, final polish attempt
        if time.time() < deadline:
            px, pc = _polish(x, clauses, idx, sign, m, deadline)
            if pc < best_c:
                best_c = pc
                best_x = px
            if pc == 0:
                return px.tolist()
        if best_c == 0:
            break

    # final polish on the best assignment found
    if best_x is not None and time.time() < deadline:
        px, pc = _polish(best_x, clauses, idx, sign, m, deadline)
        if pc == 0:
            return px.tolist()
        return px.tolist()
    if best_x is not None:
        return best_x.tolist()
    # fallback
    return [False] * n
# EVOLVE-BLOCK-END
