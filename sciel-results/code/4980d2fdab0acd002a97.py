"""Planted 3-SAT solver: vectorized WalkSAT with clause-count tracking,
score-based clause selection, and multiple restarts under a wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _build_index(n, clauses):
    """Return 0-based clause arrays and per-variable clause indices."""
    m = len(clauses)
    c0 = np.empty((m, 3), dtype=np.int32)
    sgn = np.empty((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, lit in enumerate(cl):
            c0[i, j] = abs(lit) - 1
            sgn[i, j] = 1 if lit > 0 else -1
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[int(c0[i, j])].append(i)
    return c0, sgn, var_to_clauses


def _satisfied_all(assignment, c0, sgn):
    """True if every clause is satisfied."""
    vals = assignment[c0] * sgn
    return not (vals.sum(axis=1) == 0).any()


def _walksat_once(n, c0, sgn, var_to_clauses, seed, deadline,
                  p_walk=0.55, max_iter=200000):
    """One WalkSAT run; returns (solved, assignment)."""
    rng = random.Random(seed)
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
    vals = assignment[c0] * sgn
    cc = (vals == 0).sum(axis=1)
    total = int(cc.sum())
    if total == 0:
        return True, assignment

    best_cc = total
    best_assignment = assignment.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Pick an unsatisfied clause (score-based: random among unsat).
        unsat_idx = np.where(cc == 0)[0]
        if len(unsat_idx) == 0:
            return True, assignment
        ci = unsat_idx[rng.randrange(len(unsat_idx))]
        clause = c0[ci]
        signs = sgn[ci]

        # Flip: with probability p_walk pick a random variable in the clause;
        # otherwise pick the variable that minimizes resulting unsat count.
        if rng.random() < p_walk:
            v = clause[rng.randrange(3)]
        else:
            # Compute delta in unsat count for flipping each variable in clause.
            best_delta = None
            best_v = clause[0]
            for k in range(3):
                v = clause[k]
                s = signs[k]
                cc_v = cc[var_to_clauses[v]]
                # After flip, this literal's value flips sign.
                new_cc = cc_v - s
                # Variables that become newly unsatisfied (were sat, now 0).
                new_unsat = np.count_nonzero((new_cc == 0) & (cc_v != 0))
                # Variables that become satisfied (were 0, now nonzero).
                new_sat = np.count_nonzero((new_cc != 0) & (cc_v == 0))
                delta = int(new_unsat - new_sat)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_v = v
            v = best_v

        # Apply flip.
        sgn_v = sgn[0]  # placeholder, recompute per clause
        assignment[v] = -assignment[v]
        for ci2 in var_to_clauses[v]:
            # recompute this clause's unsat count
            vals2 = assignment[c0[ci2]] * sgn[ci2]
            cc[ci2] = int((vals2 == 0).sum())
        total = int(cc.sum())
        if total < best_cc:
            best_cc = total
            best_assignment = assignment.copy()
        if total == 0:
            return True, assignment
    return False, best_assignment


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 300s kill

    c0, sgn, var_to_clauses = _build_index(n, clauses)

    # Try a batch of random initializations via vectorized random assignment.
    # Each restart uses a different seed.
    seed = 0
    while time.time() < deadline:
        solved, assignment = _walksat_once(
            n, c0, sgn, var_to_clauses, seed, deadline,
            p_walk=0.55, max_iter=200000
        )
        if solved:
            return assignment.tolist()
        seed += 1
    # Fallback: return last best (random) assignment.
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
