"""Planted 3-SAT solver (50 vars) - vectorized WalkSAT with restarts and deadline guard."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline: keep a safety margin under the 330s kill (evaluator budget is 300s).
    start = time.time()
    DEADLINE = start + 290.0

    # Precompute literal -> (index, sign) and clause structure once.
    # For each clause store: idxs (array of variable indices) and signs (True if positive).
    clause_idxs = []
    clause_signs = []
    for c in clauses:
        idxs = np.fromiter((abs(l) - 1 for l in c), dtype=np.int64, count=len(c))
        signs = np.array([l > 0 for l in c], dtype=bool)
        clause_idxs.append(idxs)
        clause_signs.append(signs)

    # For speed, also build per-variable clause membership (which clauses contain var).
    # var_clauses[v] = list of clause indices that mention variable v.
    var_clauses = [[] for _ in range(n)]
    for ci, idxs in enumerate(clause_idxs):
        for v in idxs.tolist():
            var_clauses[v].append(ci)

    rng = random.Random(12345)

    def eval_clause(ci, assign):
        idxs = clause_idxs[ci]
        signs = clause_signs[ci]
        vals = assign[idxs]
        return bool(np.any(vals == signs))

    def all_satisfied(assign):
        # vectorized-ish: loop clauses but each is tiny (3 lits)
        for ci in range(m):
            if not eval_clause(ci, assign):
                return False
        return True

    def count_unsat(assign):
        cnt = 0
        for ci in range(m):
            if not eval_clause(ci, assign):
                cnt += 1
        return cnt

    # Phase: last assignment that gave best progress
    phase = [None]

    def init_assignment():
        if phase[0] is not None:
            a = list(phase[0])
        else:
            a = [rng.random() < 0.5 for _ in range(n)]
        return a

    best_assign = None
    best_unsat = m + 1

    def update_best(assign, unsat):
        nonlocal best_assign, best_unsat
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = list(assign)
            phase[0] = list(assign)
        if unsat == 0:
            return True
        return False

    # WalkSAT single run
    def walksat(max_steps, noise):
        assign = init_assignment()
        unsat = count_unsat(assign)
        if update_best(assign, unsat):
            return assign, 0

        # Precompute per-clause quick check using the assignment array.
        # We'll recompute unsat list each step but keep it cheap.
        for step in range(max_steps):
            if time.time() > DEADLINE:
                return assign, unsat
            # find unsat clauses
            unsat_clauses = [ci for ci in range(m) if not eval_clause(ci, assign)]
            if not unsat_clauses:
                return assign, 0
            unsat = len(unsat_clauses)
            if update_best(assign, unsat):
                return assign, 0

            # pick a random unsat clause
            ci = rng.choice(unsat_clauses)
            idxs = clause_idxs[ci]
            signs = clause_signs[ci]
            # choose which literal to flip
            # score each candidate variable by how many currently-unsat other clauses it would satisfy
            # plus a small random noise
            best_var = -1
            best_score = -1e18
            for k in range(len(idxs)):
                v = int(idxs[k])
                # simulate flipping v
                new_val = not assign[v]
                # count clauses that become satisfied
                gain = 0
                for cj in var_clauses[v]:
                    if cj == ci:
                        continue
                    if eval_clause(cj, assign):
                        continue
                    # would this clause be satisfied after flipping v?
                    # check other literals of cj
                    j_idxs = clause_idxs[cj]
                    j_signs = clause_signs[cj]
                    ok = False
                    for t in range(len(j_idxs)):
                        vv = int(j_idxs[t])
                        if vv == v:
                            # this is the variable we flipped
                            if new_val == j_signs[t]:
                                ok = True
                                break
                        else:
                            if assign[vv] == j_signs[t]:
                                ok = True
                                break
                    if ok:
                        gain += 1
                score = gain + rng.random() * noise
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var < 0:
                best_var = int(idxs[0])
            assign[best_var] = not assign[best_var]

        return assign, unsat

    # Restart loop with growing step budgets
    restart = 0
    while time.time() < DEADLINE:
        if restart == 0:
            max_steps = 20000
            noise = 0.5
        elif restart == 1:
            max_steps = 50000
            noise = 0.7
        elif restart == 2:
            max_steps = 100000
            noise = 0.9
        else:
            max_steps = 200000
            noise = 1.0

        assign, unsat = walksat(max_steps, noise)
        if unsat == 0:
            return assign
        restart += 1
        if restart > 20:
            break

    if best_assign is not None:
        return best_assign
    # fallback: all false
    return [False] * n
# EVOLVE-BLOCK-END
