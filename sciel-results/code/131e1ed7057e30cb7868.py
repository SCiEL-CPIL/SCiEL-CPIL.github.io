import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 315.0  # safety margin before 330s kill

    # ---- Preprocess: vectorized literal representation ----
    # var_idx[i,j] = 0-based variable; sign[i,j] = 1 if positive literal, 0 if negative
    var_idx = np.empty((m, 3), dtype=np.int32)
    sign = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            var_idx[i, j] = abs(l) - 1
            sign[i, j] = 1 if l > 0 else 0

    # variable -> list of (clause, position) for incremental updates
    var_clauses = [[] for _ in range(n)]
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            var_clauses[v].append(i)
            var_pos[v].append(j)

    # Precompute per-variable clause arrays for fast updates
    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.empty(0, dtype=np.int32)
                      for vc in var_clauses]
    var_pos_np = [np.array(vp, dtype=np.int32) if vp else np.empty(0, dtype=np.int32)
                  for vp in var_pos]

    # Polarity counts for initialization: for each variable, count how many
    # clauses want it True (positive) vs False (negative)
    want_true = np.zeros(n, dtype=np.int32)
    want_false = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            if sign[i, j] == 1:
                want_true[v] += 1
            else:
                want_false[v] += 1

    def init_assignment(rng):
        """Polarity-biased random initialization."""
        assignment = np.empty(n, dtype=np.int8)
        for v in range(n):
            if want_true[v] > want_false[v]:
                # biased toward True
                assignment[v] = 1 if rng.random() < 0.7 else 0
            elif want_false[v] > want_true[v]:
                assignment[v] = 0 if rng.random() < 0.7 else 1
            else:
                assignment[v] = 1 if rng.random() < 0.5 else 0
        return assignment

    def init_num_true(assignment):
        """Number of satisfied literals per clause."""
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def flip_var(v, assignment, num_true):
        """Flip variable v; update num_true incrementally."""
        assignment[v] = 1 - assignment[v]
        vcs = var_clauses_np[v]
        vps = var_pos_np[v]
        if len(vcs) == 0:
            return
        # For each clause containing v, check if the literal is now satisfied
        new_vals = assignment[v]
        # satisfied at position j in clause c if assignment[v] == sign[c,j]
        s = sign[vcs, vps]  # shape (len,)
        was_sat = (new_vals == s)
        # We need the old state to compute delta. Simpler: recompute contribution.
        # num_true[c] changes by +1 if now satisfied and wasn't, -1 if was and isn't.
        # We don't track per-literal old state, so recompute: new count for these clauses
        # = (other literals satisfied) + (this literal satisfied)
        # other literals: for clause c, sum over the other 2 positions
        # Efficient: recompute full count for affected clauses
        for idx in range(len(vcs)):
            c = int(vcs[idx])
            # recompute satisfied count for clause c
            row_sat = (assignment[var_idx[c]] == sign[c])
            num_true[c] = row_sat.sum()

    def unit_propagate(assignment, num_true):
        """Set variables in unit clauses to satisfy them. Returns changed count."""
        changed = 0
        for _ in range(3 * n + 10):
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                sat_pos = np.where((assignment[var_idx[c]] == sign[c]))[0]
                if sat_pos.size == 0:
                    continue
                p = int(sat_pos[0])
                v = int(var_idx[c, p])
                target = int(sign[c, p])
                if int(assignment[v]) == target:
                    continue
                # flip v to target
                assignment[v] = target
                changed += 1
                progress = True
                # recompute num_true for all clauses containing v
                vcs = var_clauses_np[v]
                for k in range(len(vcs)):
                    cc = int(vcs[k])
                    num_true[cc] = (assignment[var_idx[cc]] == sign[cc]).sum()
                # recompute unit set
                break
            if not progress:
                break
        return changed

    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(12345)
    max_restarts = 400

    for restart in range(max_restarts):
        if time.time() >= deadline:
            break

        assignment = init_assignment(rng)
        num_true = init_num_true(assignment)

        # Unit propagation to exploit planted structure
        unit_propagate(assignment, num_true)

        cur_unsat = int((num_true == 0).sum())
        if cur_unsat == 0:
            return assignment.tolist()
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment.copy()

        # WalkSAT with hybrid heuristic
        max_flips = 4000
        for flip_i in range(max_flips):
            if time.time() >= deadline:
                break

            unsat_idx = np.where(num_true == 0)[0]
            if len(unsat_idx) == 0:
                return assignment.tolist()

            cur_unsat = len(unsat_idx)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment.copy()

            # Pick a random unsatisfied clause
            ci = int(unsat_idx[rng.randrange(len(unsat_idx))])

            # For each of the 3 vars in this clause, score = number of OTHER
            # unsatisfied clauses containing that var (classic WalkSAT weight)
            best_var = -1
            best_score = -1e9
            for j in range(3):
                v = int(var_idx[ci, j])
                vcs = var_clauses_np[v]
                if len(vcs) == 0:
                    score = 0.0
                else:
                    score = float((num_true[vcs] == 0).sum()) - 1.0  # exclude self
                # add noise for stochasticity (WalkSAT-style)
                score += rng.random() * 0.5
                if score > best_score:
                    best_score = score
                    best_var = v

            if best_var >= 0:
                flip_var(best_var, assignment, num_true)

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
