# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before the 300s/330s kill

    rng = random.Random(0)

    # ---- Convert to 0-based signed literals, build clause matrix (-1..1) ----
    # For each clause: literal l -> var = abs(l)-1, sign = +1 if l>0 else -1.
    # A clause is satisfied iff sum over its literals of (sign * a[var]) >= 1.
    # We store C[clause, var] = sign (+1/-1) for present literals, 0 otherwise.
    # Then C @ a gives, per clause, sum(sign*a). Satisfied iff that sum >= 1.
    C = np.zeros((len(clauses), n), dtype=np.int8)
    for ci, cl in enumerate(clauses):
        for l in cl:
            var = abs(l) - 1
            C[ci, var] = 1 if l > 0 else -1

    # ---- Preprocessing: tautology removal, unit propagation, pure literals ----
    # Work on a working copy; keep a map from original var -> working var (or None if fixed).
    # We'll track fixed assignments for removed variables.
    fixed = {}  # var -> bool (True=+1, False=-1) for variables eliminated
    active_vars = list(range(n))  # working indices are positions in this list

    def rebuild_matrix():
        # Build C over active_vars
        m = len(active_vars)
        varpos = {v: p for p, v in enumerate(active_vars)}
        Cw = np.zeros((len(clauses), m), dtype=np.int8)
        for ci, cl in enumerate(clauses):
            for l in cl:
                var = abs(l) - 1
                p = varpos.get(var)
                if p is None:
                    continue
                Cw[ci, p] = 1 if l > 0 else -1
        return Cw

    # Iteratively apply unit propagation and pure literal elimination.
    # Rebuild Cw each round (cheap for 150 vars).
    for _ in range(50):
        Cw = rebuild_matrix()
        m = len(active_vars)
        # satisfaction vector: Cw @ a  (a in {-1,+1})  -> but we don't have a yet.
        # Instead detect tautologies (both signs of a var present), units, pure.
        # For each clause, check tautology.
        changed = False
        # tautology: any clause containing both +v and -v
        for ci in range(len(clauses)):
            row = Cw[ci]
            nz = np.nonzero(row)[0]
            if len(nz) == 0:
                # empty clause -> unsat (shouldn't happen for planted); mark
                pass
            signs = row[nz]
            if (signs > 0).any() and (signs < 0).any():
                # tautology: clause always satisfied; zero it out
                Cw[ci] = 0
                changed = True
        # unit propagation: clause with exactly one literal
        for ci in range(len(clauses)):
            row = Cw[ci]
            nz = np.nonzero(row)[0]
            if len(nz) == 1:
                p = nz[0]
                var = active_vars[p]
                val = 1 if row[p] > 0 else -1  # needed value for var
                if var not in fixed:
                    fixed[var] = (val == 1)
                    active_vars.remove(var)
                    changed = True
                else:
                    # already fixed; check consistency
                    cur = 1 if fixed[var] else -1
                    if cur != val:
                        # conflict: instance unsat (shouldn't happen); stop
                        break
        # pure literal elimination: a var appears with only one sign across all clauses
        if changed:
            Cw = rebuild_matrix()
            m = len(active_vars)
            if m > 0:
                # column-wise: for each var position, look at nonzero signs
                cols = Cw[:, :]
                nz_mask = cols != 0
                col_sum = cols.sum(axis=0)  # sum of signs
                col_count = nz_mask.sum(axis=0)
                for p in range(m):
                    if col_count[p] > 0 and (col_count[p] == abs(col_sum[p])):
                        # pure: all occurrences same sign
                        var = active_vars[p]
                        val = 1 if col_sum[p] > 0 else -1
                        fixed[var] = (val == 1)
                        active_vars.remove(var)
                        changed = True
                        break  # rebuild and continue
                if changed:
                    continue
        if not changed:
            break

    # After preprocessing, build final working matrix over active_vars.
    Cw = rebuild_matrix()
    m = len(active_vars)
    num_clauses = len(clauses)

    # Initialize assignment for active vars randomly.
    a = np.array([1 if rng.random() < 0.5 else -1 for _ in range(m)], dtype=np.int8)

    # Helper to evaluate unsatisfied count and the satisfied mask.
    def eval_state(a):
        sat = (Cw @ a) >= 1
        return sat

    # Track best.
    best_a = a.copy()
    best_unsat = None

    def unsat_count(sat):
        return int((~sat).sum())

    sat = eval_state(a)
    best_unsat = unsat_count(sat)

    # WalkSAT with dynamic noise, restarts, growing budget, deadline guard.
    # Precompute per-clause literal positions for fast flip.
    # lit_pos[ci] = list of (p, sign) where Cw[ci,p]!=0
    lit_pos = []
    for ci in range(num_clauses):
        row = Cw[ci]
        nz = np.nonzero(row)[0]
        lit_pos.append([(int(p), int(row[p])) for p in nz])

    # clause weights for weighted selection
    weights = np.ones(num_clauses, dtype=np.float64)

    # Precompute, for each var position p, the list of (ci, sign) where it appears.
    var_clauses = [[] for _ in range(m)]
    for ci in range(num_clauses):
        for p, s in lit_pos[ci]:
            var_clauses[p].append((ci, s))

    restart = 0
    budget = 2000
    max_budget = 200000

    while True:
        if time.time() > deadline:
            break
        # run a walk of `budget` flips
        noise = 0.5  # start noise
        for step in range(budget):
            if time.time() > deadline:
                break
            sat = eval_state(a)
            uc = unsat_count(sat)
            if uc < best_unsat:
                best_unsat = uc
                best_a = a.copy()
            if uc == 0:
                # found a satisfying assignment for working vars
                a = a.copy()
                # map back
                break
            # pick an unsatisfied clause
            unsat_idx = np.nonzero(~sat)[0]
            ci = int(rng.choice(unsat_idx))
            # decay noise over the run
            noise = max(0.05, 0.5 * (1.0 - step / max(1, budget)))
            # choose a literal to flip
            lits = lit_pos[ci]
            if rng.random() < noise:
                p, s = lits[rng.randrange(len(lits))]
            else:
                # weighted by gain: flipping p changes satisfaction of its clauses
                # pick literal maximizing gain (number of clauses that would become satisfied)
                best_p = -1
                best_gain = -1
                for p, s in lits:
                    # current a[p]
                    ap = a[p]
                    # after flip, this clause's sum changes by -2*s*ap... we just count gain
                    gain = 0
                    for (cj, sj) in var_clauses[p]:
                        # clause cj currently: sat[cj]
                        # flipping p changes cj sum by -2*sj*ap
                        # if currently unsat and would become sat, +1
                        if not sat[cj]:
                            # new sum = old sum - 2*sj*ap; if old sum was 0 (unsat, since sat needs >=1)
                            # it becomes satisfied iff new sum >= 1
                            # old sum = Cw[cj] @ a ; but we only know sat mask. Approx:
                            # gain if this literal is currently "helping" vs "hurting"
                            # simpler: if s*ap == -1 (literal currently false in clause), flipping makes it true
                            if s * ap == -1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p >= 0:
                    p = best_p
                else:
                    p, s = lits[rng.randrange(len(lits))]
            a[p] = -a[p]

        # after a run, check if solved
        sat = eval_state(a)
        uc = unsat_count(sat)
        if uc < best_unsat:
            best_unsat = uc
            best_a = a.copy()
        if uc == 0:
            best_a = a.copy()
            break
        # restart: new random assignment, grow budget
        a = np.array([1 if rng.random() < 0.5 else -1 for _ in range(m)], dtype=np.int8)
        budget = min(max_budget, int(budget * 1.5))
        restart += 1
        # periodic full re-randomization is already done

    # Map best working assignment back to full original assignment.
    varpos = {v: p for p, v in enumerate(active_vars)}
    full = [False] * n
    for v in range(n):
        if v in fixed:
            full[v] = fixed[v]
        else:
            p = varpos[v]
            full[v] = (best_a[p] == 1)
    return full
# EVOLVE-BLOCK-END
