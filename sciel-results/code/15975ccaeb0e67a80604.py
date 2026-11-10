# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 290.0  # safety margin before 330s kill

    # --- Build O(1) index structures ---
    # vcl[v] -> list of (clause_idx, literal) pairs
    vcl = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            vcl[abs(l) - 1].append((i, l))

    # --- Fast WalkSAT with O(1) removal from unsat_list ---
    def walksat(rng, seed_assign, max_steps, p_walk=0.25, no_improve_limit=5000):
        x = [1 if b else 0 for b in seed_assign]
        clause_sat = [0] * m
        for i, c in enumerate(clauses):
            s = 0
            for l in c:
                if x[abs(l) - 1] == (1 if l > 0 else 0):
                    s += 1
            clause_sat[i] = s

        unsat = sum(1 for s in clause_sat if s == 0)
        if unsat == 0:
            return [bool(v) for v in x], 0

        # clause weights: bias toward persistent conflicts
        w = [1.0] * m

        # Maintain unsat_list with O(1) removal using position tracking
        unsat_list = [i for i in range(m) if clause_sat[i] == 0]
        pos_in_unsat = [-1] * m  # pos_in_unsat[i] = index in unsat_list, or -1
        for idx, ci in enumerate(unsat_list):
            pos_in_unsat[ci] = idx

        step = 0
        best_unsat = unsat
        best_x = list(x)
        steps_since_improve = 0

        while step < max_steps:
            if not unsat_list:
                return [bool(v) for v in x], 0
            if time.time() >= deadline:
                break

            # Pick an unsat clause (weighted by w for bias)
            if len(unsat_list) <= 32:
                # Weighted selection for small sets
                tot = 0.0
                picks = []
                for i in unsat_list:
                    tot += w[i]
                    picks.append(tot)
                r = rng.random() * tot
                lo, hi = 0, len(picks) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if picks[mid] < r:
                        lo = mid + 1
                    else:
                        hi = mid
                ci = unsat_list[lo]
            else:
                ci = unsat_list[rng.randrange(len(unsat_list))]

            clause = clauses[ci]
            # Choose literal to flip
            if rng.random() < p_walk:
                l = clause[rng.randrange(3)]
            else:
                # Min-conflicts: pick literal with best delta
                best_l = clause[0]
                best_delta = None
                for l in clause:
                    v = abs(l) - 1
                    d = 0
                    for (ci2, l2) in vcl[v]:
                        if clause_sat[ci2] == 0:
                            nv = 1 - x[v]
                            if nv == (1 if l2 > 0 else 0):
                                d -= 1
                        else:
                            if clause_sat[ci2] == 1 and x[v] == (1 if l2 > 0 else 0):
                                d += 1
                    if best_delta is None or d < best_delta:
                        best_delta = d
                        best_l = l
                l = best_l

            v = abs(l) - 1
            old = x[v]
            x[v] = 1 - old

            # Update clause_sat and maintain unsat_list with O(1) removal
            for (ci2, l2) in vcl[v]:
                before = clause_sat[ci2]
                now_match = (x[v] == (1 if l2 > 0 else 0))
                before_match = (old == (1 if l2 > 0 else 0))
                if before_match and not now_match:
                    clause_sat[ci2] = before - 1
                elif (not before_match) and now_match:
                    clause_sat[ci2] = before + 1

                if before == 0 and clause_sat[ci2] > 0:
                    # Clause became satisfied: remove from unsat_list in O(1)
                    pos = pos_in_unsat[ci2]
                    if pos != -1:
                        last = unsat_list[-1]
                        unsat_list[pos] = last
                        pos_in_unsat[last] = pos
                        unsat_list.pop()
                        pos_in_unsat[ci2] = -1
                    unsat -= 1
                elif before > 0 and clause_sat[ci2] == 0:
                    # Clause became unsatisfied: add to unsat_list
                    pos_in_unsat[ci2] = len(unsat_list)
                    unsat_list.append(ci2)
                    unsat += 1
                    w[ci2] += 0.3  # Increase weight for persistent conflicts

            if unsat < best_unsat:
                best_unsat = unsat
                best_x = list(x)
                steps_since_improve = 0
            else:
                steps_since_improve += 1

            # No-improvement restart within this run
            if steps_since_improve >= no_improve_limit:
                break

            step += 1

        if unsat == 0:
            return [bool(v) for v in x], 0
        # Return the best assignment found during this run
        return [bool(v) for v in best_x], best_unsat

    # --- Restart strategy ---
    rng = random.Random(42)
    base = [rng.random() < 0.5 for _ in range(n)]

    def count_unsat(assign):
        x = [1 if b else 0 for b in assign]
        c = 0
        for cl in clauses:
            s = 0
            for l in cl:
                if x[abs(l) - 1] == (1 if l > 0 else 0):
                    s += 1
            if s == 0:
                c += 1
        return c

    best_assign = None
    best_unsat = None

    # First run with generous budget
    res, u = walksat(rng, base, 300000, p_walk=0.25, no_improve_limit=10000)
    if u == 0:
        return res
    best_assign = res
    best_unsat = u

    # Restart loop with perturbation
    restart_budget = 80000
    for r in range(500):
        if time.time() >= deadline:
            break
        # Perturb current best with increasing number of flips for diversity
        num_flips = 2 + (r % 5)  # 2-6 flips
        cand = list(best_assign)
        for _ in range(num_flips):
            cand[rng.randrange(n)] = not cand[rng.randrange(n)]
        
        # Vary p_walk slightly for diversity
        p_walk = 0.2 + 0.1 * (r % 3)  # 0.2, 0.3, 0.4
        
        res, u = walksat(rng, cand, restart_budget, p_walk=p_walk, no_improve_limit=5000)
        if u == 0:
            return res
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best_assign = res

    return best_assign if best_assign is not None else [False] * n
# EVOLVE-BLOCK-END
