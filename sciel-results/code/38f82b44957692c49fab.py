"""Rewritten planted 3-SAT solver: indexed WalkSAT with restarts + adaptive weighting."""
import time
import random

_DEADLINE = 320.0  # safety margin below the 330s kill


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    rng = random.Random(0)
    t0 = time.time()

    # Pre-compute clause literals as (var_idx, sign); sign=+1 => positive literal.
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        lst = []
        for lit in cl:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            lst.append((vi, sign))
            var_to_clauses[vi].append(ci)
        clause_vars.append(lst)

    assignment = [0] * n
    cw = [1.0] * m  # per-clause selection weight

    def is_sat(ci):
        for vi, sign in clause_vars[ci]:
            if (assignment[vi] > 0) == (sign > 0):
                return True
        return False

    def restart():
        for vi in range(n):
            assignment[vi] = 1 if rng.random() < 0.5 else 0

    best_assignment = None
    best_unsat = m + 1
    step = 0

    while time.time() - t0 < _DEADLINE:
        restart()
        for i in range(m):
            cw[i] = 1.0

        # Build unsat list + membership set.
        unsat_set = set()
        unsat_list = []
        for ci in range(m):
            if not is_sat(ci):
                unsat_set.add(ci)
                unsat_list.append(ci)
        unsat_count = len(unsat_list)
        step = 0

        if unsat_count == 0:
            return [bool(x) for x in assignment]
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        while time.time() - t0 < _DEADLINE and unsat_count > 0:
            # Pick an unsatisfied clause.
            cl_idx = unsat_list[rng.randrange(len(unsat_list))]
            lst = clause_vars[cl_idx]
            vi, sign = lst[rng.randrange(len(lst))]
            assignment[vi] = 1 - assignment[vi]

            # Update membership for clauses containing vi.
            for ci in var_to_clauses[vi]:
                if is_sat(ci):
                    if ci in unsat_set:
                        unsat_set.discard(ci)
                        unsat_list.remove(ci)
                        unsat_count -= 1
                        cw[ci] = 1.0
                else:
                    if ci not in unsat_set:
                        unsat_set.add(ci)
                        unsat_list.append(ci)
                        unsat_count += 1
                        cw[ci] = min(cw[ci] * 1.5, 1000.0)

            step += 1
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Stuck detection: restart if no progress for a while.
            if step % 3000 == 0 and unsat_count > best_unsat + 5:
                break

    return [bool(x) for x in best_assignment]
