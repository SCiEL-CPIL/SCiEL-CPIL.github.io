"""Improved planted 3-SAT solver: efficient WalkSAT with clause-level bookkeeping,
weighted flip heuristic, randomized restarts, and a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 300.0

    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos_clauses[v].append(ci)
            else:
                neg_clauses[v].append(ci)

    false_count = [0] * m

    def init_false_count(assignment):
        for ci, clause in enumerate(clauses):
            cnt = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] != (lit > 0):
                    cnt += 1
            false_count[ci] = cnt

    def count_unsat():
        return sum(1 for c in false_count if c > 0)

    def flip(var, new_val):
        old_val = not new_val
        delta_unsat = 0
        for ci in pos_clauses[var]:
            if old_val == False and new_val == True:
                if false_count[ci] == 1:
                    delta_unsat -= 1
                false_count[ci] -= 1
            elif old_val == True and new_val == False:
                if false_count[ci] == 0:
                    delta_unsat += 1
                false_count[ci] += 1
        for ci in neg_clauses[var]:
            if old_val == True and new_val == False:
                if false_count[ci] == 1:
                    delta_unsat -= 1
                false_count[ci] -= 1
            elif old_val == False and new_val == True:
                if false_count[ci] == 0:
                    delta_unsat += 1
                false_count[ci] += 1
        return delta_unsat

    best_assignment = None
    best_unsat = m + 1

    def update_best(assignment, current_unsat):
        nonlocal best_assignment, best_unsat
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = list(assignment)
            return current_unsat == 0
        return current_unsat == 0

    max_steps_per_restart = 300000

    while time.time() < deadline:
        rng = random.Random()
        assignment = [rng.random() < 0.5 for _ in range(n)]
        init_false_count(assignment)
        current_unsat = count_unsat()
        if update_best(assignment, current_unsat):
            break

        steps_this_restart = 0
        while steps_this_restart < max_steps_per_restart and time.time() < deadline:
            if current_unsat == 0:
                break
            steps_this_restart += 1

            # Pick an unsatisfied clause, favoring tight ones
            if m < 800:
                unsat_idx = [i for i, c in enumerate(false_count) if c > 0]
                if not unsat_idx:
                    current_unsat = 0
                    break
                min_fc = min(false_count[i] for i in unsat_idx)
                tight = [i for i in unsat_idx if false_count[i] == min_fc]
                ci = rng.choice(tight)
            else:
                ci = -1
                for _ in range(20):
                    cand = rng.randrange(m)
                    if false_count[cand] > 0:
                        ci = cand
                        break
                if ci == -1:
                    current_unsat = 0
                    break

            clause = clauses[ci]
            best_gain = -1e18
            best_var = -1
            for lit in clause:
                v = abs(lit) - 1
                new_val = not assignment[v]
                gain = 0.0
                for cidx in pos_clauses[v]:
                    fc = false_count[cidx]
                    if new_val == True:
                        if fc == 0:
                            gain -= 1.0
                        elif fc == 1:
                            gain += 3.0
                        else:
                            gain += 0.5
                    else:
                        if fc == 0:
                            gain -= 1.0
                        else:
                            gain += 0.2
                for cidx in neg_clauses[v]:
                    fc = false_count[cidx]
                    if new_val == False:
                        if fc == 0:
                            gain -= 1.0
                        elif fc == 1:
                            gain += 3.0
                        else:
                            gain += 0.5
                    else:
                        if fc == 0:
                            gain -= 1.0
                        else:
                            gain += 0.2
                gain += rng.random() * 0.5
                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            if best_var < 0:
                break

            new_val = not assignment[best_var]
            assignment[best_var] = new_val
            current_unsat += flip(best_var, new_val)

            if steps_this_restart % 5000 == 0:
                current_unsat = count_unsat()  # guard against drift

        if update_best(assignment, current_unsat):
            break

    if best_assignment is None:
        best_assignment = [False] * n

    return best_assignment
# EVOLVE-BLOCK-END
