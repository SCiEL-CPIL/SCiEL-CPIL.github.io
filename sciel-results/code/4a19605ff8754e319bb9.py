"""Rewritten planted 3-SAT solver: efficient incremental WalkSAT with restarts and a deadline guard."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    m = len(clauses)

    # Precompute clause literal info (0-based var, is_pos) and variable -> clause incidence.
    clause_vars = []   # list of lists of (var, is_pos)
    var_clauses = [[] for _ in range(n)]  # var -> list of (clause_idx, is_pos)
    for ci, c in enumerate(clauses):
        lits = []
        for lit in c:
            v = abs(lit) - 1
            is_pos = lit > 0
            lits.append((v, is_pos))
            var_clauses[v].append((ci, is_pos))
        clause_vars.append(lits)

    deadline = time.time() + 280.0  # safety margin before the 330s kill
    best_assignment = None
    best_unsat = m + 1

    walk_p = 0.5          # probability of a random "walk" flip
    restart_window = 1500 # steps without improvement before a restart
    max_restarts = 500

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count = [0] * m
        for ci in range(m):
            cnt = 0
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    cnt += 1
            sat_count[ci] = cnt
        n_unsat = sum(1 for s in sat_count if s == 0)
        if n_unsat == 0:
            return assignment
        if n_unsat < best_unsat:
            best_unsat = n_unsat
            best_assignment = assignment[:]

        last_improvement = n_unsat
        no_improve = 0
        while True:
            if time.time() > deadline:
                break
            if no_improve >= restart_window:
                break

            # Pick a random unsatisfied clause.
            unsat_list = [ci for ci in range(m) if sat_count[ci] == 0]
            if not unsat_list:
                return assignment
            ci = unsat_list[rng.randrange(len(unsat_list))]
            lits = clause_vars[ci]

            # Choose which variable to flip.
            if rng.random() < walk_p:
                v, is_pos = lits[rng.randrange(len(lits))]
            else:
                # Flip the literal whose flip minimizes damage (clauses going unsat).
                best_var = None
                best_damage = None
                for v, is_pos in lits:
                    damage = 0
                    for (c2, pos2) in var_clauses[v]:
                        # Damage if this clause is currently satisfied only by v's literal.
                        if sat_count[c2] == 1 and assignment[v] == pos2:
                            damage += 1
                    if best_damage is None or damage < best_damage:
                        best_damage = damage
                        best_var = v
                v = best_var

            # Flip v and update sat_count / n_unsat incrementally.
            for (c2, pos2) in var_clauses[v]:
                before = sat_count[c2]
                after = before - 1 if assignment[v] == pos2 else before + 1
                sat_count[c2] = after
                if before == 0 and after != 0:
                    n_unsat -= 1
                elif before != 0 and after == 0:
                    n_unsat += 1
            assignment[v] = not assignment[v]

            if n_unsat == 0:
                return assignment
            if n_unsat < best_unsat:
                best_unsat = n_unsat
                best_assignment = assignment[:]
            if n_unsat < last_improvement:
                last_improvement = n_unsat
                no_improve = 0
            else:
                no_improve += 1

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
