"""DPLL + WalkSAT hybrid with unit propagation, pure literal elimination,
and clause simplification for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 275  # safety margin before 300s kill

    # Precompute: for each variable, which clauses contain it (and with what sign)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append((ci, lit > 0))

    # ---------- DPLL with unit propagation and pure literal elimination ----------

    def dpll_search(clauses_copy, assignment, deadline):
        """Complete DPLL search. Returns (success, assignment) or (None, None)."""
        # assignment: list of (None, True, False) for unassigned/true/false
        # clauses_copy: list of lists of literals (signed, 1-based)

        # Pure literal elimination + unit propagation (iterative)
        while True:
            # Remove satisfied clauses
            new_clauses = []
            for cl in clauses_copy:
                sat = False
                for lit in cl:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val is not None and (val == (lit > 0)):
                        sat = True
                        break
                if not sat:
                    new_clauses.append(cl)
            clauses_copy = new_clauses

            if not clauses_copy:
                return True, assignment

            # Check for empty clause (contradiction)
            for cl in clauses_copy:
                if not cl:
                    return False, None

            # Pure literal elimination
            var_polarity = {}  # var_idx -> set of polarities seen
            for cl in clauses_copy:
                for lit in cl:
                    v = abs(lit) - 1
                    pol = 1 if lit > 0 else -1
                    if v not in var_polarity:
                        var_polarity[v] = set()
                    var_polarity[v].add(pol)

            pure_found = True
            while pure_found:
                pure_found = False
                for v, pols in list(var_polarity.items()):
                    if len(pols) == 1 and assignment[v] is None:
                        # Pure literal
                        pol = list(pols)[0]
                        assignment[v] = (pol == 1)
                        pure_found = True
                        # Rebuild var_polarity after assignment
                        var_polarity = {}
                        for cl in clauses_copy:
                            for lit in cl:
                                vv = abs(lit) - 1
                                if assignment[vv] is None:
                                    pp = 1 if lit > 0 else -1
                                    if vv not in var_polarity:
                                        var_polarity[vv] = set()
                                    var_polarity[vv].add(pp)
                        break
                if pure_found:
                    # Re-filter clauses
                    new_clauses = []
                    for cl in clauses_copy:
                        sat = False
                        for lit in cl:
                            v = abs(lit) - 1
                            val = assignment[v]
                            if val is not None and (val == (lit > 0)):
                                sat = True
                                break
                        if not sat:
                            new_clauses.append(cl)
                    clauses_copy = new_clauses
                    if not clauses_copy:
                        return True, assignment
                    for cl in clauses_copy:
                        if not cl:
                            return False, None
                    break  # restart outer loop for unit prop

            # Unit propagation
            unit_found = True
            while unit_found:
                unit_found = False
                new_clauses = []
                for cl in clauses_copy:
                    if len(cl) == 1:
                        lit = cl[0]
                        v = abs(lit) - 1
                        if assignment[v] is None:
                            assignment[v] = (lit > 0)
                            unit_found = True
                            # Skip this clause (now satisfied)
                            continue
                        elif assignment[v] == (lit > 0):
                            continue  # satisfied
                        else:
                            return False, None
                    new_clauses.append(cl)
                clauses_copy = new_clauses
                if not clauses_copy:
                    return True, assignment

            if time.time() > deadline:
                return None, None

            # Variable selection: pick variable appearing in most clauses
            var_count = {}
            for cl in clauses_copy:
                seen = set()
                for lit in cl:
                    v = abs(lit) - 1
                    if v not in seen:
                        var_count[v] = var_count.get(v, 0) + 1
                        seen.add(v)

            if not var_count:
                return True, assignment

            # Pick variable with max occurrence
            best_var = max(var_count, key=lambda v: var_count[v])

            # Try both branches
            for val in [True, False]:
                assignment_copy = assignment[:]
                assignment_copy[best_var] = val

                # Simplify clauses
                simplified = []
                for cl in clauses_copy:
                    new_cl = []
                    sat = False
                    for lit in cl:
                        v = abs(lit) - 1
                        if assignment_copy[v] is not None:
                            if assignment_copy[v] == (lit > 0):
                                sat = True
                                break
                            # else: literal is false, remove it
                        else:
                            new_cl.append(lit)
                    if not sat:
                        simplified.append(new_cl)

                if not simplified:
                    return True, assignment_copy

                # Check for empty clause
                has_empty = any(not cl for cl in simplified)
                if has_empty:
                    continue

                # Recurse
                result, final_assign = dpll_search(simplified, assignment_copy, deadline)
                if result is True:
                    return True, final_assign
                elif result is None:
                    return None, None

            return False, None

    # ---------- WalkSAT (fast, for fallback) ----------

    def make_satisfied_array(assignment):
        sat = [False] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat[ci] = True
                    break
        return sat

    def count_unsat(sat):
        return sum(1 for s in sat if not s)

    def flip_var(assignment, var_idx, sat):
        assignment[var_idx] = not assignment[var_idx]
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == assignment[var_idx]:
                sat[ci] = True
            else:
                clause = clauses[ci]
                still_sat = False
                for lit in clause:
                    if lit == var_idx + 1 or lit == -(var_idx + 1):
                        continue
                    if assignment[abs(lit) - 1] == (lit > 0):
                        still_sat = True
                        break
                sat[ci] = still_sat

    def walksat_step(assignment, sat, rng, p_walk):
        unsat = [ci for ci in range(m) if not sat[ci]]
        if not unsat:
            return
        ci = rng.choice(unsat)
        clause = clauses[ci]
        if rng.random() < p_walk:
            best_var = -1
            best_score = -10**9
            for lit in clause:
                v = abs(lit) - 1
                gains = 0
                losses = 0
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if is_pos2 == assignment[v]:
                        clause2 = clauses[c2]
                        other_sat = False
                        for lit2 in clause2:
                            if lit2 == lit or lit2 == -lit:
                                continue
                            if assignment[abs(lit2) - 1] == (lit2 > 0):
                                other_sat = True
                                break
                        if not other_sat:
                            losses += 1
                    else:
                        gains += 1
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat)
            else:
                lit = rng.choice(clause)
                flip_var(assignment, abs(lit) - 1, sat)
        else:
            lit = rng.choice(clause)
            flip_var(assignment, abs(lit) - 1, sat)

    # ---------- Main: try DPLL first, then WalkSAT ----------

    # Phase 1: DPLL with time limit
    dpll_deadline = time.time() + 180  # give DPLL most of the budget
    init_assignment = [None] * n
    dpll_result, dpll_assign = dpll_search([cl[:] for cl in clauses], init_assignment, dpll_deadline)

    if dpll_result is True and dpll_assign is not None:
        # Convert to boolean list
        return [bool(v) for v in dpll_assign]

    # Phase 2: WalkSAT fallback with remaining time
    remaining_deadline = deadline
    best_assignment = None
    best_unsat = m + 1

    for restart in range(100):
        if time.time() > remaining_deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat = make_satisfied_array(assignment)
        cur_unsat = count_unsat(sat)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        max_steps = 3000
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.6

        for step in range(max_steps):
            if time.time() > remaining_deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            walksat_step(assignment, sat, rng, p_walk)

            if step % 50 == 0:
                cur_unsat = count_unsat(sat)
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    p_walk = max(0.3, p_walk - 0.02)
                else:
                    stall_count += 1
                prev_unsat = cur_unsat

            if stall_count > 20:
                break

    if best_assignment:
        return best_assignment

    # Last resort: all False
    return [False] * n
# EVOLVE-BLOCK-END
