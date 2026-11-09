"""WalkSAT with clause learning, unit propagation, and adaptive restarts."""
import random
import time
import math


def _unit_propagate(assignment, clause_sat, var_occurs, clauses, n):
    """Perform unit propagation. Returns (assignment, clause_sat, solved, conflict)."""
    changed = True
    while changed:
        changed = False
        for ci in range(len(clauses)):
            if clause_sat[ci] == 1:
                # Unit clause: exactly one unsatisfied literal
                cl = clauses[ci]
                for lit in cl:
                    v = abs(lit) - 1
                    sign = 1 if lit > 0 else -1
                    desired = 1 if sign == 1 else 0
                    if assignment[v] != desired:
                        assignment[v] = desired
                        # Update all clauses containing v
                        for (cci, sgn) in var_occurs[v]:
                            old_val = 1 if (assignment[v] == (1 if sgn == 1 else 0)) else 0
                            # Actually we just changed it, so compute delta
                            # The old value was 1 - desired for this var in this context
                            # Let's just recompute contribution
                            pass
                        # Simpler: recompute clause_sat for all clauses with v
                        for (cci, sgn) in var_occurs[v]:
                            # Recompute c_sat[cci]
                            new_sat = 0
                            for lit2 in clauses[cci]:
                                v2 = abs(lit2) - 1
                                sgn2 = 1 if lit2 > 0 else -1
                                desired2 = 1 if sgn2 == 1 else 0
                                if assignment[v2] == desired2:
                                    new_sat += 1
                            clause_sat[cci] = new_sat
                        changed = True
                        break
            elif clause_sat[ci] == 0:
                # Check if all literals are satisfied? No, 0 means none satisfied
                # But wait, we need to check for conflict: empty clause or all false
                # clause_sat == 0 means no literal is true, which is a conflict if clause is non-empty
                # Actually for 3-SAT, if clause_sat == 0, the clause is unsatisfied
                # This isn't a conflict in the DPLL sense unless we've assigned all vars in it
                # For unit propagation, we only act on unit clauses (sat count == 1)
                pass
    return assignment, clause_sat


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Pre-compute: for each variable, list of (clause_index, sign) where sign is +1 if lit == var, -1 if lit == -var
    var_occurs = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_occurs[abs(lit) - 1].append((ci, 1 if lit > 0 else -1))

    deadline = time.time() + 280  # 300s limit, keep 20s safety margin

    best_assignment = None
    best_unsat_count = m + 1

    # Learned clauses: list of (clause, weight) where weight decays
    learned_clauses = []
    max_learned = 200

    rng = random.Random()

    max_steps_per_restart = 30000
    restart_budget = 10000  # max total restarts (will be limited by time)

    alpha = 0.4  # noise level

    for restart in range(restart_budget):
        if time.time() > deadline - 2:
            break

        # Random initial assignment
        assignment = [random.getrandbits(1) for _ in range(n)]

        # Compute initial clause satisfaction counts
        clause_sat = [0] * m
        for ci, cl in enumerate(clauses):
            s = 0
            for lit in cl:
                if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                    s += 1
            clause_sat[ci] = s

        # Also compute satisfaction for learned clauses
        learned_sat = [0] * len(learned_clauses)
        for li, (lcl, lw) in enumerate(learned_clauses):
            s = 0
            for lit in lcl:
                if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                    s += 1
            learned_sat[li] = s

        unsat_count = sum(1 for s in clause_sat if s == 0)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)

        steps = 0
        max_steps = max_steps_per_restart

        assign = assignment
        c_sat = clause_sat
        n_vars = n
        var_occ = var_occurs
        l_sat = learned_sat
        l_clauses = [lc for lc, lw in learned_clauses]
        l_weights = [lw for lc, lw in learned_clauses]
        n_learned = len(learned_clauses)

        # Track recent flips for learning
        flip_history = []  # list of (var, old_val, new_val)
        max_history = 20

        while steps < max_steps:
            if time.time() > deadline - 2:
                break

            # Pick an unsat clause (from original or learned)
            # Weighted selection: original clauses have weight 1, learned have their weight
            # For simplicity, pick from original clauses first
            tries = 0
            ci = -1
            while True:
                ci = random.randrange(m)
                if c_sat[ci] == 0:
                    break
                tries += 1
                if tries > 50:
                    # Fallback: scan
                    unsat_list = [i for i in range(m) if c_sat[i] == 0]
                    if not unsat_list:
                        break
                    ci = random.choice(unsat_list)
                    break

            if c_sat[ci] > 0:
                # Check if all satisfied
                all_sat = True
                for i in range(m):
                    if c_sat[i] == 0:
                        all_sat = False
                        break
                if all_sat:
                    break
                continue

            cl = clauses[ci]

            # Compute scores for each literal in the clause
            scores = []
            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1

                gain = 0
                loss = 0
                for (cci, sgn) in var_occ[v]:
                    if cci == ci:
                        continue
                    currently_sat = c_sat[cci] > 0
                    cur_val = 1 if (assign[v] == (1 if sgn == 1 else 0)) else 0
                    new_val = 1 if ((1 - assign[v]) == (1 if sgn == 1 else 0)) else 0

                    if cur_val == 0 and new_val == 1:
                        if not currently_sat:
                            gain += 1
                    elif cur_val == 1 and new_val == 0:
                        if c_sat[cci] == 1:
                            loss += 1

                # Add learned clause penalty
                for li in range(n_learned):
                    if l_sat[li] > 0:
                        continue  # Already satisfied
                    lcl = l_clauses[li]
                    # Check if flipping v affects this learned clause
                    affected = False
                    for llit in lcl:
                        if abs(llit) - 1 == v:
                            affected = True
                            break
                    if affected:
                        loss += l_weights[li]

                score = gain - loss
                scores.append((score, v, sign))

            # Noisy selection
            r = random.random()
            if r < alpha:
                idx = random.randrange(len(cl))
                _, v, sign = scores[idx]
            else:
                # Pick the one with max score, with random tie-breaking among top
                max_score = max(s[0] for s in scores)
                top = [i for i, s in enumerate(scores) if s[0] == max_score]
                idx = random.choice(top)
                _, v, sign = scores[idx]

            # Record flip
            old_val = assign[v]
            new_val = 1 - old_val
            assign[v] = new_val
            flip_history.append((v, old_val, new_val))
            if len(flip_history) > max_history:
                flip_history.pop(0)

            # Update clause_sat for all clauses containing v
            for (cci, sgn) in var_occ[v]:
                old_contrib = 1 if (old_val == (1 if sgn == 1 else 0)) else 0
                new_contrib = 1 if (new_val == (1 if sgn == 1 else 0)) else 0
                c_sat[cci] += new_contrib - old_contrib

            # Update learned clause satisfaction
            for li in range(n_learned):
                lcl = l_clauses[li]
                for llit in lcl:
                    if abs(llit) - 1 == v:
                        sgn2 = 1 if llit > 0 else -1
                        old_contrib = 1 if (old_val == (1 if sgn2 == 1 else 0)) else 0
                        new_contrib = 1 if (new_val == (1 if sgn2 == 1 else 0)) else 0
                        l_sat[li] += new_contrib - old_contrib
                        break

            steps += 1

            # Periodic check
            if steps % 20 == 0:
                unsat_count = sum(1 for s in c_sat if s == 0)
                if unsat_count == 0:
                    break
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = list(assign)

                # Learn a clause if stuck
                if unsat_count > best_unsat_count * 0.8 and steps > 1000:
                    # Collect unsat clauses
                    unsat_cis = [i for i in range(m) if c_sat[i] == 0]
                    if unsat_cis:
                        # Pick a random unsat clause as seed
                        seed_ci = random.choice(unsat_cis)
                        seed_cl = clauses[seed_ci]
                        # Build a learned clause from literals that are currently false
                        # and were recently flipped
                        learned_lits = []
                        for lit in seed_cl:
                            v = abs(lit) - 1
                            # Check if this var was recently flipped
                            for (fv, fo, fn) in reversed(flip_history):
                                if fv == v:
                                    # The literal that is currently false
                                    # If lit is positive and assign[v] is False, keep lit
                                    # If lit is negative and assign[v] is True, keep -lit
                                    if lit > 0 and assign[v] == 0:
                                        learned_lits.append(lit)
                                    elif lit < 0 and assign[v] == 1:
                                        learned_lits.append(lit)
                                    break
                        if len(learned_lits) >= 2 and len(learned_lits) <= 5:
                            # Check if this learned clause is not already in learned_clauses
                            dup = False
                            for (lc, lw) in learned_clauses:
                                if set(lc) == set(learned_lits):
                                    dup = True
                                    break
                            if not dup:
                                if len(learned_clauses) >= max_learned:
                                    # Remove oldest
                                    learned_clauses.pop(0)
                                learned_clauses.append((learned_lits, 1.0))
                                # Recompute learned_sat
                                l_sat = [0] * len(learned_clauses)
                                l_clauses = [lc for lc, lw in learned_clauses]
                                l_weights = [lw for lc, lw in learned_clauses]
                                n_learned = len(learned_clauses)
                                for li, (lcl, lw) in enumerate(learned_clauses):
                                    s = 0
                                    for lit in lcl:
                                        if assign[abs(lit) - 1] == (1 if lit > 0 else 0):
                                            s += 1
                                    l_sat[li] = s

        # End of restart loop
        unsat_count = sum(1 for s in c_sat if s == 0)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assign)

        # Decay alpha slightly
        alpha = max(0.15, alpha * 0.98)

        # Decay learned clause weights
        for i in range(len(learned_clauses)):
            learned_clauses[i] = (learned_clauses[i][0], learned_clauses[i][1] * 0.9)

    # Final polish: try to fix remaining unsat clauses with targeted flips
    if best_assignment is not None:
        final_assign = list(best_assignment)
        final_c_sat = [0] * m
        for ci, cl in enumerate(clauses):
            s = 0
            for lit in cl:
                if final_assign[abs(lit) - 1] == (1 if lit > 0 else 0):
                    s += 1
            final_c_sat[ci] = s

        # Try a few rounds of targeted fixes
        for _ in range(100):
            unsat_cis = [i for i in range(m) if final_c_sat[i] == 0]
            if not unsat_cis:
                break
            ci = random.choice(unsat_cis)
            cl = clauses[ci]
            # Try flipping each var in the clause
            best_v = -1
            best_gain = -999
            for lit in cl:
                v = abs(lit) - 1
                gain = 0
                for (cci, sgn) in var_occurs[v]:
                    if final_c_sat[cci] == 0:
                        # Would this flip satisfy cci?
                        desired = 1 if sgn == 1 else 0
                        if final_assign[v] != desired:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v >= 0 and best_gain > 0:
                old_val = final_assign[best_v]
                final_assign[best_v] = 1 - old_val
                for (cci, sgn) in var_occurs[best_v]:
                    old_contrib = 1 if (old_val == (1 if sgn == 1 else 0)) else 0
                    new_contrib = 1 if ((1 - old_val) == (1 if sgn == 1 else 0)) else 0
                    final_c_sat[cci] += new_contrib - old_contrib

        final_unsat = sum(1 for s in final_c_sat if s == 0)
        if final_unsat == 0:
            return final_assign
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = final_assign

    return best_assignment
