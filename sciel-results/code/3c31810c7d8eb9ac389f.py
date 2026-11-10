"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295.0  # safety margin before 330s kill

    # Precompute clause structure
    m = len(clauses)
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = [abs(l) - 1 for l in c]
        signs = [l > 0 for l in c]
        clause_vars.append(vars_)
        clause_signs.append(signs)

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i][j]
            var_to_clauses[v].append(i)

    def check_assignment(assignment):
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == cs[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsatisfied(assignment):
        count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == cs[j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Generate initializations
    initializations = []
    initializations.append([True] * n)
    initializations.append([False] * n)
    rng = random.Random(0)
    initializations.append([rng.random() < 0.5 for _ in range(n)])
    pos_count = [0] * n
    neg_count = [0] * n
    for c in clauses:
        for l in c:
            if l > 0:
                pos_count[l - 1] += 1
            else:
                neg_count[-l - 1] += 1
    freq_init = []
    for i in range(n):
        freq_init.append(pos_count[i] >= neg_count[i])
    initializations.append(freq_init)

    best_assignment = None
    best_unsat = m + 1

    def simulated_annealing(assignment, max_steps, seed_offset=0):
        nonlocal best_assignment, best_unsat
        rng_sa = random.Random(seed_offset)
        curr = assignment[:]

        # Track per-clause satisfaction status and unsat count
        clause_sat = [False] * m
        curr_unsat = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for j in range(3):
                if curr[cv[j]] == cs[j]:
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                curr_unsat += 1

        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = curr[:]
        if curr_unsat == 0:
            return curr

        # Simulated annealing parameters
        T_start = max(1.0, float(curr_unsat) * 0.5)
        T_end = 0.01
        cool_rate = (T_end / T_start) ** (1.0 / max(1, max_steps))

        T = T_start
        noise_prob = 0.3

        # Track unsatisfied clause indices
        unsat_clauses = [i for i in range(m) if not clause_sat[i]]

        for step in range(max_steps):
            if time.time() > deadline:
                return curr

            if not unsat_clauses:
                return curr

            # Pick an unsatisfied clause
            chosen_clause = rng_sa.choice(unsat_clauses)
            cv = clause_vars[chosen_clause]
            cs = clause_signs[chosen_clause]

            # Select variable to flip
            if rng_sa.random() < noise_prob:
                v = rng_sa.choice(cv)
            else:
                # Weighted selection: prefer variables in more unsat clauses
                var_scores = {}
                for v in cv:
                    score = 0
                    for ci in var_to_clauses[v]:
                        if not clause_sat[ci]:
                            score += 1
                    var_scores[v] = score
                max_score = max(var_scores.values())
                candidates = [v for v in cv if var_scores[v] == max_score]
                v = rng_sa.choice(candidates)

            # Determine new value
            old_val = curr[v]
            # Find sign of v in chosen clause
            v_sign = None
            for j in range(3):
                if cv[j] == v:
                    v_sign = cs[j]
                    break

            if rng_sa.random() < 0.6:
                new_val = v_sign
            else:
                new_val = not old_val

            if new_val == old_val:
                T *= cool_rate
                continue

            # Compute delta in unsatisfied count
            delta = 0
            affected = var_to_clauses[v]
            for ci in affected:
                was_sat = clause_sat[ci]
                cvars = clause_vars[ci]
                csigns = clause_signs[ci]
                # Check if any other literal satisfies this clause
                other_sat = False
                for j in range(3):
                    if cvars[j] != v and curr[cvars[j]] == csigns[j]:
                        other_sat = True
                        break
                # Check if v's literal is satisfied with new_val
                v_satisfied = False
                for j in range(3):
                    if cvars[j] == v:
                        v_satisfied = (new_val == csigns[j])
                        break

                if other_sat:
                    new_sat = True
                else:
                    new_sat = v_satisfied

                if was_sat and not new_sat:
                    delta += 1
                elif not was_sat and new_sat:
                    delta -= 1

            # Accept/reject
            if delta <= 0:
                accept = True
            else:
                if T > 1e-10:
                    prob = math.exp(-delta / T)
                    accept = rng_sa.random() < prob
                else:
                    accept = False

            if accept:
                curr[v] = new_val
                for ci in affected:
                    cvars = clause_vars[ci]
                    csigns = clause_signs[ci]
                    new_sat = False
                    for j in range(3):
                        if curr[cvars[j]] == csigns[j]:
                            new_sat = True
                            break
                    if clause_sat[ci] and not new_sat:
                        curr_unsat += 1
                        unsat_clauses.append(ci)
                    elif not clause_sat[ci] and new_sat:
                        curr_unsat -= 1
                    clause_sat[ci] = new_sat

                # Periodically rebuild unsat_clauses
                if step % 200 == 0:
                    unsat_clauses = [i for i in range(m) if not clause_sat[i]]

                if curr_unsat < best_unsat:
                    best_unsat = curr_unsat
                    best_assignment = curr[:]
                if curr_unsat == 0:
                    return curr

            T *= cool_rate

        return curr

    # Try each initialization with SA
    step_budgets = [5000, 15000, 30000, 60000, 120000]
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline:
            break
        for step_idx, budget in enumerate(step_budgets):
            if time.time() > deadline:
                break
            result = simulated_annealing(init, budget, seed_offset=init_idx * 1000 + step_idx)
            if best_unsat == 0:
                return best_assignment

    # Final polish: if close, try focused local search
    if best_assignment is not None and 0 < best_unsat <= 5:
        if time.time() < deadline - 30:
            for _ in range(10000):
                if time.time() > deadline:
                    break
                improved = False
                for v in range(n):
                    old_val = best_assignment[v]
                    best_assignment[v] = not old_val
                    new_unsat = count_unsatisfied(best_assignment)
                    if new_unsat < best_unsat:
                        best_unsat = new_unsat
                        improved = True
                        if best_unsat == 0:
                            return best_assignment
                    else:
                        best_assignment[v] = old_val
                if not improved:
                    break

    if best_assignment is not None:
        return best_assignment
    rng_final = random.Random(12345)
    return [rng_final.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
