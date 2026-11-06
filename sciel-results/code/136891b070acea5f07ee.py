# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin before 300s limit

    # Preprocess clauses to 0-based signed literals
    clauses_pre = []
    for c in clauses:
        pre_c = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pre_c.append((var_idx, sign))
        clauses_pre.append(pre_c)

    m = len(clauses_pre)

    # Build variable-to-clause index for efficient updates
    var_clause = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_pre):
        for (var, sign) in c:
            var_clause[var].append((ci, sign))

    def count_unsat(clause_idx, assignment):
        c = clauses_pre[clause_idx]
        cnt = 0
        for (var, sign) in c:
            if assignment[var] != (sign == 1):
                cnt += 1
        return cnt

    def is_satisfied(clause_idx, assignment):
        c = clauses_pre[clause_idx]
        for (var, sign) in c:
            if assignment[var] == (sign == 1):
                return True
        return False

    def total_unsat(assignment):
        cnt = 0
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                cnt += 1
        return cnt

    def stochastic_ac3(assignment, max_rounds, rng):
        """Stochastic Local Consistency: randomly assign variables to enforce arc consistency.
        
        Returns (fixed_assignment, num_fixed) where fixed_assignment has None for unfixed vars.
        """
        # Start with the given assignment as hints, but treat all as unassigned
        domains = [None] * n  # None means unassigned, True/False means fixed
        fixed = 0
        
        # First pass: use the initial assignment as a soft bias
        # We'll randomly fix variables based on the initial assignment
        for _ in range(min(max_rounds, n)):
            # Pick a random unassigned variable
            unassigned = [i for i in range(n) if domains[i] is None]
            if not unassigned:
                break
            
            var = rng.choice(unassigned)
            # Use the initial assignment as a guide with some noise
            if rng.random() < 0.7:
                val = assignment[var]
            else:
                val = rng.random() < 0.5
            
            domains[var] = val
            fixed += 1
            
            # Propagate: check all clauses containing this variable
            for (ci, sign) in var_clause[var]:
                clause = clauses_pre[ci]
                # Check if this clause is now satisfied or if it forces other variables
                true_cnt = 0
                false_forced = []
                for (v2, s2) in clause:
                    if domains[v2] is not None:
                        if domains[v2] == (s2 == 1):
                            true_cnt += 1
                    else:
                        false_forced.append((v2, s2))
                
                if true_cnt > 0:
                    continue  # Clause already satisfied
                
                # Clause not satisfied; if only one unassigned var, it's forced
                if len(false_forced) == 1:
                    v2, s2 = false_forced[0]
                    domains[v2] = (s2 == 1)
                    fixed += 1
                    # Recursively propagate for this newly fixed variable
                    for (ci3, sign3) in var_clause[v2]:
                        clause3 = clauses_pre[ci3]
                        true_cnt3 = 0
                        false_forced3 = []
                        for (v3, s3) in clause3:
                            if domains[v3] is not None:
                                if domains[v3] == (s3 == 1):
                                    true_cnt3 += 1
                            else:
                                false_forced3.append((v3, s3))
                        if true_cnt3 > 0:
                            continue
                        if len(false_forced3) == 1:
                            v3, s3 = false_forced3[0]
                            if domains[v3] is None:
                                domains[v3] = (s3 == 1)
                                fixed += 1
        
        return domains, fixed

    def walksat(assignment, max_steps, rng, p_flip):
        """WalkSAT with random restarts. Returns (solved, final_assignment, steps_used)."""
        current = assignment[:]
        best_unsat = total_unsat(current)
        best_assignment = current[:]

        if best_unsat == 0:
            return True, current, 0

        for step in range(max_steps):
            unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, current)]
            if not unsat_clauses:
                return True, current, step + 1

            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = current[:]

            clause_idx = rng.choice(unsat_clauses)
            clause = clauses_pre[clause_idx]

            best_var = -1
            best_score = -10**9
            best_flip_val = None

            for (var, sign) in clause:
                cur_val = current[var]
                new_val = not cur_val

                score = 0
                for (ci2, sign2) in var_clause[var]:
                    c2 = clauses_pre[ci2]
                    true_cnt = 0
                    for (v2, s2) in c2:
                        if current[v2] == (s2 == 1):
                            true_cnt += 1

                    var_was_true = (cur_val == (sign2 == 1))
                    var_will_be_true = (new_val == (sign2 == 1))

                    if true_cnt == 0:
                        if var_will_be_true:
                            score += 1
                    elif true_cnt == 1 and var_was_true:
                        if not var_will_be_true:
                            score -= 1

                score += rng.random() * 0.1

                if score > best_score:
                    best_score = score
                    best_var = var
                    best_flip_val = new_val

            if best_var >= 0:
                current[best_var] = best_flip_val

            if all(is_satisfied(ci, current) for ci in range(m)):
                return True, current, step + 1

        if total_unsat(best_assignment) < total_unsat(current):
            return False, best_assignment, max_steps
        return False, current, max_steps

    def simulated_annealing(assignment, max_steps, rng):
        """Simulated annealing for 3-SAT. Returns (solved, final_assignment, steps_used)."""
        current = assignment[:]
        best_unsat = total_unsat(current)
        best_assignment = current[:]

        if best_unsat == 0:
            return True, current, 0

        T_start = 5.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))

        T = T_start
        for step in range(max_steps):
            T *= cooling

            unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, current)]
            if not unsat_clauses:
                return True, current, step + 1

            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = current[:]

            var = rng.randint(0, n - 1)
            new_val = not current[var]

            delta = 0
            for (ci2, sign2) in var_clause[var]:
                c2 = clauses_pre[ci2]
                true_cnt = 0
                for (v2, s2) in c2:
                    if current[v2] == (s2 == 1):
                        true_cnt += 1

                var_was_true = (current[var] == (sign2 == 1))
                var_will_be_true = (new_val == (sign2 == 1))

                if true_cnt == 0:
                    if var_will_be_true:
                        delta -= 1
                elif true_cnt == 1 and var_was_true:
                    if not var_will_be_true:
                        delta += 1

            if delta <= 0:
                current[var] = new_val
            else:
                if rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current[var] = new_val

            if all(is_satisfied(ci, current) for ci in range(m)):
                return True, current, step + 1

        if total_unsat(best_assignment) < total_unsat(current):
            return False, best_assignment, max_steps
        return False, current, max_steps

    def local_search_with_tabu(assignment, max_steps, rng, tabu_len=10):
        """Local search with tabu list. Returns (solved, final_assignment, steps_used)."""
        current = assignment[:]
        best_unsat = total_unsat(current)
        best_assignment = current[:]

        if best_unsat == 0:
            return True, current, 0

        tabu = {}
        for step in range(max_steps):
            unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, current)]
            if not unsat_clauses:
                return True, current, step + 1

            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = current[:]

            clause_idx = rng.choice(unsat_clauses)
            clause = clauses_pre[clause_idx]

            best_var = -1
            best_score = -10**9

            for (var, sign) in clause:
                if var in tabu and step < tabu[var]:
                    continue

                cur_val = current[var]
                new_val = not cur_val

                score = 0
                for (ci2, sign2) in var_clause[var]:
                    c2 = clauses_pre[ci2]
                    true_cnt = 0
                    for (v2, s2) in c2:
                        if current[v2] == (s2 == 1):
                            true_cnt += 1

                    var_was_true = (cur_val == (sign2 == 1))
                    var_will_be_true = (new_val == (sign2 == 1))

                    if true_cnt == 0:
                        if var_will_be_true:
                            score += 1
                    elif true_cnt == 1 and var_was_true:
                        if not var_will_be_true:
                            score -= 1

                score += rng.random() * 0.5

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var < 0:
                best_var = rng.randint(0, n - 1)

            current[best_var] = not current[best_var]
            tabu[best_var] = step + tabu_len

            if all(is_satisfied(ci, current) for ci in range(m)):
                return True, current, step + 1

        if total_unsat(best_assignment) < total_unsat(current):
            return False, best_assignment, max_steps
        return False, current, max_steps

    # Main search loop with multiple strategies and restarts
    rng = random.Random(0)
    best_overall = None
    best_unsat_overall = float('inf')

    # Initial random assignments
    initial_assignments = []
    for _ in range(5):
        init = [rng.random() < 0.5 for _ in range(n)]
        initial_assignments.append(init)

    initial_assignments.append([True] * n)
    initial_assignments.append([False] * n)

    strategies = [
        ("walksat", walksat),
        ("sa", simulated_annealing),
        ("tabu", local_search_with_tabu),
    ]

    steps_per_call = 5000
    p_flip_values = [0.5, 0.3, 0.7, 0.4, 0.6]

    attempt = 0
    while time.time() < deadline:
        strat_name, strat_func = strategies[attempt % len(strategies)]
        init = initial_assignments[attempt % len(initial_assignments)]

        # Apply stochastic AC-3 preprocessing to get a better starting point
        # This fixes some variables based on local consistency
        ac3_rounds = 30 + (attempt % 4) * 10  # Vary the number of rounds
        domains, num_fixed = stochastic_ac3(init, ac3_rounds, rng)
        
        # Create a new assignment: use fixed values, random for unfixed
        ac3_init = []
        for i in range(n):
            if domains[i] is not None:
                ac3_init.append(domains[i])
            else:
                ac3_init.append(init[i])  # Fallback to original initial value

        # Use the AC-3 refined assignment for the local search
        search_init = ac3_init

        if strat_name == "walksat":
            p = p_flip_values[attempt % len(p_flip_values)]
            solved, result, steps = strat_func(search_init, steps_per_call, rng, p)
        elif strat_name == "sa":
            solved, result, steps = strat_func(search_init, steps_per_call, rng)
        elif strat_name == "tabu":
            solved, result, steps = strat_func(search_init, steps_per_call, rng, tabu_len=10)
        else:
            solved, result, steps = strat_func(search_init, steps_per_call, rng)

        unsat_count = total_unsat(result)
        if unsat_count < best_unsat_overall:
            best_unsat_overall = unsat_count
            best_overall = result[:]

        if solved:
            return result

        attempt += 1

        if attempt % 10 == 0:
            fresh = [rng.random() < 0.5 for _ in range(n)]
            initial_assignments.append(fresh)

    # Final polish
    if best_overall is not None and best_unsat_overall > 0 and best_unsat_overall <= 10:
        while time.time() < deadline:
            solved, result, steps = walksat(best_overall, steps_per_call, rng, 0.5)
            if solved:
                return result
            unsat_count = total_unsat(result)
            if unsat_count < best_unsat_overall:
                best_unsat_overall = unsat_count
                best_overall = result[:]

    return best_overall if best_overall is not None else [False] * n
# EVOLVE-BLOCK-END
