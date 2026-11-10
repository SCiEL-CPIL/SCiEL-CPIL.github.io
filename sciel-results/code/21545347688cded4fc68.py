"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit

    # Preprocess: convert to 0-based variable indices and signs
    clause_vars = []
    clause_sigs = []
    for c in clauses:
        cv = [abs(l) - 1 for l in c]
        cs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(cv)
        clause_sigs.append(cs)

    # Build var -> list of (clause_idx, pos_in_clause) for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append((i, j))

    # Precompute variable degrees for static heuristic (backup)
    var_degree = [0] * n
    for i in range(m):
        for j in range(3):
            var_degree[clause_vars[i][j]] += 1

    # Precompute clause weights (sum of degrees of its variables) for static heuristic
    clause_weights = [0] * m
    for i in range(m):
        w = 0
        for j in range(3):
            w += var_degree[clause_vars[i][j]]
        clause_weights[i] = w

    rng = random.Random(42)

    def make_random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def walk_sat_run(assignment, max_flips, p_init=0.5):
        """WalkSAT with dynamic clause weighting and min-confl literal heuristic."""
        
        # Initialize clause satisfaction status
        clause_sat_count = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        # Initialize unsat_list and unsat_pos for O(1) selection and updates
        unsat_list = []
        unsat_pos = [-1] * m  # -1 if not in unsat_list, else index in unsat_list
        
        # Initialize var_unsat_degree
        var_unsat_degree = [0] * n

        for i in range(m):
            if clause_sat_count[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)
                # Update var_unsat_degree for vars in this unsat clause
                for j in range(3):
                    var_unsat_degree[clause_vars[i][j]] += 1

        p = p_init
        no_improve_count = 0
        prev_unsat_count = len(unsat_list)

        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None, assignment

            if len(unsat_list) == 0:
                return assignment[:], assignment

            # Adaptive p: if no improvement, increase randomness
            if len(unsat_list) < prev_unsat_count:
                no_improve_count = 0
                p = max(0.2, p * 0.95) # Decrease randomness (more greedy)
            else:
                no_improve_count += 1
                if no_improve_count > 10:
                    p = min(0.9, p * 1.1) # Increase randomness
            prev_unsat_count = len(unsat_list)

            # Pick an unsatisfied clause
            # Strategy: With high probability, pick the clause with highest dynamic weight (sum of var_unsat_degree).
            # With low probability, pick randomly to escape local minima.
            if rng.random() < 0.8:
                best_ci = -1
                best_w = -1
                for idx in unsat_list:
                    w = 0
                    for j in range(3):
                        w += var_unsat_degree[clause_vars[idx][j]]
                    if w > best_w:
                        best_w = w
                        best_ci = idx
                ci = best_ci
            else:
                idx = rng.randint(0, len(unsat_list) - 1)
                ci = unsat_list[idx]

            # Choose literal to flip using min-confl heuristic
            if rng.random() < p:
                li = rng.randint(0, 2)
            else:
                best_li = 0
                best_gain = -10
                for li in range(3):
                    var = clause_vars[ci][li]
                    gain = 0
                    old_val = assignment[var]
                    new_val = not old_val
                    
                    for (oi, oj) in var_to_clauses[var]:
                        if oi == ci:
                            continue
                        
                        osig = clause_sigs[oi][oj]
                        old_sat = (old_val == (osig > 0))
                        new_sat = (new_val == (osig > 0))
                        
                        if unsat_pos[oi] != -1:
                            # Clause oi is currently unsat
                            if not old_sat and new_sat:
                                gain += 1
                        else:
                            # Clause oi is currently sat
                            if old_sat and not new_sat:
                                # We are breaking a satisfied literal
                                # Check if oi becomes unsat
                                if clause_sat_count[oi] == 1:
                                    gain -= 1

                    if gain > best_gain:
                        best_gain = gain
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count, unsat_list, and var_unsat_degree
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))
                
                if was_sat and not now_sat:
                    # Clause became less satisfied
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        # Clause became unsat
                        unsat_pos[oi] = len(unsat_list)
                        unsat_list.append(oi)
                        # Update var_unsat_degree for all vars in this clause
                        for k in range(3):
                            var_unsat_degree[clause_vars[oi][k]] += 1
                elif not was_sat and now_sat:
                    # Clause became more satisfied
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] > 0:
                        # Clause became sat (or stayed sat, but was unsat before? No, was_sat was False)
                        # So it was unsat (count 0) and is now sat (count 1)
                        if unsat_pos[oi] != -1:
                            pos = unsat_pos[oi]
                            last_idx = len(unsat_list) - 1
                            last_clause = unsat_list[last_idx]
                            
                            # Swap with last
                            unsat_list[pos] = last_clause
                            unsat_pos[last_clause] = pos
                            
                            # Pop last
                            unsat_list.pop()
                            unsat_pos[oi] = -1
                            
                            # Update var_unsat_degree for all vars in this clause
                            for k in range(3):
                                var_unsat_degree[clause_vars[oi][k]] -= 1

        if len(unsat_list) == 0:
            return assignment[:], assignment
        return None, assignment

    def count_unsat(assignment):
        unsat_count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count

    def polish(assignment, unsat_clauses):
        """Try to fix a small number of unsat clauses by flipping variables."""
        if not unsat_clauses:
            return assignment
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        vars_to_try = set()
        for ci in unsat_clauses:
            for j in range(3):
                vars_to_try.add(clause_vars[ci][j])
        
        best_assign = assignment[:]
        best_count = current_unsat
        
        for var in vars_to_try:
            assignment[var] = not assignment[var]
            new_count = count_unsat(assignment)
            if new_count < best_count:
                best_count = new_count
                best_assign = assignment[:]
            assignment[var] = not assignment[var]
            
        return best_assign

    best_assignment = None
    best_unsat = m + 1

    max_restarts = 1000
    multipliers = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    base_flips = 100

    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        mult = multipliers[restart % len(multipliers)]
        current_base = base_flips + (restart // 10) * 100
        flips = current_base * mult
        if flips > 5000:
            flips = 5000

        assignment = make_random_assignment()
        
        result, final_assignment = walk_sat_run(assignment, max_flips=flips, p_init=0.5)
        
        if result is not None:
            return result

        unsat_count = count_unsat(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]
            
            if unsat_count <= 3:
                unsat_clauses = []
                for i in range(m):
                    cv = clause_vars[i]
                    cs = clause_sigs[i]
                    sat = False
                    for j in range(3):
                        if final_assignment[cv[j]] == (cs[j] > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(i)
                
                polished = polish(final_assignment, unsat_clauses)
                polished_unsat = count_unsat(polished)
                if polished_unsat == 0:
                    return polished
                if polished_unsat < best_unsat:
                    best_unsat = polished_unsat
                    best_assignment = polished[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
