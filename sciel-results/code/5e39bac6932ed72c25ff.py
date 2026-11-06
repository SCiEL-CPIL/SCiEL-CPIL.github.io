import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 315.0  # 15s margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [False] * n

    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_list = [abs(lit) - 1 for lit in clause]
        signs_list = [lit > 0 for lit in clause]
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    # Build var_to_clauses: for each variable, list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            var_to_clauses[v].append((i, clause_signs[i][j]))

    def init_state(assignment):
        clause_sat_count = np.zeros(m, dtype=np.int32)
        unsat_set = set()
        for i in range(m):
            count = 0
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == clause_signs[i][j]:
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_set.add(i)
        return clause_sat_count, unsat_set

    def flip_var(v, assignment, clause_sat_count, unsat_set):
        """Flip variable v and update state."""
        old_val = assignment[v]
        assignment[v] = not old_val
        
        for ci, sign in var_to_clauses[v]:
            old_sat = clause_sat_count[ci]
            if old_val == sign:
                # Was satisfied, now not
                clause_sat_count[ci] = old_sat - 1
                if old_sat - 1 == 0:
                    unsat_set.add(ci)
            else:
                # Was not satisfied, now is
                clause_sat_count[ci] = old_sat + 1
                if old_sat == 0:
                    unsat_set.discard(ci)

    def unit_propagate(assignment, clause_sat_count, unsat_set):
        """
        If a clause has exactly 1 unsatisfied literal (sat_count == 2 for 3-clause),
        the variable of that literal MUST be flipped to satisfy the clause.
        """
        changed = True
        while changed:
            if time.time() - start_time > timeout - 2.0:
                return
            changed = False
            # Scan for unit clauses
            for i in range(m):
                if clause_sat_count[i] == 2:
                    # Find the unsatisfied literal
                    for j, v in enumerate(clause_vars[i]):
                        if not (assignment[v] == clause_signs[i][j]):
                            # This literal is false, so we must flip v to make it true
                            flip_var(v, assignment, clause_sat_count, unsat_set)
                            changed = True
                            break # Move to next clause check
                    break # Restart scan as state changed

    rng = random.Random(42)
    
    # WalkSAT parameters
    max_flips_per_restart = 2000
    base_noise = 0.45
    
    while time.time() - start_time < timeout:
        # Random Restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count, unsat_set = init_state(assignment)

        if not unsat_set:
            return assignment

        # Unit Propagation
        unit_propagate(assignment, clause_sat_count, unsat_set)
        
        if not unsat_set:
            return assignment

        # WalkSAT
        for flip_num in range(max_flips_per_restart):
            if not unsat_set:
                return assignment
            
            if time.time() - start_time > timeout - 5.0:
                break

            # Pick random unsatisfied clause
            # Converting set to list is O(k). k is small.
            unsat_list = list(unsat_set)
            ci = rng.choice(unsat_list)
            
            # Calculate scores for variables in this clause
            vars_in_clause = clause_vars[ci]
            scores = []
            for v in vars_in_clause:
                score = 0
                for cj, sign in var_to_clauses[v]:
                    if cj != ci and clause_sat_count[cj] == 0:
                        score += 1
                scores.append(score)
            
            # Adaptive noise: decay over the restart
            noise = base_noise * (1.0 - 0.5 * (flip_num / max_flips_per_restart))
            
            if rng.random() < noise:
                # Random pick
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
            else:
                # Min score pick with random tie-breaking
                min_score = min(scores)
                best_vars = [v for v, s in zip(vars_in_clause, scores) if s == min_score]
                v = rng.choice(best_vars)
            
            flip_var(v, assignment, clause_sat_count, unsat_set)
            
            # Periodic unit propagation
            if flip_num % 5 == 0:
                unit_propagate(assignment, clause_sat_count, unsat_set)
                if not unsat_set:
                    return assignment

    return assignment
