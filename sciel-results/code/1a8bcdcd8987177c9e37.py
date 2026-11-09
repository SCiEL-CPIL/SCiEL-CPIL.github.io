import time
import random
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    # Safety margin: evaluator kills at 330s, keep 280s for safety
    deadline = start_time + 280
    
    # Preprocessing
    # Convert to 0-based indices and store polarity
    # clause_data: list of lists of (var_idx, polarity)
    # var_clauses: list of lists of (clause_idx, polarity)
    clause_data = []
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
        
    m = len(clauses)
    
    # Helper to calculate initial satisfaction
    def calc_sat_counts(assignment):
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_list = []
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_list.append(c_idx)
        return sat_count, is_unsat, unsat_list

    best_unsat_count = float('inf')
    best_assignment = None
    
    # Parameters
    max_restarts = 5000
    base_steps = 100
    
    # Use a single RNG for the entire process to avoid bias, 
    # but reset state on restart? No, just use a new seed for each restart 
    # to ensure independent walks.
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # New RNG for each restart to ensure independence
        rng = random.Random(restart * 1000003 + 42)
        
        # Initialization
        if best_assignment is not None and restart > 0 and rng.random() < 0.2:
            # Occasionally start from best found with perturbations
            assignment = best_assignment[:]
            num_perturb = max(1, n // 10)
            for _ in range(num_perturb):
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
        sat_count, is_unsat, unsat_list = calc_sat_counts(assignment)
        unsat_count = len(unsat_list)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Adaptive steps: increase steps if we are making progress or close
        # Simple heuristic: if unsat_count is low, allow more steps
        if unsat_count <= 5:
            steps_per_restart = 500
        elif unsat_count <= 20:
            steps_per_restart = 200
        else:
            steps_per_restart = base_steps
            
        # Local Search Loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Clean up unsat_list if it has grown large with stale entries
            if len(unsat_list) > unsat_count * 2 + 10:
                unsat_list = [i for i in unsat_list if is_unsat[i]]
                
            if not unsat_list:
                return assignment

            # Pick a random unsat clause
            c_idx = unsat_list[rng.randrange(len(unsat_list))]
            
            # Find the best variable to flip in this clause
            # We use a weighted min-conflicts strategy.
            # Score = fixed - 0.5 * broken
            # We prefer flips that fix many clauses and break few.
            
            best_var = -1
            best_score = float('inf')
            best_fixed = 0
            best_broken = 0
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                for c2_idx, p2 in var_clauses[v]:
                    # Check if this variable currently satisfies clause c2
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        # Flipping v will make this literal false in c2
                        # If c2 was satisfied only by this literal, it becomes unsat
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        # Flipping v will make this literal true in c2
                        # If c2 was unsat, it becomes sat
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                # Weighted score: fixing is good, breaking is bad.
                # Tuning factor 0.5 means breaking 1 clause is "costly" equivalent to not fixing 2?
                # Let's try score = fixed - 0.5 * broken
                score = fixed - 0.5 * broken
                if score < best_score:
                    best_score = score
                    best_var = v
                    best_fixed = fixed
                    best_broken = broken
                    
            # Adaptive Noise
            # If we are stuck (low improvement), increase noise.
            # Simple heuristic: noise decreases as unsat_count decreases?
            # Or noise increases if we haven't improved in a while?
            # Let's use a fixed noise that adapts to unsat_count.
            if unsat_count <= 3:
                noise_prob = 0.1
            elif unsat_count <= 10:
                noise_prob = 0.2
            else:
                noise_prob = 0.35
                
            if rng.random() < noise_prob:
                # Random variable from the clause
                v_to_flip = clause_data[c_idx][rng.randrange(3)][0]
            else:
                v_to_flip = best_var
                
            if v_to_flip == -1:
                continue

            # Apply Flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update State
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    # Lost satisfaction
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                else:
                    # Gained satisfaction
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
    # Final Polish: If we have a small number of unsat clauses, try a targeted exhaustive search
    if best_assignment is not None and 0 < best_unsat_count <= 15:
        assignment = best_assignment[:]
        sat_count, is_unsat, unsat_list = calc_sat_counts(assignment)
        unsat_count = len(unsat_list)
        
        # Identify variables involved in unsat clauses
        involved_vars = set()
        for c_idx in unsat_list:
            for v, p in clause_data[c_idx]:
                involved_vars.add(v)
        
        # Try single flips on involved variables
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return assignment
                
            for v in involved_vars:
                old_val = assignment[v]
                new_val = not old_val
                assignment[v] = new_val
                
                # Calculate new unsat count quickly? 
                # Or just update and check?
                # For polish, n is small (150), m is ~500. 
                # Recalculating all is O(m*k) ~ 1500 ops. Doing this for each v in involved_vars (maybe 20-30 vars) is ~45000 ops.
                # This is fast enough for a few iterations.
                
                temp_sat_count, temp_is_unsat, temp_unsat_list = calc_sat_counts(assignment)
                temp_unsat_count = len(temp_unsat_list)
                
                if temp_unsat_count < unsat_count:
                    unsat_count = temp_unsat_count
                    sat_count = temp_sat_count
                    is_unsat = temp_is_unsat
                    unsat_list = temp_unsat_list
                    improved = True
                    if unsat_count == 0:
                        return assignment
                else:
                    # Revert
                    assignment[v] = old_val
                    
            # If no single flip improved, try double flips?
            # This is O(V^2). If V=20, 400 pairs. 400 * 1500 = 600k ops. Acceptable.
            if not improved and len(involved_vars) <= 20:
                vars_list = list(involved_vars)
                for i in range(len(vars_list)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, len(vars_list)):
                        v1 = vars_list[i]
                        v2 = vars_list[j]
                        old1 = assignment[v1]
                        old2 = assignment[v2]
                        
                        assignment[v1] = not old1
                        assignment[v2] = not old2
                        
                        temp_sat_count, temp_is_unsat, temp_unsat_list = calc_sat_counts(assignment)
                        temp_unsat_count = len(temp_unsat_list)
                        
                        if temp_unsat_count < unsat_count:
                            unsat_count = temp_unsat_count
                            sat_count = temp_sat_count
                            is_unsat = temp_is_unsat
                            unsat_list = temp_unsat_list
                            improved = True
                            if unsat_count == 0:
                                return assignment
                        else:
                            assignment[v1] = old1
                            assignment[v2] = old2
                            
            if not improved:
                break

    if best_assignment is not None:
        return best_assignment
    return assignment
