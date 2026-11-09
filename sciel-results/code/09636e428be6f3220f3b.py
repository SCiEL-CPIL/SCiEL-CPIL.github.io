import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Preprocessing
    # clauses are lists of signed 1-based literals
    # Convert to (var_idx, polarity)
    # var_idx 0-based, polarity True/False
    
    # Data structures
    clause_data = [] # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)] # list of lists of (clause_idx, polarity)
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
        
    m = len(clauses)
    
    best_unsat_count = float('inf')
    best_assignment = None
    
    # Parameters tuned for 150 vars, ~450-600 clauses
    # Adaptive restart strategy with longer walks
    max_restarts = 3000 
    base_steps = 150
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 7 + 1)
        
        # Adaptive initialization: start from best found so far if available
        if best_assignment is not None and restart > 0:
            # With some probability, start from best assignment with small perturbations
            if rng.random() < 0.3:
                assignment = best_assignment[:]
                # Perturb a few random variables
                num_perturb = max(1, n // 20)
                for _ in range(num_perturb):
                    v = rng.randrange(n)
                    assignment[v] = not assignment[v]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calc initial sat counts
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_list = []
        unsat_count = 0
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_list.append(c_idx)
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Adaptive steps: longer walks when close to solution
        if unsat_count <= 5:
            steps_per_restart = base_steps * 3
        elif unsat_count <= 20:
            steps_per_restart = base_steps * 2
        else:
            steps_per_restart = base_steps
            
        # Walk
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Pick random unsat clause
            # Clean stale entries if list is much larger than unsat_count
            if len(unsat_list) > unsat_count * 3 and len(unsat_list) > 10:
                unsat_list = [i for i in unsat_list if is_unsat[i]]
            
            c_idx = None
            # Limit retries to avoid pathological cases
            for _ in range(10):
                idx = unsat_list[rng.randrange(len(unsat_list))]
                if is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx is None:
                # Fallback: scan for an unsat clause
                for idx in range(m):
                    if is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    return assignment

            # Evaluate flips for literals in c_idx
            # Weighted min-conflicts: weight fixed clauses more heavily
            # Score = broken * 1.0 - fixed * 1.5 (prefer fixing more clauses)
            
            best_var = -1
            best_score = float('inf')
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                # Iterate over clauses containing v
                for c2_idx, p2 in var_clauses[v]:
                    # Check if v is currently satisfied in c2
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        # Becomes unsat if this was the only satisfied literal
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        # Becomes sat
                        # If it was 0, it becomes 1 (fixed)
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                # Weighted score: fix clauses are more valuable
                score = broken * 1.0 - fixed * 1.5
                if score < best_score:
                    best_score = score
                    best_var = v
                    
            # Adaptive noise: less noise when close to solution
            if unsat_count <= 3:
                noise_prob = 0.1
            elif unsat_count <= 10:
                noise_prob = 0.2
            else:
                noise_prob = 0.35
                
            if rng.random() < noise_prob:
                v_to_flip = clause_data[c_idx][rng.randrange(3)][0]
            else:
                v_to_flip = best_var
                
            if v_to_flip == -1:
                continue

            # Apply flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    # Lost a satisfaction
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        # Became unsat
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                else:
                    # Gained a satisfaction
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        # Became sat
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

    # Final polish: try to fix remaining unsat clauses with targeted search
    if best_assignment is not None and best_unsat_count > 0 and best_unsat_count <= 20:
        assignment = best_assignment[:]
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_list = []
        unsat_count = 0
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_list.append(c_idx)
                unsat_count += 1
        
        # Targeted local search on remaining unsat clauses
        polish_steps = 500
        for step in range(polish_steps):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return assignment
                
            # Pick an unsat clause
            c_idx = None
            for _ in range(10):
                idx = unsat_list[rng.randrange(len(unsat_list))]
                if is_unsat[idx]:
                    c_idx = idx
                    break
            if c_idx is None:
                for idx in range(m):
                    if is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    break
                    
            # Try each variable in the clause, pick the one that breaks fewest
            best_var = -1
            best_broken = float('inf')
            for v, p in clause_data[c_idx]:
                broken = 0
                for c2_idx, p2 in var_clauses[v]:
                    if assignment[v] == p2 and sat_count[c2_idx] == 1:
                        broken += 1
                if broken < best_broken:
                    best_broken = broken
                    best_var = v
                    
            if best_var == -1:
                continue
                
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            for c2_idx, p2 in var_clauses[best_var]:
                was_sat = (old_val == p2)
                if was_sat:
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                else:
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
        if unsat_count == 0:
            return assignment

    # Fallback
    if best_assignment is not None:
        return best_assignment
    return assignment
