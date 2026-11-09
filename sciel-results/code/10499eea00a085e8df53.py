import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Preprocessing
    # clauses are lists of signed 1-based literals
    # Convert to (var_idx, polarity)
    # var_idx 0-based, polarity True/False
    
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
    
    # Luby restart schedule generator
    def luby_sequence():
        k = 1
        while True:
            for _ in range(k):
                yield k
            k *= 2
            
    luby_gen = luby_sequence()
    
    # Parameters
    # We use a mix of Luby steps and a cap to prevent extremely long single runs if stuck
    # But for planted SAT, long runs with phase saving are good.
    max_steps_per_restart = 50000 # Cap for safety, though Luby will rarely hit this in 285s for n=150
    
    # Phase saving parameters
    phase_save_flip_ratio = 0.1 # Flip 10% of variables from best state
    
    while time.time() < deadline:
        # Get next restart length from Luby sequence
        # Scale Luby sequence: Luby values are 1,1,2,1,1,2,4... 
        # For n=150, a step is ~10-50us. 
        # Luby 1 -> 1 step is too short. 
        # Let's scale Luby by a factor. 
        # Empirically, for 150 vars, restarts of 100-10000 steps are common.
        # Let's use raw Luby values but multiply by a base step count, e.g., 50.
        base_steps = 50
        steps = next(luby_gen) * base_steps
        
        # Cap steps to avoid wasting time if one restart takes too long
        if steps > max_steps_per_restart:
            steps = max_steps_per_restart
            
        # Ensure we don't exceed deadline in this restart
        if time.time() + (steps * 0.0001) > deadline: # Rough estimate 100us/step
             steps = max(1, int((deadline - time.time()) * 10000))

        # Initialize assignment
        if best_assignment is None:
            rng = random.Random()
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Phase Saving: Start from best assignment and flip some variables
            assignment = best_assignment[:]
            # Flip a random subset
            num_flips = max(1, int(n * phase_save_flip_ratio))
            indices_to_flip = random.sample(range(n), num_flips)
            for idx in indices_to_flip:
                assignment[idx] = not assignment[idx]
        
        # Calc initial state
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_count = 0
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Walk
        # Use a dedicated RNG for reproducibility within a restart if needed, 
        # but random.random() is fine for performance.
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Pick a random unsat clause
            # Efficient selection: 
            # If unsat_count is small, we can iterate. 
            # If large, we need a list. 
            # Maintaining a list of unsat clauses is O(1) update but O(M) init.
            # For simplicity and speed in Python, if unsat_count is small (< 100), 
            # we can just scan. But scanning M=600 is fast.
            # Let's maintain a list of unsat clause indices for O(1) random pick.
            # We'll rebuild it if it gets too stale or just use a simple scan if unsat is small.
            # Actually, let's just scan for an unsat clause. M is small (600).
            # 600 iterations is negligible compared to the gain calculation.
            
            c_idx = -1
            # Random start index to avoid bias
            start_scan = random.randrange(m)
            for i in range(m):
                idx = (start_scan + i) % m
                if is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx == -1:
                return assignment # Should not happen if unsat_count > 0

            # Evaluate flips for literals in c_idx
            # Heuristic: Min-Conflicts
            best_var = -1
            best_score = float('inf')
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                for c2_idx, p2 in var_clauses[v]:
                    # Skip the clause we are trying to fix? 
                    # No, flipping v in c_idx satisfies c_idx (since c_idx is unsat, 
                    # and v is in c_idx, flipping v makes that literal true).
                    # So c_idx is "fixed" (unsat -> sat).
                    # In the loop, sat_count[c_idx] is 0. 
                    # was_sat (old_val == p2) is False (since c_idx is unsat).
                    # So it goes to else: fixed += 1 if sat_count was 0.
                    # So c_idx contributes +1 to fixed. Correct.
                    
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        # Becomes unsat if this was the only satisfied literal
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        # Becomes sat
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v
                    
            # Noise
            # Adaptive noise: higher when far from solution, lower when close
            if unsat_count > 20:
                noise_prob = 0.5
            elif unsat_count > 5:
                noise_prob = 0.3
            else:
                noise_prob = 0.1
                
            if random.random() < noise_prob:
                v_to_flip = clause_data[c_idx][random.randrange(3)][0]
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

    # Final polish: Greedy Min-Conflicts on best_assignment
    if best_assignment is not None:
        polish_assignment = best_assignment[:]
        polish_deadline = deadline - 5
        
        p_sat_count = [0] * m
        p_is_unsat = [False] * m
        p_unsat_count = 0
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if polish_assignment[v] == p:
                    cnt += 1
            p_sat_count[c_idx] = cnt
            if cnt == 0:
                p_is_unsat[c_idx] = True
                p_unsat_count += 1
        
        # Tighter local search: pure greedy, no noise
        # Limit steps to avoid infinite loop if stuck in local min with 0 gain
        max_polish_steps = 2000
        for step in range(max_polish_steps):
            if time.time() > polish_deadline:
                break
            if p_unsat_count == 0:
                return polish_assignment
            
            # Find an unsat clause
            c_idx = -1
            for idx in range(m):
                if p_is_unsat[idx]:
                    c_idx = idx
                    break
            if c_idx == -1:
                return polish_assignment
            
            # Find best flip
            best_v = -1
            best_s = float('inf')
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                for c2_idx, p2 in var_clauses[v]:
                    was_sat = (polish_assignment[v] == p2)
                    if was_sat:
                        if p_sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        if p_sat_count[c2_idx] == 0:
                            fixed += 1
                s = broken - fixed
                if s < best_s:
                    best_s = s
                    best_v = v
            
            if best_v == -1:
                break
            
            # If best score is still positive (increases unsat), we might be stuck.
            # In a polish phase, if we can't reduce unsat, we stop.
            if best_s >= 0:
                break
            
            # Apply flip
            old_val = polish_assignment[best_v]
            new_val = not old_val
            polish_assignment[best_v] = new_val
            
            for c2_idx, p2 in var_clauses[best_v]:
                was_sat = (old_val == p2)
                if was_sat:
                    p_sat_count[c2_idx] -= 1
                    if p_sat_count[c2_idx] == 0:
                        if not p_is_unsat[c2_idx]:
                            p_is_unsat[c2_idx] = True
                            p_unsat_count += 1
                else:
                    p_sat_count[c2_idx] += 1
                    if p_sat_count[c2_idx] == 1:
                        if p_is_unsat[c2_idx]:
                            p_is_unsat[c2_idx] = False
                            p_unsat_count -= 1
                            
        if p_unsat_count < best_unsat_count:
            return polish_assignment
        return polish_assignment
    
    return best_assignment if best_assignment is not None else [False] * n
