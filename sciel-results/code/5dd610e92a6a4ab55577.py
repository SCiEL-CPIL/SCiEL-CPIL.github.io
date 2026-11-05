import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Preprocessing
    # clause_info: list of lists of (var_idx, is_pos)
    clause_info = []
    # var_clauses: list of lists of clause indices for each variable
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            info.append((var_idx, is_pos))
            var_clauses[var_idx].append(i)
        clause_info.append(info)
        
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_assignment = None
    best_unsat = m + 1
    
    # WalkSAT parameters
    p = 0.5  # Probability of random flip vs greedy flip in unsat clause
    max_steps_per_restart = 10000
    max_restarts = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize unsat status
        unsat = [False] * m
        unsat_count = 0
        
        for i, info in enumerate(clause_info):
            sat = False
            for var_idx, is_pos in info:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat[i] = True
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            # Find a variable to flip
            # Strategy: 
            # 1. Try to find a flip that reduces unsat_count (greedy)
            # 2. If none, pick a random unsatisfied clause and flip one of its vars
            
            # To be efficient, we can just pick a random unsatisfied clause and apply the p-strategy
            # This is standard WalkSAT.
            
            # Pick a random unsatisfied clause
            # We can maintain a list of unsat clause indices for O(1) random pick, 
            # but rebuilding it or scanning is O(M). M is small (~200).
            # Let's scan for an unsat clause.
            unsat_clause_indices = [i for i in range(m) if unsat[i]]
            if not unsat_clause_indices:
                return assignment
                
            c_idx = rng.choice(unsat_clause_indices)
            c_info = clause_info[c_idx]
            
            # Evaluate each literal in the clause
            # We want to pick a variable to flip.
            # For each variable in the clause, calculate the delta in unsat_count if we flip it.
            
            best_var = -1
            best_delta = float('inf')
            greedy_var = -1
            greedy_delta = 0
            
            for var_idx, is_pos in c_info:
                # Simulate flip
                old_val = assignment[var_idx]
                new_val = not old_val
                
                delta = 0
                # Check all clauses containing this variable
                for ci in var_clauses[var_idx]:
                    # Check if clause ci was satisfied before
                    # We need to know if it's satisfied by other literals or just this one?
                    # Easier: re-evaluate satisfaction of clause ci before and after?
                    # Since we only flip one var, we can check:
                    # Was ci satisfied before?
                    # If yes, and this var was the ONLY one satisfying it, flipping it makes it unsat.
                    # If no, and this var was the only one unsatisfied? No, clause is OR.
                    
                    # Let's just count satisfied literals in ci for the new assignment state for this var.
                    # Actually, incremental logic:
                    # A clause is unsatisfied if ALL its literals are false.
                    
                    # Let's define a helper to check if clause is satisfied given assignment, 
                    # but we only care about clauses in var_clauses[var_idx].
                    
                    # Optimization: 
                    # We can precompute "support" count? No, dynamic.
                    
                    # Let's just check the status of clause ci before and after flip.
                    # Before: 
                    # If unsat[ci] is True, it was unsatisfied.
                    # If we flip var_idx, does it become satisfied?
                    # It becomes satisfied if the literal (var_idx, is_pos) becomes True.
                    # Since we are flipping var_idx to new_val, the literal is satisfied if new_val == is_pos.
                    # If new_val == is_pos, the clause is satisfied (regardless of others).
                    # If new_val != is_pos, the clause remains unsatisfied ONLY IF all other literals are also false.
                    
                    # This logic is getting complex. Given M is small, let's just re-evaluate the specific clauses affected.
                    
                    # Clause ci status before:
                    # sat_before = not unsat[ci]
                    
                    # Clause ci status after:
                    # We need to check if any literal in ci is true in the new assignment.
                    # The only change is var_idx.
                    # If new_val == is_pos for this literal in ci, then ci is satisfied.
                    # Else, we need to check if any OTHER literal in ci is satisfied.
                    
                    # To avoid O(M) check for each var in clause (which is O(3*M) per step), 
                    # let's just do a full check for the few clauses affected.
                    
                    pass # Placeholder for logic below
            
            # Simplified WalkSAT Step:
            # 1. Pick random unsat clause c.
            # 2. For each variable v in c:
            #    - Flip v.
            #    - Update unsat_count by checking all clauses containing v.
            #    - Record delta.
            #    - Unflip v.
            # 3. If any delta < 0, pick one (or best).
            # 4. Else, pick with prob p a random var, else pick var with min delta.
            
            deltas = []
            for var_idx, is_pos in c_info:
                # Flip
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                new_unsat_count = 0
                # We need to update unsat array for clauses in var_clauses[var_idx]
                # To avoid modifying global state during evaluation, we can compute delta directly.
                
                delta = 0
                for ci in var_clauses[var_idx]:
                    # Check if clause ci is satisfied in the CURRENT (modified) assignment
                    # Note: assignment[var_idx] is already flipped.
                    sat = False
                    for vi, ip in clause_info[ci]:
                        if assignment[vi] == ip:
                            sat = True
                            break
                    
                    if not sat:
                        # Clause is unsatisfied now
                        # Was it unsatisfied before?
                        # We can check if it was unsatisfied before the flip.
                        # If it was unsat before, and still unsat, delta 0.
                        # If it was sat before, and now unsat, delta +1.
                        # If it was unsat before, and now sat, delta -1.
                        # If it was sat before, and now sat, delta 0.
                        
                        # We need to know if it was sat before.
                        # If ci == c_idx, we know it was unsat (since we picked it from unsat list).
                        # For other clauses, we can check their status in the original assignment?
                        # No, we modified assignment. 
                        
                        # Let's revert and do a cleaner check.
                        pass
                
                # This "simulate and revert" for each var in clause is O(3 * deg(v) * 3).
                # deg(v) is roughly M*3/N ~ 200*3/50 = 12.
                # So 3 * 12 * 3 = 108 operations per var. 3 vars per clause. 324 ops per step.
                # 10000 steps * 324 = 3.24M ops per restart. Very fast.
                
                # Let's implement the delta calculation properly.
                
                # Revert assignment[var_idx] for now to calculate delta cleanly?
                # No, let's just calculate delta based on the flip.
                
                # For each clause ci in var_clauses[var_idx]:
                # 1. Determine if ci is satisfied in assignment BEFORE flip.
                #    We can store the original unsat array.
                # 2. Determine if ci is satisfied in assignment AFTER flip.
                
                # Let's store original unsat state for this restart loop iteration.
                # Actually, `unsat` is updated at the end of the step.
                # So `unsat` currently reflects the state BEFORE the flip.
                
                for ci in var_clauses[var_idx]:
                    sat_before = not unsat[ci]
                    
                    # Check sat_after
                    sat_after = False
                    for vi, ip in clause_info[ci]:
                        if assignment[vi] == ip: # assignment is currently flipped
                            sat_after = True
                            break
                    
                    if sat_before and not sat_after:
                        delta += 1
                    elif not sat_before and sat_after:
                        delta -= 1
                
                # Unflip
                assignment[var_idx] = old_val
                deltas.append(delta)
                
            # Now choose var
            min_delta = min(deltas)
            if min_delta < 0:
                # Greedy: pick one with min delta
                # If multiple, pick random among them?
                candidates = [i for i, d in enumerate(deltas) if d == min_delta]
                idx = rng.choice(candidates)
            else:
                # Walk: pick random with prob p, else min delta
                if rng.random() < p:
                    idx = rng.randint(0, len(c_info) - 1)
                else:
                    candidates = [i for i, d in enumerate(deltas) if d == min_delta]
                    idx = rng.choice(candidates)
            
            # Apply the flip
            var_idx_to_flip = c_info[idx][0]
            assignment[var_idx_to_flip] = not assignment[var_idx_to_flip]
            
            # Update unsat array and count
            for ci in var_clauses[var_idx_to_flip]:
                # Check new status
                sat = False
                for vi, ip in clause_info[ci]:
                    if assignment[vi] == ip:
                        sat = True
                        break
                
                if sat and unsat[ci]:
                    unsat[ci] = False
                    unsat_count -= 1
                elif not sat and not unsat[ci]:
                    unsat[ci] = True
                    unsat_count += 1
            
            steps += 1
            
    return best_assignment if best_assignment else [False] * n
