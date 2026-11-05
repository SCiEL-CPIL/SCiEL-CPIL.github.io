import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 295
    
    # Pre-compute variable to clauses mapping
    # var_to_clauses[i] = list of clause indices
    var_to_clauses = [[] for _ in range(n)]
    # clause_var_index[c] = {var_idx: index_in_clause} to quickly find the literal sign?
    # Actually, we just need to know if the literal is positive or negative.
    # Let's store the sign for each variable in each clause? Or just re-derive from literal.
    # Storing literal values is fine.
    
    # For faster access, let's store for each clause, the variables and their signs.
    # But var_to_clauses is the main driver for updates.
    
    for i, clause in enumerate(clauses):
        for l in clause:
            var_to_clauses[abs(l)-1].append(i)
            
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT Parameters
    p_noise = 0.4
    max_restarts = 10000
    
    # Helper to check if clause is satisfied given assignment
    # We will use sat_counts instead for main loop efficiency
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize sat_counts and unsat set
        m = len(clauses)
        sat_counts = [0] * m
        unsat_indices = set()
        
        for i, c in enumerate(clauses):
            count = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    count += 1
            sat_counts[i] = count
            if count == 0:
                unsat_indices.add(i)
                
        if not unsat_indices:
            return assignment
            
        cur_unsat = len(unsat_indices)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Local search
        # Steps per restart: dynamic or fixed? 
        # With O(1) updates, we can afford more steps.
        # Let's try 20000 steps per restart, but break if time is up.
        steps_per_restart = 20000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            if not unsat_indices:
                return assignment
                
            # Pick a random unsat clause
            # To avoid list(unsat_indices) cost, we can keep a list of unsat clauses and pick from it?
            # But set is fine for 50 vars. Number of clauses is usually ~300-500.
            # list(unsat_indices) is O(U).
            c_idx = rng.choice(list(unsat_indices))
            clause = clauses[c_idx]
            
            # Evaluate flip for each literal in the clause
            # We want to minimize "broken" clauses.
            # Broken: clauses that are currently satisfied (sat_count > 0) and will become unsat (sat_count becomes 0).
            
            best_var = None
            best_score = float('inf')
            best_gain = -1 # Secondary criterion: maximize fixed clauses?
            
            for l in clause:
                var_idx = abs(l) - 1
                
                # Simulate flipping var_idx
                # 1. The current clause c_idx is unsat. Flipping var_idx will make literal l True.
                #    So c_idx becomes sat. (Gain +1 for c_idx)
                
                broken = 0
                fixed = 0
                
                # Check other clauses containing var_idx
                for other_c_idx in var_to_clauses[var_idx]:
                    if other_c_idx == c_idx:
                        continue
                    
                    # Current state of other_c_idx
                    current_count = sat_counts[other_c_idx]
                    
                    # Determine if the literal for var_idx in other_c_idx is currently True
                    # Find the literal in other_c_idx that corresponds to var_idx
                    # We can just iterate the clause to find the sign, but that's O(3).
                    # Better: precompute sign? Or just check.
                    # Since clause size is 3, iterating is fast.
                    lit_val = None
                    for ol in clauses[other_c_idx]:
                        if abs(ol) - 1 == var_idx:
                            lit_val = ol
                            break
                            
                    # lit_val is the literal (signed)
                    # Is it currently true?
                    is_lit_true = (lit_val > 0 and assignment[var_idx]) or (lit_val < 0 and not assignment[var_idx])
                    
                    # If we flip var_idx:
                    # If is_lit_true was True, it becomes False. sat_count decreases by 1.
                    # If is_lit_true was False, it becomes True. sat_count increases by 1.
                    
                    if is_lit_true:
                        new_count = current_count - 1
                    else:
                        new_count = current_count + 1
                    
                    if current_count == 0:
                        # Was unsat.
                        if new_count > 0:
                            fixed += 1
                    else:
                        # Was sat.
                        if new_count == 0:
                            broken += 1
                            
                # Score: Minimize broken, then maximize fixed
                if broken < best_score or (broken == best_score and fixed > best_gain):
                    best_score = broken
                    best_gain = fixed
                    best_var = var_idx
                    
            # Decide flip
            if rng.random() < p_noise:
                # Random flip from clause
                l = rng.choice(clause)
                var_idx = abs(l) - 1
            else:
                var_idx = best_var
                
            if var_idx is None:
                continue
                
            # Perform flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update sat_counts and unsat_indices
            for c_idx in var_to_clauses[var_idx]:
                # Find literal sign for var_idx in c_idx
                # We can do this efficiently if we stored it, but let's just find it.
                # Actually, we can just re-evaluate the clause? No, O(3) is fine.
                # But we need to know if the specific literal changed from T to F or F to T.
                
                # Let's find the literal
                lit_val = None
                for ol in clauses[c_idx]:
                    if abs(ol) - 1 == var_idx:
                        lit_val = ol
                        break
                        
                # Determine if this literal is NOW true (after flip)
                is_lit_now_true = (lit_val > 0 and assignment[var_idx]) or (lit_val < 0 and not assignment[var_idx])
                
                # Determine if it WAS true (before flip)
                # Since we just flipped, if it is NOW true, it was False before.
                # If it is NOW false, it was True before.
                
                if is_lit_now_true:
                    # Changed F -> T, so sat_count increases by 1
                    old_count = sat_counts[c_idx]
                    new_count = old_count + 1
                    sat_counts[c_idx] = new_count
                    
                    if old_count == 0 and new_count > 0:
                        unsat_indices.remove(c_idx)
                else:
                    # Changed T -> F, so sat_count decreases by 1
                    old_count = sat_counts[c_idx]
                    new_count = old_count - 1
                    sat_counts[c_idx] = new_count
                    
                    if old_count > 0 and new_count == 0:
                        unsat_indices.add(c_idx)
                        
        # End local search loop
        if time.time() > deadline - 5:
            break

    if best_assignment:
        return best_assignment
    return [False] * n
