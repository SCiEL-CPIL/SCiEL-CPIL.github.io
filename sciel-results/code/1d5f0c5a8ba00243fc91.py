import time
import random
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin for safety

    # Preprocessing: Convert clauses to (var, sign)
    # var is 0-indexed, sign is 1 or -1
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        c_data = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            c_data.append((var, sign))
            var_to_clauses[var].append(i)
        clause_data.append(c_data)

    # Helper to check if a clause is satisfied given current assignment
    # assignment is a list of booleans (True/False)
    # sign 1 means var must be True, sign -1 means var must be False
    def is_sat(clause_idx, assignment):
        for var, sign in clause_data[clause_idx]:
            if assignment[var] == (sign == 1):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for i in range(m):
            if not is_sat(i, assignment):
                count += 1
        return count

    best_assignment = None
    best_unsat = m + 1

    # Restart loop
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Random initialization
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat count
        unsat_count = 0
        clause_sat = [False] * m
        for i in range(m):
            if is_sat(i, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # WalkSAT search
        # Parameters: 
        # - step_limit: max steps per restart
        # - gain_weight: probability to use gain heuristic vs random (or just use gain always with some noise)
        # Standard WalkSAT uses a parameter p (probability of random flip).
        # Here we implement Gain WalkSAT: choose flip with max gain, with some probability choose random.
        
        step_limit = 100 * n
        p_random = 0.3  # 30% chance to pick a random variable from the clause
        
        for step in range(step_limit):
            if time.time() > deadline - 5.0:
                return best_assignment
            
            if unsat_count == 0:
                return assignment

            # Pick a random unsatisfied clause
            # To avoid O(m) scan every step, we could maintain a list, but m is small.
            # However, finding a random unsat clause efficiently:
            # Simple approach: pick random index, if sat, try again.
            # Better: maintain a list of unsat clause indices? 
            # Given m ~ 1000, scanning is fine.
            
            # Let's pick a random unsat clause
            unsat_clauses = [i for i in range(m) if not clause_sat[i]]
            if not unsat_clauses:
                return assignment
                
            c_idx = random.choice(unsat_clauses)
            
            # Calculate gain for each variable in this clause
            # We want to flip a variable v such that the literal in c_idx becomes true.
            # Let's evaluate gain for setting v to satisfy the literal in c_idx.
            
            best_gain = -1000
            best_var = -1
            best_sign = 0
            
            # With probability p_random, pick a random variable from the clause
            if random.random() < p_random:
                var, sign = random.choice(clause_data[c_idx])
                best_var = var
                best_sign = sign
            else:
                for var, sign in clause_data[c_idx]:
                    # Calculate gain of setting assignment[var] = (sign == 1)
                    # New value for var
                    new_val = (sign == 1)
                    
                    # If var is already set to new_val, gain is 0 (no change)
                    # But wait, if c_idx is unsat, the literal (var, sign) is currently false.
                    # So assignment[var] != new_val.
                    # So flipping will change the state.
                    
                    gain = 0
                    # Check all clauses containing var
                    for c2_idx in var_to_clauses[var]:
                        # Determine if c2 is currently sat
                        is_c2_sat = clause_sat[c2_idx]
                        
                        # Determine if c2 would be sat after flip
                        # We need to check if any other literal in c2 is sat, 
                        # or if the literal for var becomes sat.
                        
                        # Efficient check:
                        # If c2 is currently sat:
                        #   It becomes unsat ONLY IF the literal for var was the ONLY true literal in c2.
                        # If c2 is currently unsat:
                        #   It becomes sat ONLY IF the literal for var becomes true (and it was false before).
                        
                        # Let's find the literal for var in c2
                        # We can iterate c2 to find var's sign, but that's slow.
                        # Precompute literal signs? 
                        # clause_data[c2] is list of (var, sign).
                        # We can just check.
                        
                        # Optimization: 
                        # If is_c2_sat:
                        #   Check if var's literal is true. It must be, since c2 is sat.
                        #   Check if any OTHER literal in c2 is true.
                        #   If no other literal is true, then flipping var to new_val (which makes var's literal false? 
                        #   Wait.
                        #   Current state: c2 is sat.
                        #   We are flipping var to new_val.
                        #   Does new_val make var's literal in c2 true or false?
                        #   We don't know the sign of var in c2 yet.
                        
                        # Let's do the standard O(size of clause) check for each c2.
                        # m is small, n is 150.
                        
                        # Find sign of var in c2
                        sign_in_c2 = None
                        for v2, s2 in clause_data[c2_idx]:
                            if v2 == var:
                                sign_in_c2 = s2
                                break
                                
                        if sign_in_c2 is None:
                            continue # Should not happen as var is in var_to_clauses[var]
                            
                        # Current truth value of var's literal in c2
                        current_lit_true = (assignment[var] == (sign_in_c2 == 1))
                        
                        # New truth value of var's literal in c2
                        new_lit_true = (new_val == (sign_in_c2 == 1))
                        
                        if is_c2_sat:
                            # Clause is currently satisfied.
                            # It becomes unsatisfied if:
                            # 1. The literal for var was true (current_lit_true is True)
                            # 2. The literal for var becomes false (new_lit_true is False)
                            # 3. No other literal in c2 is true.
                            if current_lit_true and not new_lit_true:
                                # Check if any other literal is true
                                other_true = False
                                for v2, s2 in clause_data[c2_idx]:
                                    if v2 != var:
                                        if assignment[v2] == (s2 == 1):
                                            other_true = True
                                            break
                                if not other_true:
                                    gain -= 1
                        else:
                            # Clause is currently unsatisfied.
                            # It becomes satisfied if:
                            # 1. The literal for var was false (current_lit_true is False)
                            # 2. The literal for var becomes true (new_lit_true is True)
                            if not current_lit_true and new_lit_true:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                        best_sign = sign
            
            # Flip best_var
            if best_var != -1:
                assignment[best_var] = (best_sign == 1)
                
                # Update clause_sat and unsat_count
                # Only need to update clauses in var_to_clauses[best_var]
                for c2_idx in var_to_clauses[best_var]:
                    was_sat = clause_sat[c2_idx]
                    is_now_sat = is_sat(c2_idx, assignment)
                    
                    if was_sat and not is_now_sat:
                        unsat_count += 1
                    elif not was_sat and is_now_sat:
                        unsat_count -= 1
                    
                    clause_sat[c2_idx] = is_now_sat

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback
        return [False] * n
