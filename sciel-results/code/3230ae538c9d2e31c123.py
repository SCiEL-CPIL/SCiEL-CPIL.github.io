"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using an optimized WalkSAT with:
    1. Phase saving heuristic initialization
    2. Incremental maintenance of unsatisfied clauses and satisfaction counts
    3. Multiple restarts with increasing effort
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Build literal occurrence counts for phase saving
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    
    # Pre-compute clause data and variable-clause mappings
    # clause_data[i] = list of (var_idx, sign)
    clause_data = []
    # var_clauses[j] = list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cdata = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = lit > 0
            cdata.append((var_idx, sign))
            var_clauses[var_idx].append((i, sign))
        clause_data.append(cdata)
    
    # Helper to initialize assignment and state
    def init_state(assignment):
        # clause_sat_counts[i] = number of literals in clause i that are satisfied
        clause_sat_counts = [0] * m
        unsat_clauses = []
        
        for i in range(m):
            cdata = clause_data[i]
            count = 0
            for var_idx, sign in cdata:
                if assignment[var_idx] == sign:
                    count += 1
            clause_sat_counts[i] = count
            if count == 0:
                unsat_clauses.append(i)
        
        return clause_sat_counts, unsat_clauses
    
    # Phase saving initial assignment
    base_assignment = [pos_count[i] > neg_count[i] for i in range(n)]
    
    # Check if base assignment is a solution
    clause_sat_counts, unsat_clauses = init_state(base_assignment)
    if not unsat_clauses:
        return base_assignment
    
    # WalkSAT parameters
    max_time = 290.0 # Leave some buffer for 300s limit
    start_time = time.time()
    
    # Restart strategy
    restarts = 0
    max_restarts = 50
    base_steps = 10000
    growth_factor = 1.2
    
    while restarts < max_restarts:
        if time.time() - start_time > max_time:
            break
            
        # Initialize assignment
        if restarts == 0:
            assignment = base_assignment[:]
        else:
            # Random restart with some phase saving bias
            rng_init = random.Random(1000 + restarts)
            assignment = [rng_init.random() < 0.5 for _ in range(n)]
            # Blend with phase saving: 30% chance to use phase saving value
            for i in range(n):
                if rng_init.random() < 0.3:
                    assignment[i] = base_assignment[i]
        
        clause_sat_counts, unsat_clauses = init_state(assignment)
        
        if not unsat_clauses:
            return assignment
        
        steps = int(base_steps * (growth_factor ** restarts))
        
        rng = random.Random(2000 + restarts)
        
        for step in range(steps):
            if not unsat_clauses:
                return assignment
            
            if time.time() - start_time > max_time:
                break
            
            # Select a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cdata = clause_data[ci]
            
            # Determine which variable to flip
            # We want to minimize the number of broken clauses
            # and maximize the number of fixed clauses.
            
            best_var = cdata[0][0]
            best_score = float('inf')
            
            for var_idx, sign in cdata:
                # Calculate score: (broken - fixed)
                # Broken: clauses that are currently satisfied ONLY by this variable's current value
                # Fixed: clauses that are currently unsatisfied and would be satisfied by flipping this variable
                
                broken = 0
                fixed = 0
                
                for k, (other_ci, other_sign) in enumerate(var_clauses[var_idx]):
                    # Check if this clause is the one we selected? No, we check all.
                    # Actually, for the selected clause, flipping this var will definitely satisfy it
                    # because the clause is unsat (0 sat literals) and we are setting the var to the required sign.
                    # Wait, if the clause has 0 sat literals, flipping any var in it to the correct sign makes it 1 sat literal.
                    # So the selected clause is always 'fixed' (or rather, becomes satisfied).
                    
                    # Let's iterate through all clauses containing var_idx
                    # We need to know if the clause is currently satisfied and if it would remain satisfied.
                    
                    # Current state of clause 'other_ci':
                    # sat_count = clause_sat_counts[other_ci]
                    
                    # If we flip var_idx:
                    # old_value = assignment[var_idx]
                    # new_value = not old_value
                    
                    # Does the clause currently rely on var_idx?
                    # It relies on var_idx if:
                    # 1. The literal (var_idx, sign_in_clause) is currently satisfied (assignment[var_idx] == sign_in_clause)
                    # 2. AND the clause has exactly 1 satisfied literal (sat_count == 1)
                    
                    # Does the clause benefit from the flip?
                    # It benefits if:
                    # 1. The clause is currently unsatisfied (sat_count == 0)
                    # 2. AND the new value of var_idx satisfies the literal (not assignment[var_idx] == sign_in_clause)
                    
                    sat_count = clause_sat_counts[other_ci]
                    
                    # Check if current assignment satisfies the literal in this clause
                    if assignment[var_idx] == other_sign:
                        # The literal is currently satisfied.
                        if sat_count == 1:
                            # Flipping will break this clause
                            broken += 1
                    else:
                        # The literal is currently not satisfied.
                        if sat_count == 0:
                            # Flipping will satisfy this clause (since we are flipping to the value that satisfies it? 
                            # Wait. We are flipping var_idx. 
                            # The new value is not assignment[var_idx].
                            # If assignment[var_idx] != other_sign, then not assignment[var_idx] == other_sign.
                            # So yes, the literal becomes satisfied.
                            fixed += 1
                    
                    score = broken - fixed
                    if score < best_score:
                        best_score = score
                        best_var = var_idx
            
            # Apply noise: with some probability, flip a random variable from the clause
            # Noise probability can be adaptive, but let's keep it simple: 0.25
            if rng.random() < 0.25:
                var_idx, sign = rng.choice(cdata)
            else:
                var_idx = best_var
            
            # Perform the flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update state
            for k, (other_ci, other_sign) in enumerate(var_clauses[var_idx]):
                # Determine if the literal was satisfied before
                was_sat = (old_val == other_sign)
                # Determine if the literal is satisfied after
                is_sat = (not old_val == other_sign)
                
                if was_sat and not is_sat:
                    # Clause lost a satisfied literal
                    clause_sat_counts[other_ci] -= 1
                    if clause_sat_counts[other_ci] == 0:
                        unsat_clauses.append(other_ci)
                elif not was_sat and is_sat:
                    # Clause gained a satisfied literal
                    clause_sat_counts[other_ci] += 1
                    if clause_sat_counts[other_ci] > 0:
                        # It was unsat (0) and is now sat (>0)
                        # Remove from unsat_clauses
                        # Since it's a list, removal is O(N). 
                        # For better performance, we could use a set or a heap, but list is okay for small N.
                        # To avoid O(N^2) if many removals, we can just check and remove if present.
                        # However, `unsat_clauses` can have duplicates if we are not careful?
                        # No, we only add when count becomes 0. We only remove when count becomes >0.
                        # So each clause is in the list at most once.
                        if other_ci in unsat_clauses:
                            unsat_clauses.remove(other_ci)
            
            if not unsat_clauses:
                return assignment
        
        restarts += 1
    
    # If we ran out of time or restarts, return the current assignment (which might not be a solution)
    # But the task expects a satisfying assignment. If we fail, we return what we have.
    # The score is fraction solved, so returning a non-satisfying assignment scores 0 for that instance.
    return assignment
# EVOLVE-BLOCK-END
