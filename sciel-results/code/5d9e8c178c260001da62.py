"""Improved planted 3-SAT solver with adaptive WalkSAT, clause weighting, and time guard."""
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Pre-compute clause literal indices for speed
    # Store as list of (var_idx, polarity) where polarity: 0 means False satisfies, 1 means True satisfies
    clause_vars = []
    for c in clauses:
        vars_in_c = []
        for l in c:
            var_idx = abs(l) - 1
            # If l > 0, literal is true when assignment[var_idx] is True (polarity 1)
            # If l < 0, literal is true when assignment[var_idx] is False (polarity 0)
            polarity = 1 if l > 0 else 0
            vars_in_c.append((var_idx, polarity))
        clause_vars.append(vars_in_c)
    
    # Count occurrences of each variable in clauses for weighting
    var_occurrence = [0] * n
    for c in clause_vars:
        for var_idx, _ in c:
            var_occurrence[var_idx] += 1
    
    # Build a deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    def is_satisfied(clause_idx):
        c_vars = clause_vars[clause_idx]
        for var_idx, polarity in c_vars:
            if assignment[var_idx] == (polarity == 1):
                return True
        return False
    
    def count_unsat_clauses(assignment):
        count = 0
        for i in range(num_clauses):
            c_vars = clause_vars[i]
            sat = False
            for var_idx, polarity in c_vars:
                if assignment[var_idx] == (polarity == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Multiple restarts with different parameters
    # Use a mix of WalkSAT variants
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Adaptive parameters based on restart
        # Early restarts: more random, later restarts: more greedy
        if restart < 10:
            p_greedy = 0.6
            flip_budget = 15000
        elif restart < 25:
            p_greedy = 0.7
            flip_budget = 20000
        else:
            p_greedy = 0.8
            flip_budget = 25000
            
        # Seed based on restart for reproducibility but variation
        rng = random.Random(restart * 12345 + 67890)
        
        # Initialize assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track unsatisfied clauses
        unsat_indices = []
        for i in range(num_clauses):
            c_vars = clause_vars[i]
            sat = False
            for var_idx, polarity in c_vars:
                if assignment[var_idx] == (polarity == 1):
                    sat = True
                    break
            if not sat:
                unsat_indices.append(i)
        
        current_unsat = len(unsat_indices)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # WalkSAT loop
        for step in range(flip_budget):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return assignment
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            c_vars = clause_vars[clause_idx]
            
            # Count how many literals in this clause are false (all are false since clause is unsat)
            # We want to flip a variable. Use a scoring scheme:
            # For each variable in the clause, compute the "gain" if we flip it
            # Gain = number of clauses that would become satisfied - number that would become unsatisfied
            
            best_var = None
            best_score = -float('inf')
            best_is_greedy = False
            
            for var_idx, polarity in c_vars:
                # If we flip var_idx, this clause becomes satisfied (since currently all literals are false)
                # We need to check the impact on other clauses
                # For efficiency, we'll use a heuristic: prefer variables with higher occurrence
                # and use a random tie-breaking
                
                # Compute a simple score: occurrence count with some randomness
                score = var_occurrence[var_idx]
                
                # Add some randomness
                score += rng.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_var = var_idx
                    best_is_greedy = True
            
            # With probability (1 - p_greedy), pick randomly from the clause
            if rng.random() > p_greedy:
                var_idx, polarity = c_vars[rng.randrange(len(c_vars))]
                best_var = var_idx
                best_is_greedy = False
            
            # Flip the variable
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update unsat_indices: we need to check clauses containing best_var
            # For simplicity and correctness, we'll do a partial update
            # But for efficiency, let's just recompute unsat_indices periodically
            # Actually, let's track it properly but efficiently
            
            # Remove the clause we just satisfied from unsat_indices if it was there
            # And check if any other clauses became unsatisfied
            
            # For efficiency, we'll use a simpler approach: maintain a count and occasionally recompute
            # Let's just recompute unsat_indices every 100 steps or when needed
            
            if step % 50 == 49 or step == flip_budget - 1:
                new_unsat_indices = []
                for i in range(num_clauses):
                    c_vars_i = clause_vars[i]
                    sat = False
                    for v_idx, pol in c_vars_i:
                        if assignment[v_idx] == (pol == 1):
                            sat = True
                            break
                    if not sat:
                        new_unsat_indices.append(i)
                unsat_indices = new_unsat_indices
                current_unsat = len(unsat_indices)
                
                if current_unsat == 0:
                    return assignment
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
            
            # Also check if we just solved it (quick check)
            if len(unsat_indices) <= 1:
                # Verify
                for i in range(num_clauses):
                    c_vars_i = clause_vars[i]
                    sat = False
                    for v_idx, pol in c_vars_i:
                        if assignment[v_idx] == (pol == 1):
                            sat = True
                            break
                    if not sat:
                        break
                else:
                    return assignment
        
        # After flip budget, check final state
        final_unsat = count_unsat_clauses(assignment)
        if final_unsat == 0:
            return assignment
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = assignment[:]
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    # Fallback: return a random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
