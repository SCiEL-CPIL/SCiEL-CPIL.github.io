"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute variable indices for faster access
    # Convert clauses to 0-based indices with sign
    # l > 0 means variable l-1 is True, l < 0 means variable -l-1 is False
    # We store (var_idx, is_positive)
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            if l > 0:
                pc.append((l - 1, True))
            else:
                pc.append((-l - 1, False))
        processed_clauses.append(pc)
        
    # Map variable to clauses containing it for gain calculation
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(processed_clauses):
        for v_idx, _ in c:
            var_to_clauses[v_idx].append(i)
            
    def is_satisfied(clause, assignment):
        for v_idx, is_pos in clause:
            if assignment[v_idx] == is_pos:
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for c in processed_clauses:
            if not is_satisfied(c, assignment):
                count += 1
        return count

    def walk_sat_step(assignment, rng, p_random):
        # Find unsatisfied clauses
        unsat_clauses = [c for c in processed_clauses if not is_satisfied(c, assignment)]
        if not unsat_clauses:
            return assignment, True
        
        # Pick a random unsatisfied clause
        clause = rng.choice(unsat_clauses)
        
        # If random flip, pick a random variable in the clause
        if rng.random() < p_random:
            v_idx, is_pos = rng.choice(clause)
            assignment[v_idx] = not assignment[v_idx]
            return assignment, False
        
        # Otherwise, pick the variable that gives the best gain
        # Gain = (unsat clauses that become sat) - (sat clauses that become unsat)
        # Since the clause is unsat, flipping any variable in it will satisfy the clause (gain +1 for this clause).
        # We need to check the other clauses containing that variable.
        best_gain = -1000
        best_var = None
        
        for v_idx, is_pos in clause:
            # Current state of variable
            current_val = assignment[v_idx]
            # New state
            new_val = not current_val
            
            # Calculate gain
            gain = 0
            # 1. The selected clause becomes satisfied (since it was unsat and we are flipping one of its literals to match)
            # Wait, is it guaranteed to become satisfied?
            # Yes, if we flip the variable to satisfy the literal in this clause.
            # The literal is (v_idx, is_pos). 
            # If is_pos is True, we set assignment[v_idx] to True.
            # If is_pos is False, we set assignment[v_idx] to False.
            # So this clause will be satisfied.
            gain += 1
            
            # 2. Check other clauses containing this variable
            for c_idx in var_to_clauses[v_idx]:
                if c_idx == id(clause): # This is not a reliable check, need index
                    continue
                # We need to know if this clause was satisfied before and if it will be unsatisfied after
                # It's easier to just simulate the flip for these clauses
                # But we can optimize:
                # A clause containing v_idx is affected only if v_idx is the *only* literal in that clause that is currently satisfied.
                # If so, flipping v_idx will make it unsatisfied.
                
                c = processed_clauses[c_idx]
                # Check if clause is currently satisfied
                is_sat_before = False
                # Check if any other literal in the clause is satisfied
                # If the clause is currently unsatisfied, flipping v_idx to satisfy its literal won't make it unsat (it was already unsat).
                # Actually, if it's unsat, flipping v_idx might make it sat (if v_idx was the only unsat literal) or keep it unsat.
                # If it's sat, flipping v_idx might make it unsat (if v_idx was the only sat literal).
                
                # Let's just count the delta directly for simplicity and correctness, n is small (150)
                # Clauses per var is roughly 3 * (m/n) * 2? No, 3 literals per clause.
                # Avg clauses per var is 3 * m / n. If m=450, n=150, avg 9.
                # So checking ~9 clauses per variable is fast.
                
                # Check state before flip
                sat_before = False
                for v2, pos2 in c:
                    if v2 == v_idx:
                        # This is the variable we are flipping
                        if assignment[v2] == pos2:
                            sat_before = True
                            break
                    else:
                        if assignment[v2] == pos2:
                            sat_before = True
                            break
                
                # Check state after flip
                sat_after = False
                for v2, pos2 in c:
                    if v2 == v_idx:
                        if new_val == pos2:
                            sat_after = True
                            break
                    else:
                        if assignment[v2] == pos2:
                            sat_after = True
                            break
                
                if sat_before and not sat_after:
                    gain -= 1
                elif not sat_before and sat_after:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_var = (v_idx, is_pos)
        
        if best_var:
            v_idx, is_pos = best_var
            assignment[v_idx] = not assignment[v_idx]
        
        return assignment, False

    # Main loop with restarts and time guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    max_steps_per_restart = 5000
    max_restarts = 10000
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        if current_unsat == 0:
            return assignment
        
        # WalkSAT steps
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            assignment, is_solved = walk_sat_step(assignment, rng, 0.2)
            
            if is_solved:
                return assignment
            
            # Update best if improved
            # Optimization: Only recount if we flipped a variable that might have improved things?
            # Or just check every 10 steps? 
            # For n=150, recounting is fast enough.
            current_unsat = count_unsat(assignment)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            if current_unsat == 0:
                return assignment
        
    return best_assignment

# EVOLVE-BLOCK-END
