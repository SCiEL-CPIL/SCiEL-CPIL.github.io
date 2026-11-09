"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 280 seconds to be safe
    deadline = time.time() + 280
    
    # Precompute: for each variable, list of (clause_idx, sign) where sign is +1 if positive literal, -1 if negative
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((ci, sign))
    
    m = len(clauses)
    
    # Parameters
    base_noise = 0.5
    max_steps_per_restart = 20000
    restarts = 0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def is_lit_true(var_idx, sign, assignment):
        """Check if a literal is true given the assignment."""
        val = assignment[var_idx]
        return val if sign == 1 else (not val)
    
    def compute_clause_sat_counts(assignment):
        """Compute how many literals are true in each clause."""
        sat_counts = [0] * m
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var_idx = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                if is_lit_true(var_idx, sign, assignment):
                    count += 1
            sat_counts[ci] = count
        return sat_counts
    
    while time.time() < deadline:
        restarts += 1
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Calculate initial sat counts and unsat count
        sat_counts = compute_clause_sat_counts(assignment)
        unsat_count = sum(1 for sc in sat_counts if sc == 0)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive noise: start high, decrease if making progress
        noise = base_noise
        consecutive_no_improve = 0
        last_unsat = unsat_count
        
        # WalkSAT loop
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            # Find an unsatisfied clause (sat_count == 0)
            unsat_clause_idx = -1
            for ci in range(m):
                if sat_counts[ci] == 0:
                    unsat_clause_idx = ci
                    break
            
            if unsat_clause_idx == -1:
                # Satisfied
                return assignment
            
            unsat_clause = clauses[unsat_clause_idx]
            
            # Heuristic: MinConfl - for each literal in the clause, calculate damage
            min_damage = float('inf')
            best_lit = unsat_clause[0]
            
            for lit in unsat_clause:
                var_idx = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                
                # Damage: number of currently satisfied clauses that would become unsatisfied
                # if we flip var_idx
                damage = 0
                for (ci, s) in var_to_clauses[var_idx]:
                    # Clause ci contains variable var_idx with sign s
                    # Check if clause ci is currently satisfied
                    if sat_counts[ci] == 0:
                        continue  # Already unsatisfied, no damage
                    
                    # Check if the literal in clause ci for var_idx is currently true
                    lit_true = is_lit_true(var_idx, s, assignment)
                    if not lit_true:
                        # This literal is false, so flipping won't affect this clause's satisfaction
                        # (it was already not contributing)
                        continue
                    
                    # The literal is true. Check if it's the ONLY true literal in this clause
                    # i.e., sat_counts[ci] == 1
                    if sat_counts[ci] == 1:
                        damage += 1
                
                if damage < min_damage:
                    min_damage = damage
                    best_lit = lit
            
            # Decide flip
            if random.random() < noise:
                lit_to_flip = random.choice(unsat_clause)
            else:
                lit_to_flip = best_lit
            
            var_idx = abs(lit_to_flip) - 1
            sign = 1 if lit_to_flip > 0 else -1
            
            # Update assignment
            assignment[var_idx] = not assignment[var_idx]
            
            # Update sat_counts incrementally
            # For each clause containing var_idx, update its sat count
            for (ci, s) in var_to_clauses[var_idx]:
                # Before flip, check if literal was true
                old_true = is_lit_true(var_idx, s, assignment)  # This is now the NEW value
                # We need to be careful: assignment was already flipped
                # So old_true is actually the new truth value
                # Let me redo this:
                # After flip, the truth value of the literal with sign s is:
                # new_val = assignment[var_idx]
                # literal_true = new_val if s == 1 else not new_val
                
                # Actually, let's just recompute for affected clauses
                pass
            
            # Recompute sat_counts for affected clauses
            for (ci, s) in var_to_clauses[var_idx]:
                new_count = 0
                for l in clauses[ci]:
                    v = abs(l) - 1
                    sg = 1 if l > 0 else -1
                    if is_lit_true(v, sg, assignment):
                        new_count += 1
                sat_counts[ci] = new_count
            
            steps += 1
            
            # Check if satisfied
            if all(sc > 0 for sc in sat_counts):
                return assignment
            
            # Update best periodically
            if steps % 5 == 0:
                cur_unsat = sum(1 for sc in sat_counts if sc == 0)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment[:]
                
                # Adaptive noise
                if cur_unsat < last_unsat:
                    consecutive_no_improve = 0
                    last_unsat = cur_unsat
                    noise = max(0.1, base_noise - 0.05 * min(restarts, 10))
                else:
                    consecutive_no_improve += 1
                    if consecutive_no_improve > 50:
                        noise = min(0.9, base_noise + 0.1 * min(consecutive_no_improve // 50, 5))
        
        # End of restart steps
        # If we are running low on time, break outer loop
        if time.time() > deadline - 5:
            break
            
    return best_assignment
# EVOLVE-BLOCK-END
