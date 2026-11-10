import random
import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 300s limit, use 290s to be safe
    deadline = time.time() + 290.0
    
    # Pre-compute variable to clauses mapping
    # var_clauses[i] = list of (clause_index, sign) where sign is 1 if literal is positive, -1 if negative
    var_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[var_idx].append((c_idx, sign))
    
    m = len(clauses)
    
    def initialize_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]
    
    def calculate_sat_counts(assignment):
        sat_count = [0] * m
        for c_idx, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var_idx = abs(lit) - 1
                is_true = assignment[var_idx]
                if (lit > 0 and is_true) or (lit < 0 and not is_true):
                    count += 1
            sat_count[c_idx] = count
        return sat_count

    def flip_variable(assignment, var_idx, sat_count):
        """
        Flips assignment[var_idx] and updates sat_count in place.
        Returns the change in total unsatisfied clauses.
        """
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        
        change_in_unsat = 0
        for c_idx, sign in var_clauses[var_idx]:
            # Determine if the literal was satisfied before and after
            # Literal is satisfied if (sign > 0 and assignment is True) or (sign < 0 and assignment is False)
            
            # Before flip:
            if sign > 0:
                was_sat = old_val
                is_sat = new_val
            else:
                was_sat = not old_val
                is_sat = not new_val
            
            if was_sat and not is_sat:
                # Literal became unsatisfied
                sat_count[c_idx] -= 1
                if sat_count[c_idx] == 0:
                    change_in_unsat += 1 # Clause became unsatisfied
            elif not was_sat and is_sat:
                # Literal became satisfied
                sat_count[c_idx] += 1
                if sat_count[c_idx] == 1:
                    change_in_unsat -= 1 # Clause became satisfied
            # If was_sat and is_sat or not was_sat and not is_sat, no change in sat_count or status
            
        return change_in_unsat

    def get_gain(assignment, sat_count, var_idx, target_clause_idx):
        """
        Calculate gain (reduction in unsat count) if we flip var_idx.
        Assumes var_idx is in an unsatisfied clause (target_clause_idx).
        """
        # The target clause is unsatisfied, so all its literals are false.
        # Flipping var_idx will make its literal in target_clause_idx true.
        # So target_clause_idx will become satisfied. Gain +1 from this clause.
        gain = 1
        
        # Check other clauses containing var_idx
        for c_idx, sign in var_clauses[var_idx]:
            if c_idx == target_clause_idx:
                continue
            # If this clause is currently satisfied, and the literal is currently true,
            # flipping var_idx will make the literal false.
            # If this was the only satisfied literal, the clause becomes unsatisfied.
            if sat_count[c_idx] > 0:
                # Check if the literal is currently satisfied
                var_val = assignment[var_idx]
                if sign > 0:
                    lit_sat = var_val
                else:
                    lit_sat = not var_val
                
                if lit_sat:
                    # If we flip, this literal becomes unsatisfied.
                    # If sat_count[c_idx] was 1, it becomes 0 (unsat).
                    if sat_count[c_idx] == 1:
                        gain -= 1
                    # If sat_count[c_idx] > 1, it remains satisfied.
        
        return gain

    def walk_sat_step(assignment, sat_count, unsat_count, rng, noise_prob):
        # Find unsatisfied clauses
        unsat_clauses = [i for i in range(m) if sat_count[i] == 0]
        if not unsat_clauses:
            return True, 0 # Solved
        
        # Pick a random unsatisfied clause
        target_c_idx = rng.choice(unsat_clauses)
        clause = clauses[target_c_idx]
        
        # Pick a variable
        if rng.random() < noise_prob:
            lit = rng.choice(clause)
            var_idx = abs(lit) - 1
        else:
            # Best flip
            best_var = -1
            best_gain = -100
            for lit in clause:
                var_idx = abs(lit) - 1
                gain = get_gain(assignment, sat_count, var_idx, target_c_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_var = var_idx
            
            if best_var == -1:
                # Fallback to random if something went wrong
                lit = rng.choice(clause)
                best_var = abs(lit) - 1
            
            var_idx = best_var
        
        # Flip and update
        delta_unsat = flip_variable(assignment, var_idx, sat_count)
        return False, delta_unsat

    rng = random.Random()
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 1000
    steps_per_restart = 5000
    noise_prob = 0.2
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = initialize_assignment(rng)
        sat_count = calculate_sat_counts(assignment)
        unsat_count = sum(1 for c in sat_count if c == 0)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        steps_without_improvement = 0
        local_best_unsat = unsat_count
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment if best_assignment is not None else assignment
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            if steps_without_improvement > 100:
                break # Restart
                
            _, delta_unsat = walk_sat_step(assignment, sat_count, unsat_count, rng, noise_prob)
            unsat_count += delta_unsat
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
                
    # If we exhausted time or restarts, return the best found
    if best_assignment is not None:
        return best_assignment
    else:
        return assignment
