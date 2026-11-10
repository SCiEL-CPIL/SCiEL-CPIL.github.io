"""Improved 3-SAT solver using local search with restarts and time management."""

import random
import time

# Module-level state for time management
_start_time = time.time()
# Assume 300s total limit, leave 10s buffer for safety/overhead
_DEADLINE = _start_time + 290.0


def solve(instance):
    """
    Solves a 3-SAT instance.
    Returns a list of booleans representing the assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process: Map variable indices to clauses containing them
    # Variables are 1-based in literals, 0-based in assignment
    var_to_clauses = [[] for _ in range(n)]
    for idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(idx)
    
    # Pre-compute clause literals as (var_idx, is_positive) for faster access
    clauses_processed = []
    for clause in clauses:
        processed_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            processed_clause.append((var_idx, is_pos))
        clauses_processed.append(processed_clause)
    
    m = len(clauses)
    
    # State: assignment as list of booleans, clause_satisfied as list of booleans
    # Initialize random assignment
    assignment = [random.random() < 0.5 for _ in range(n)]
    
    # Compute initial clause satisfaction
    clause_satisfied = [False] * m
    for i, clause in enumerate(clauses_processed):
        for var_idx, is_pos in clause:
            if assignment[var_idx] == is_pos:
                clause_satisfied[i] = True
                break
    
    unsat_count = sum(1 for s in clause_satisfied if not s)
    
    # Best solution tracking
    best_assignment = assignment.copy()
    best_unsat_count = unsat_count
    
    if unsat_count == 0:
        return assignment
    
    # WalkSAT with restarts
    # Parameters:
    # - max_restarts: number of restarts
    # - steps_per_restart: max steps per restart
    # - flip_threshold: probability of choosing random variable from clause (vs best)
    
    max_restarts = 100
    steps_per_restart = 2000
    flip_threshold = 0.5  # 50% chance to pick random, 50% best
    
    for restart in range(max_restarts):
        # Check time
        if time.time() > _DEADLINE - 2:
            return best_assignment

        # Reset to random assignment for each restart
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Recompute clause satisfaction
        clause_satisfied = [False] * m
        for i, clause in enumerate(clauses_processed):
            for var_idx, is_pos in clause:
                if assignment[var_idx] == is_pos:
                    clause_satisfied[i] = True
                    break
        
        unsat_count = sum(1 for s in clause_satisfied if not s)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        for step in range(steps_per_restart):
            # Check time
            if time.time() > _DEADLINE - 2:
                return best_assignment

            # Find unsatisfied clauses
            unsat_indices = [i for i in range(m) if not clause_satisfied[i]]
            if not unsat_indices:
                return assignment
            
            # Pick random unsatisfied clause
            clause_idx = random.choice(unsat_indices)
            clause = clauses_processed[clause_idx]
            
            # Decide: random flip or best flip
            if random.random() < flip_threshold:
                # Random flip: pick random variable from clause
                var_idx, is_pos = random.choice(clause)
            else:
                # Best flip: pick variable that minimizes increase in unsat count
                best_var = -1
                best_delta = float('inf')
                
                for var_idx, is_pos in clause:
                    # Calculate delta if we flip var_idx
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        c = clauses_processed[c_idx]
                        # Check if this clause would change status
                        # If clause is currently satisfied, flipping var_idx might unsatisfy it
                        # If clause is currently unsatisfied, flipping var_idx might satisfy it
                        old_sat = clause_satisfied[c_idx]
                        
                        # Determine if flipping var_idx changes satisfaction
                        # A clause is satisfied if any literal is true
                        # Flipping var_idx changes the truth value of the literal in this clause
                        # If the clause has other literals that are true, it stays satisfied
                        # If the flipped literal was the only true one, it becomes unsatisfied
                        # If the flipped literal was false and clause was unsatisfied, it might become satisfied
                        
                        # Count how many other literals in clause are true
                        other_true_count = 0
                        for v, p in c:
                            if v != var_idx:
                                if assignment[v] == p:
                                    other_true_count += 1
                        
                        # Current truth of var_idx literal
                        current_val = assignment[var_idx]
                        current_lit_true = (current_val == is_pos)
                        
                        # New truth of var_idx literal
                        new_lit_true = not current_lit_true
                        
                        # New satisfaction
                        new_sat = other_true_count > 0 or new_lit_true
                        
                        if old_sat and not new_sat:
                            delta += 1
                        elif not old_sat and new_sat:
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_var = var_idx
                
                var_idx = best_var
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause satisfaction efficiently
            for c_idx in var_to_clauses[var_idx]:
                c = clauses_processed[c_idx]
                # Recompute satisfaction for this clause
                new_sat = False
                for v, p in c:
                    if assignment[v] == p:
                        new_sat = True
                        break
                
                # Update unsat_count
                if clause_satisfied[c_idx] and not new_sat:
                    unsat_count += 1
                elif not clause_satisfied[c_idx] and new_sat:
                    unsat_count -= 1
                
                clause_satisfied[c_idx] = new_sat
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

    return best_assignment
