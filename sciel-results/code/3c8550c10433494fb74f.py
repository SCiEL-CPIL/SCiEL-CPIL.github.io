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
    m = len(clauses)
    
    # Pre-process: Map variable indices to clauses containing them
    # Variables are 1-based in literals, 0-based in assignment
    var_to_clauses = [[] for _ in range(n)]
    for idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(idx)
    
    # Pre-compute clause literal info for faster evaluation
    # clause_info[c] = list of (var_idx, sign) where sign is 1 for positive, -1 for negative
    clause_info = []
    for clause in clauses:
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            info.append((var_idx, sign))
        clause_info.append(info)
    
    def clause_sat_count(clause_idx, assignment):
        """Count how many literals in clause are satisfied (0, 1, 2, or 3)."""
        count = 0
        for var_idx, sign in clause_info[clause_idx]:
            if assignment[var_idx] == (sign > 0):
                count += 1
        return count
    
    def init_clause_sat(assignment):
        """Initialize clause satisfaction counts."""
        sat_counts = [0] * m
        for c in range(m):
            sat_counts[c] = clause_sat_count(c, assignment)
        return sat_counts
    
    def update_clause_sat(var_idx, new_val, sat_counts):
        """Update satisfaction counts for all clauses containing var_idx."""
        for c in var_to_clauses[var_idx]:
            old_count = sat_counts[c]
            new_count = 0
            for var, sign in clause_info[c]:
                if var == var_idx:
                    if new_val == (sign > 0):
                        new_count += 1
                else:
                    if assignment[var] == (sign > 0):
                        new_count += 1
            sat_counts[c] = new_count
    
    def count_unsat_from_sat(sat_counts):
        """Count unsatisfied clauses from satisfaction counts."""
        return sum(1 for s in sat_counts if s == 0)
    
    def get_unsat_clauses(sat_counts):
        """Get list of unsatisfied clause indices."""
        return [c for c in range(m) if sat_counts[c] == 0]
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Weighted random restart schedule
    # Use a geometric distribution for restart lengths
    base_steps = 2000
    max_steps = 20000
    
    restart = 0
    while time.time() < _DEADLINE - 5:
        # Check time
        if time.time() > _DEADLINE - 2:
            if best_assignment:
                return best_assignment
            return [random.random() < 0.5 for _ in range(n)]
        
        # Determine steps for this restart (geometric-like)
        steps_this_restart = min(max_steps, int(base_steps * (1 + random.expovariate(1.0))))
        
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Initial clause satisfaction counts
        sat_counts = init_clause_sat(assignment)
        unsat_count = count_unsat_from_sat(sat_counts)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        for step in range(steps_this_restart):
            # Check time periodically (every 100 steps to avoid overhead)
            if step % 100 == 0 and time.time() > _DEADLINE - 2:
                if best_assignment:
                    return best_assignment
                return assignment
            
            # Find unsatisfied clauses
            unsat_clauses = get_unsat_clauses(sat_counts)
            if not unsat_clauses:
                return assignment
            
            # Randomly pick an unsatisfied clause
            clause_idx = random.choice(unsat_clauses)
            
            # Find best literal to flip in this clause
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, sign in clause_info[clause_idx]:
                new_val = not assignment[var_idx]
                
                # Calculate delta: change in unsat count
                delta = 0
                for c in var_to_clauses[var_idx]:
                    old_sat = sat_counts[c]
                    # Compute new sat count for this clause
                    new_sat = 0
                    for var, s in clause_info[c]:
                        if var == var_idx:
                            if new_val == (s > 0):
                                new_sat += 1
                        else:
                            if assignment[var] == (s > 0):
                                new_sat += 1
                    
                    # Delta in unsat status
                    old_unsat = (old_sat == 0)
                    new_unsat = (new_sat == 0)
                    if not old_unsat and new_unsat:
                        delta += 1
                    elif old_unsat and not new_unsat:
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
                elif delta == best_delta and random.random() < 0.5:
                    best_var = var_idx
            
            # Flip the variable
            if best_var != -1:
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                new_val = assignment[best_var]
                
                # Update clause satisfaction counts
                update_clause_sat(best_var, new_val, sat_counts)
                new_unsat_count = count_unsat_from_sat(sat_counts)
                
                if new_unsat_count == 0:
                    return assignment
                
                if new_unsat_count < best_unsat_count:
                    best_unsat_count = new_unsat_count
                    best_assignment = assignment.copy()
        
        restart += 1
    
    # Final polish: if we have a good best_assignment, try to improve it further
    if best_assignment is not None and best_unsat_count > 0 and time.time() < _DEADLINE - 10:
        # Start from best_assignment and do more intensive local search
        assignment = best_assignment.copy()
        sat_counts = init_clause_sat(assignment)
        unsat_count = count_unsat_from_sat(sat_counts)
        
        # Use a smaller step limit but more restarts
        for polish_restart in range(20):
            if time.time() > _DEADLINE - 5:
                break
            
            for step in range(5000):
                if time.time() > _DEADLINE - 5:
                    break
                
                unsat_clauses = get_unsat_clauses(sat_counts)
                if not unsat_clauses:
                    return assignment
                
                clause_idx = random.choice(unsat_clauses)
                
                best_var = -1
                best_delta = float('inf')
                
                for var_idx, sign in clause_info[clause_idx]:
                    new_val = not assignment[var_idx]
                    
                    delta = 0
                    for c in var_to_clauses[var_idx]:
                        old_sat = sat_counts[c]
                        new_sat = 0
                        for var, s in clause_info[c]:
                            if var == var_idx:
                                if new_val == (s > 0):
                                    new_sat += 1
                            else:
                                if assignment[var] == (s > 0):
                                    new_sat += 1
                        
                        old_unsat = (old_sat == 0)
                        new_unsat = (new_sat == 0)
                        if not old_unsat and new_unsat:
                            delta += 1
                        elif old_unsat and not new_unsat:
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                    elif delta == best_delta and random.random() < 0.5:
                        best_var = var_idx
                
                if best_var != -1:
                    assignment[best_var] = not assignment[best_var]
                    update_clause_sat(best_var, assignment[best_var], sat_counts)
                    new_unsat_count = count_unsat_from_sat(sat_counts)
                    
                    if new_unsat_count == 0:
                        return assignment
                    
                    if new_unsat_count < best_unsat_count:
                        best_unsat_count = new_unsat_count
                        best_assignment = assignment.copy()
    
    if best_assignment:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
