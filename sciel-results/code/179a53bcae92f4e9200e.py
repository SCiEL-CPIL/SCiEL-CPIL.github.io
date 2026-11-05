"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Preprocess clauses into numpy arrays for vectorization
    # Convert 1-based signed literals to 0-based indices and signs
    # indices: shape (m, 3), values in [0, n-1]
    # signs: shape (m, 3), values in {+1, -1}
    
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
            
    # Variable occurrence lists for fast gain calculation
    # var_to_clauses[var] = list of clause indices containing var
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_to_clauses[v].append(i)
    
    # Precompute positions of each variable in its clauses for faster gain calc
    # var_positions[var] = list of (clause_idx, position_in_clause)
    var_positions = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_positions[v].append((i, j))
    
    # Convert to numpy arrays of lists for easier handling? 
    # Actually, lists of lists are fine for the inner loop if we optimize the gain calc.
    # But we can precompute the "other literals" for each variable occurrence to speed up gain.
    
    # state: boolean array of assignments
    # sat_counts: integer array of number of satisfied literals per clause (0, 1, 2, 3)
    
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def calculate_sat_counts(assignment):
        val = np.take(assignment, indices)
        is_sat = (val == (signs > 0))
        return np.sum(is_sat, axis=1)

    def flip_variable(var, assignment, sat_counts):
        """
        Flip variable var, update assignment and sat_counts.
        Returns new assignment and sat_counts.
        """
        positions = var_positions[var]
        if not positions:
            return assignment, sat_counts
        
        # Update assignment
        assignment[var] = not assignment[var]
        
        # Update sat counts for affected clauses using precomputed positions
        for ci, pos in positions:
            # Recompute sat count for this clause
            val = np.take(assignment, indices[ci])
            is_sat = (val == (signs[ci] > 0))
            sat_counts[ci] = np.sum(is_sat)
        
        return assignment, sat_counts

    def calculate_gain(var, assignment, sat_counts):
        """Calculate the gain (number of clauses that become satisfied minus those that become unsatisfied) if we flip var."""
        positions = var_positions[var]
        if not positions:
            return 0
        
        gain = 0
        current_val = assignment[var]
        new_val = not current_val
        
        for ci, pos in positions:
            desired = signs[ci, pos] > 0
            old_sat = (current_val == desired)
            new_sat = (new_val == desired)
            
            if old_sat and not new_sat:
                # This clause loses a satisfied literal
                if sat_counts[ci] == 1:
                    gain -= 1
            elif not old_sat and new_sat:
                # This clause gains a satisfied literal
                if sat_counts[ci] == 0:
                    gain += 1
        
        return gain

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    noise = 0.3
    base_flip_budget = 10000
    restart_count = 0
    last_improvement_step = 0
    
    while time.time() < deadline:
        # Restart logic with random restart
        assignment = initialize_assignment()
        sat_counts = calculate_sat_counts(assignment)
        
        unsat_clauses = np.where(sat_counts == 0)[0]
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            last_improvement_step = 0
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive flip budget: increase if we're making progress, decrease if stuck
        if restart_count == 0:
            flip_budget = base_flip_budget
        else:
            # Exponential backoff with some randomness
            flip_budget = int(base_flip_budget * (1.0 + 0.5 * random.random()))
        
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = current_unsat
        
        while steps < flip_budget and time.time() < deadline:
            if len(unsat_clauses) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_clauses[random.randint(0, len(unsat_clauses) - 1)]
            
            # Pick a literal to flip
            # WalkSAT heuristic: flip with probability noise randomly, otherwise pick best gain
            if random.random() < noise:
                # Random flip from clause
                pos = random.randint(0, 2)
                var = indices[clause_idx, pos]
            else:
                # Find var in clause that maximizes gain
                best_gain = -100
                best_var = -1
                
                for pos in range(3):
                    var = indices[clause_idx, pos]
                    gain = calculate_gain(var, assignment, sat_counts)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            assignment, sat_counts = flip_variable(var, assignment, sat_counts)
            
            # Update unsat_clauses list
            unsat_clauses = np.where(sat_counts == 0)[0]
            
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                no_improvement_steps = 0
                last_improvement_step = steps
            else:
                no_improvement_steps += 1
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improvement_steps = 0
            
            steps += 1
            
            # Adapt noise: decrease gradually, but increase if stuck
            if steps % 200 == 0:
                if no_improvement_steps > 500:
                    noise = min(0.7, noise + 0.05)
                else:
                    noise = max(0.1, noise - 0.01)
        
        restart_count += 1

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
