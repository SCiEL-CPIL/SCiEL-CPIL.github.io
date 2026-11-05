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

    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
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
    
    # Precompute for each variable occurrence: the other two literals' clause and position
    # var_occ_data[var] = list of (clause_idx, pos, other1_clause, other1_pos, other2_clause, other2_pos)
    # Actually, for gain we need to know for each clause containing var, what are the other literals
    # Let's precompute: for each (var, clause_idx, pos), the indices and signs of the other 2 literals
    
    # State tracking:
    # assignment: boolean array
    # lit_sat: for each literal occurrence (m,3), is it satisfied?
    # sat_counts: for each clause, how many literals are satisfied
    
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def compute_initial_state(assignment):
        """Compute lit_sat and sat_counts from assignment."""
        # lit_sat[i,j] = assignment[indices[i,j]] if signs[i,j]>0 else not assignment[indices[i,j]]
        val = np.take(assignment, indices)
        lit_sat = (val == (signs > 0))
        sat_counts = np.sum(lit_sat, axis=1)
        return lit_sat, sat_counts

    def flip_variable(var, assignment, lit_sat, sat_counts):
        """
        Flip variable var, update assignment, lit_sat, and sat_counts incrementally.
        """
        positions = var_positions[var]
        if not positions:
            return assignment, lit_sat, sat_counts
        
        # Update assignment
        assignment[var] = not assignment[var]
        
        # Update lit_sat and sat_counts for affected clauses
        for ci, pos in positions:
            old_sat = lit_sat[ci, pos]
            lit_sat[ci, pos] = not old_sat
            sat_counts[ci] += 1 if not old_sat else -1
        
        return assignment, lit_sat, sat_counts

    def calculate_gain(var, lit_sat, sat_counts):
        """Calculate the gain if we flip var."""
        positions = var_positions[var]
        if not positions:
            return 0
        
        gain = 0
        for ci, pos in positions:
            if lit_sat[ci, pos]:
                # This literal is currently satisfied; flipping makes it unsatisfied
                if sat_counts[ci] == 1:
                    gain -= 1
            else:
                # This literal is currently unsatisfied; flipping makes it satisfied
                if sat_counts[ci] == 0:
                    gain += 1
        
        return gain

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    noise = 0.3
    base_flip_budget = 50000
    restart_count = 0
    
    # Phase saving: remember last successful assignment
    phase = None
    
    while time.time() < deadline:
        # Restart logic with phase saving
        if phase is None or random.random() < 0.3:
            assignment = initialize_assignment()
        else:
            # Perturb phase: flip some random variables
            assignment = phase.copy()
            num_flips = random.randint(1, max(1, n // 10))
            flip_vars = random.sample(range(n), min(num_flips, n))
            for v in flip_vars:
                assignment[v] = not assignment[v]
        
        lit_sat, sat_counts = compute_initial_state(assignment)
        
        unsat_clauses = np.where(sat_counts == 0)[0]
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            phase = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive flip budget
        flip_budget = base_flip_budget
        
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = current_unsat
        local_best_assignment = assignment.copy()
        
        while steps < flip_budget and time.time() < deadline:
            if current_unsat == 0:
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
                    gain = calculate_gain(var, lit_sat, sat_counts)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            assignment, lit_sat, sat_counts = flip_variable(var, assignment, lit_sat, sat_counts)
            
            # Update unsat_clauses incrementally: recompute which clauses are unsatisfied
            # This is still O(m) but we can optimize by tracking changes
            # For now, use vectorized check
            unsat_clauses = np.where(sat_counts == 0)[0]
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                phase = assignment.copy()
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                local_best_assignment = assignment.copy()
                no_improvement_steps = 0
            
            steps += 1
            
            # Adapt noise: decrease gradually, but increase if stuck
            if steps % 500 == 0:
                if no_improvement_steps > 1000:
                    noise = min(0.8, noise + 0.05)
                else:
                    noise = max(0.05, noise - 0.02)
        
        restart_count += 1
        # Decay noise back to baseline between restarts
        noise = max(0.15, noise * 0.8)

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
