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
    
    # Preprocess clauses into numpy arrays
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
    
    # Variable occurrence lists
    var_positions = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_positions[v].append((i, j))
    
    # Convert to numpy arrays
    var_clause_idx = []
    var_pos_in_clause = []
    for v in range(n):
        if var_positions[v]:
            arr = np.array(var_positions[v], dtype=np.int32)
            var_clause_idx.append(arr[:, 0])
            var_pos_in_clause.append(arr[:, 1])
        else:
            var_clause_idx.append(np.empty(0, dtype=np.int32))
            var_pos_in_clause.append(np.empty(0, dtype=np.int32))
    
    # Build clause-to-variables mapping for fast lookup
    # clause_vars[clause_idx] = list of (var, pos)
    clause_vars = []
    for i in range(m):
        clause_vars.append([(indices[i, j], j) for j in range(3)])
    
    # Precompute for each variable: which clauses it appears in and with what sign
    # var_clause_signs[var] = dict {clause_idx: sign}
    var_clause_signs = [dict() for _ in range(n)]
    for v in range(n):
        ci_arr = var_clause_idx[v]
        pos_arr = var_pos_in_clause[v]
        for k in range(len(ci_arr)):
            ci = int(ci_arr[k])
            p = int(pos_arr[k])
            var_clause_signs[v][ci] = int(signs[ci, p])
    
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def calculate_sat_counts(assignment):
        val = np.take(assignment, indices)
        is_sat = (val == (signs > 0))
        return np.sum(is_sat, axis=1)

    def flip_variable(var, assignment, sat_counts, unsat_list):
        """
        Flip variable var, update assignment, sat_counts, and unsat_list incrementally.
        Returns updated sat_counts and unsat_list.
        """
        ci_arr = var_clause_idx[var]
        
        if len(ci_arr) == 0:
            return sat_counts, unsat_list
        
        # Update assignment
        assignment[var] = not assignment[var]
        new_val = assignment[var]
        
        # Update sat counts for affected clauses
        for k in range(len(ci_arr)):
            ci = int(ci_arr[k])
            p = int(var_pos_in_clause[var][k])
            
            lit_desired = signs[ci, p] > 0
            old_sat = (assignment[var] == lit_desired)  # This is the NEW value after flip
            # Actually we need to be careful: we already flipped, so let's recompute
            # Before flip: assignment was not new_val
            # After flip: assignment is new_val
            
            # Recompute sat count for this clause
            val = np.take(assignment, indices[ci])
            is_sat = (val == (signs[ci] > 0))
            new_count = int(np.sum(is_sat))
            old_count = int(sat_counts[ci])
            
            sat_counts[ci] = new_count
            
            # Update unsat list
            if old_count == 0 and new_count > 0:
                # Was unsat, now sat - remove from unsat_list
                unsat_list.remove(ci)
            elif old_count > 0 and new_count == 0:
                # Was sat, now unsat - add to unsat_list
                unsat_list.append(ci)
        
        return sat_counts, unsat_list

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    base_noise = 0.3
    base_flip_budget = 100000
    restart_count = 0
    
    # Frequency tracking for variables in unsat clauses
    var_unsat_freq = np.zeros(n, dtype=np.int32)
    
    while time.time() < deadline:
        # Restart logic
        if restart_count == 0 or best_assignment is None:
            assignment = initialize_assignment()
        elif random.random() < 0.4:
            # Perturb best assignment
            assignment = best_assignment.copy()
            num_flips = max(1, n // 15)
            flip_vars = random.sample(range(n), num_flips)
            for v in flip_vars:
                assignment[v] = not assignment[v]
        else:
            assignment = initialize_assignment()
        
        sat_counts = calculate_sat_counts(assignment)
        
        # Build unsat list
        unsat_indices = np.where(sat_counts == 0)[0]
        unsat_list = unsat_indices.tolist()
        current_unsat = len(unsat_list)
        
        # Reset frequency tracking
        var_unsat_freq = np.zeros(n, dtype=np.int32)
        for ci in unsat_list:
            for j in range(3):
                v = int(indices[ci, j])
                var_unsat_freq[v] += 1
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive flip budget
        if restart_count == 0:
            flip_budget = base_flip_budget
        else:
            flip_budget = int(base_flip_budget * (0.7 + 0.6 * random.random()))
        
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = current_unsat
        noise = base_noise
        
        while steps < flip_budget and time.time() < deadline:
            if len(unsat_list) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_list[random.randint(0, len(unsat_list) - 1)]
            
            # Pick a literal to flip using frequency-based weighting
            if random.random() < noise:
                # Random flip from clause
                pos = random.randint(0, 2)
                var = int(indices[clause_idx, pos])
            else:
                # Use frequency-based selection with smooth surrogate
                # For each variable in the clause, compute a score
                best_score = -1e10
                best_var = -1
                
                for pos in range(3):
                    var = int(indices[clause_idx, pos])
                    
                    # Calculate gain for this variable
                    ci_arr = var_clause_idx[var]
                    pos_arr = var_pos_in_clause[var]
                    
                    if len(ci_arr) == 0:
                        gain = 0
                    else:
                        current_val = assignment[var]
                        gain = 0
                        
                        for k in range(len(ci_arr)):
                            ci = int(ci_arr[k])
                            p = int(pos_arr[k])
                            
                            lit_desired = signs[ci, p] > 0
                            old_sat = (current_val == lit_desired)
                            new_sat = (not current_val == lit_desired)
                            
                            if old_sat and not new_sat:
                                if sat_counts[ci] == 1:
                                    gain -= 1
                            elif not old_sat and new_sat:
                                if sat_counts[ci] == 0:
                                    gain += 1
                    
                    # Frequency-based penalty: prefer variables that appear less often in unsat clauses
                    # This helps escape flat regions by exploring less "committed" variables
                    freq = var_unsat_freq[var]
                    
                    # Smooth surrogate: use log-sum-exp style weighting
                    # Higher gain and lower frequency both increase score
                    # Use a soft-max-like approach
                    score = gain - 0.1 * np.log1p(freq)
                    
                    if score > best_score:
                        best_score = score
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = int(indices[clause_idx, 0])
                    
            # Perform flip
            sat_counts, unsat_list = flip_variable(var, assignment, sat_counts, unsat_list)
            
            # Update frequency tracking for flipped variable
            ci_arr = var_clause_idx[var]
            for k in range(len(ci_arr)):
                ci = int(ci_arr[k])
                if sat_counts[ci] == 0:
                    var_unsat_freq[var] += 1
                else:
                    if var_unsat_freq[var] > 0:
                        var_unsat_freq[var] -= 1
            
            current_unsat = len(unsat_list)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improvement_steps = 0
            
            steps += 1
            
            # Adapt noise: decrease gradually, but increase if stuck
            if steps % 50 == 0:
                if no_improvement_steps > 200:
                    noise = min(0.6, noise + 0.04)
                else:
                    noise = max(0.15, noise - 0.015)
        
        restart_count += 1

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
