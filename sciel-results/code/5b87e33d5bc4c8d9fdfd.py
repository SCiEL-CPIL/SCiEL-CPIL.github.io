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
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
            
    # Precompute variable occurrence lists and positions for fast gain calculation
    # var_positions[var] = list of (clause_idx, position_in_clause)
    var_positions = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_positions[v].append((i, j))

    # Helper to initialize assignment
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    # Helper to calculate initial sat counts
    def calculate_sat_counts(assignment):
        val = np.take(assignment, indices)
        is_sat = (val == (signs > 0))
        return np.sum(is_sat, axis=1)

    # Main search loop
    best_assignment = None
    best_unsat = m + 1
    
    # Parameters
    noise_start = 0.3
    noise_min = 0.1
    noise_max = 0.7
    base_flip_budget = 20000  # Increased budget per restart
    restart_count = 0
    
    while time.time() < deadline:
        # Restart logic
        assignment = initialize_assignment()
        sat_counts = calculate_sat_counts(assignment)
        
        # Initialize unsat list
        unsat_clauses = np.where(sat_counts == 0)[0]
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive parameters for this restart
        noise = noise_start
        flip_budget = base_flip_budget
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = current_unsat
        
        while steps < flip_budget and time.time() < deadline:
            if len(unsat_clauses) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            # Using random.choice on a numpy array is slow, so we pick an index
            u_idx = random.randint(0, len(unsat_clauses) - 1)
            clause_idx = unsat_clauses[u_idx]
            
            # Determine variable to flip
            if random.random() < noise:
                # Random flip from the clause
                pos = random.randint(0, 2)
                var = indices[clause_idx, pos]
            else:
                # WalkSAT heuristic: pick variable with best gain
                best_gain = -100
                best_var = -1
                
                # Check the 3 variables in the clause
                for pos in range(3):
                    var = indices[clause_idx, pos]
                    
                    # Calculate gain for flipping this var
                    gain = 0
                    current_val = assignment[var]
                    new_val = not current_val
                    
                    # Iterate over all clauses containing this variable
                    for ci, p in var_positions[var]:
                        # Determine if this literal was satisfied before and after flip
                        desired = signs[ci, p] > 0
                        old_sat = (current_val == desired)
                        new_sat = (new_val == desired)
                        
                        if old_sat and not new_sat:
                            # Literal becomes unsatisfied. 
                            # Clause becomes unsat only if it was satisfied only by this literal.
                            if sat_counts[ci] == 1:
                                gain -= 1
                        elif not old_sat and new_sat:
                            # Literal becomes satisfied.
                            # Clause becomes sat only if it was unsat (count 0).
                            if sat_counts[ci] == 0:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            assignment[var] = not assignment[var]
            
            # Update sat_counts and unsat_clauses incrementally
            # We only need to update clauses containing 'var'
            old_sat_counts_for_var = []
            affected_clauses = []
            
            for ci, p in var_positions[var]:
                # Recompute sat count for this specific clause
                # This is fast as it only touches 3 elements
                val = np.take(assignment, indices[ci])
                is_sat = (val == (signs[ci] > 0))
                new_count = np.sum(is_sat)
                
                old_count = sat_counts[ci]
                sat_counts[ci] = new_count
                old_sat_counts_for_var.append((ci, old_count, new_count))
                affected_clauses.append(ci)
                
            # Update unsat_clauses list
            # This is the tricky part. We need to add clauses that became unsat (count 0)
            # and remove clauses that became sat (count > 0).
            
            # Since unsat_clauses is a numpy array, modifying it in place is hard.
            # We can use a set or a list for unsat clauses for O(1) add/remove, 
            # but converting back to array for random selection is O(m).
            # Alternative: Keep unsat_clauses as a list of indices.
            # But we started with a numpy array. Let's switch to a list for unsat_clauses 
            # to allow efficient updates, or just rebuild it if the number of changes is small?
            # Rebuilding is O(m). m is ~450. This is acceptable if we do it every step? 
            # 20000 steps * 450 ops = 9M ops. That's fast in C/NumPy but slow in Python loop.
            # Let's try to maintain a list of unsat clause indices.
            
            # To optimize, we'll convert unsat_clauses to a list once at the start of the restart?
            # No, we already have it as a numpy array.
            # Let's just rebuild the unsat array. It's simple and robust.
            # Optimization: Only rebuild if the change is significant? No, just rebuild.
            # Actually, we can update the list in place if we maintain it as a Python list.
            
            # Let's refactor to use a Python list for unsat_clauses for O(1) updates.
            # We need to change the initialization and the loop logic slightly.
            
            # For now, let's stick to rebuilding the numpy array because it's simpler 
            # and m is small enough that `np.where` is very fast (C loop).
            unsat_clauses = np.where(sat_counts == 0)[0]
            
            current_unsat = len(unsat_clauses)
            
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
            
            # Adapt noise
            # Decrease noise slowly, increase if stuck
            if steps % 500 == 0:
                if no_improvement_steps > 1000:
                    noise = min(noise_max, noise + 0.05)
                else:
                    noise = max(noise_min, noise - 0.01)
                    
            # Early restart if stuck too long
            if no_improvement_steps > 5000:
                break
                
        restart_count += 1
        # If we are close to deadline, do not start a new restart
        if time.time() > deadline - 10.0:
            break

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
