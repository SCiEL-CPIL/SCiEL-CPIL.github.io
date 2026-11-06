"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 320.0  # Leave 10s margin before 330s kill
    
    m = len(clauses)
    
    # Precompute clause structure as flat arrays for speed
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else 0
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_to_clauses[v].append((i, j))
    
    # Precompute clause positions per variable for faster greedy scoring
    var_clause_pos = [[] for _ in range(n)]
    for v in range(n):
        var_clause_pos[v] = [(ci, j) for ci, j in var_to_clauses[v]]
    
    # Phase saving: for each variable, count how many clauses it appears in positive vs negative
    var_pos_count = [0] * n
    var_neg_count = [0] * n
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            if clause_signs[i, j] == 1:
                var_pos_count[v] += 1
            else:
                var_neg_count[v] += 1
    
    # State tracking using numpy for vectorized operations
    def init_state(assignment):
        """Initialize clause satisfaction count from boolean assignment."""
        assign_arr = np.array(assignment, dtype=np.int8)
        var_vals = assign_arr[clause_vars]  # shape (m, 3)
        sat = (var_vals == clause_signs).astype(np.int32)
        return sat.sum(axis=1)
    
    def flip_var(v, assignment, clause_sat_count, unsat_list):
        """Flip variable v and update clause_sat_count and unsat_list in place."""
        new_val = not assignment[v]
        assignment[v] = new_val
        new_val_int = 1 if new_val else 0
        for ci, j in var_to_clauses[v]:
            old_sat = clause_sat_count[ci]
            if new_val_int == int(clause_signs[ci, j]):
                clause_sat_count[ci] = old_sat + 1
            else:
                clause_sat_count[ci] = old_sat - 1
            new_sat = clause_sat_count[ci]
            # Update unsat_list
            if old_sat == 0 and new_sat > 0:
                # Remove from unsat list (mark as -1, we'll compact later)
                pass  # We'll rebuild unsat_list periodically
            elif old_sat > 0 and new_sat == 0:
                pass  # Add to unsat list
        # We track unsat count separately and rebuild list when needed
        return new_val_int
    
    def build_unsat_list(clause_sat_count):
        """Build a list of unsatisfied clause indices."""
        return [i for i in range(m) if clause_sat_count[i] == 0]
    
    # WalkSAT with adaptive parameters and restarts
    best_assignment = None
    best_unsat_count = m + 1
    best_clause_sat = None
    
    rng = random.Random(42)
    
    # Adaptive parameters
    base_noise = 0.25
    base_max_flips = 15000
    
    restart_count = 0
    
    while time.time() - start_time < timeout:
        # Choose restart strategy: random or perturb best
        if best_assignment is not None and restart_count > 0 and rng.random() < 0.4:
            # Perturb best assignment: flip a few random variables
            assignment = best_assignment.copy()
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
        else:
            # Random restart with phase saving bias
            assignment = []
            for v in range(n):
                if var_pos_count[v] > var_neg_count[v]:
                    assignment.append(rng.random() < 0.6)
                elif var_neg_count[v] > var_pos_count[v]:
                    assignment.append(rng.random() < 0.4)
                else:
                    assignment.append(rng.random() < 0.5)
        
        clause_sat_count = init_state(assignment)
        unsat_list = build_unsat_list(clause_sat_count)
        current_unsat = len(unsat_list)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            best_clause_sat = clause_sat_count.copy()
        
        max_flips = base_max_flips
        noise = base_noise
        
        # Track unsat list for fast random selection
        unsat_dirty = False
        unsat_count = current_unsat
        
        for flip_num in range(max_flips):
            if time.time() - start_time > timeout - 5.0:
                break
            
            if unsat_count == 0:
                return assignment
            
            # Rebuild unsat list if dirty or periodically
            if unsat_dirty or flip_num % 100 == 0:
                unsat_list = build_unsat_list(clause_sat_count)
                unsat_count = len(unsat_list)
                unsat_dirty = False
                if unsat_count == 0:
                    return assignment
            
            # Pick a random unsatisfied clause
            if unsat_count > 0:
                ci = unsat_list[rng.randint(0, unsat_count - 1)]
            else:
                break
            
            # Adaptive noise: decrease as we get closer to solution
            adaptive_noise = noise * (1.0 + 0.5 * (unsat_count / m))
            adaptive_noise = min(adaptive_noise, 0.4)
            
            # WalkSAT heuristic
            if rng.random() < adaptive_noise:
                # Random pick among the 3 vars
                j = rng.randint(0, 2)
                v = int(clause_vars[ci, j])
            else:
                # Greedy: pick var that satisfies most other unsat clauses
                best_var = -1
                best_score = -1
                for j in range(3):
                    v = int(clause_vars[ci, j])
                    score = 0
                    for other_ci, _ in var_to_clauses[v]:
                        if other_ci != ci and clause_sat_count[other_ci] == 0:
                            score += 1
                    # Tie-break with phase saving
                    if score == best_score:
                        if best_var == -1:
                            best_var = v
                        else:
                            # Prefer the polarity that satisfies more clauses overall
                            v_pref = 1 if var_pos_count[v] >= var_neg_count[v] else 0
                            bv_pref = 1 if var_pos_count[best_var] >= var_neg_count[best_var] else 0
                            if v_pref != bv_pref:
                                if v_pref == 1 and assignment[v] == False:
                                    best_var = v
                                elif bv_pref == 1 and assignment[best_var] == False:
                                    pass
                    if score > best_score:
                        best_score = score
                        best_var = v
                v = best_var
            
            if v >= 0:
                flip_var(v, assignment, clause_sat_count, unsat_list)
                unsat_dirty = True
            
            # Periodically rebuild and check
            if flip_num % 50 == 0:
                unsat_list = build_unsat_list(clause_sat_count)
                unsat_count = len(unsat_list)
                unsat_dirty = False
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment.copy()
                    best_clause_sat = clause_sat_count.copy()
        
        restart_count += 1
        
        # Final check after restart
        if best_assignment is not None:
            final_check = init_state(best_assignment)
            if np.all(final_check > 0):
                return best_assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
