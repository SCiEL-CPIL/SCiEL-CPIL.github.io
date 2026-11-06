"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 320.0  # Leave margin for overhead (evaluator kills at 330s)
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Preprocess clauses into numpy arrays for fast evaluation
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Precompute sign_bool for faster satisfaction check
    sign_bool = (clause_data == 1)  # (m, 3) bool array
    
    # Build variable-to-clauses mapping for incremental updates
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    def compute_satisfaction(assignment):
        """Compute boolean array of satisfied clauses from scratch."""
        vals = assignment[clause_vars]  # (m, 3) bools
        literal_true = (vals == sign_bool)  # (m, 3)
        return np.any(literal_true, axis=1)  # (m,) bools
    
    def walk_sat(restart_assignment, max_flips, seed, phase_save=None):
        """Run WalkSAT with incremental clause tracking and phase saving."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        # Initialize satisfaction state
        sat = compute_satisfaction(assignment)
        unsat_count = int(np.sum(~sat))
        
        if unsat_count == 0:
            return assignment.tolist()
        
        # Track which clauses are unsatisfied for quick access
        unsat_set = set(np.where(~sat)[0].tolist())
        
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        for step in range(max_flips):
            if time.time() - start_time > timeout - 2:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # Track best state
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            if unsat_set:
                clause_idx = list(unsat_set)[rng.randint(0, len(unsat_set) - 1)]
            else:
                return assignment.tolist()
            
            # Choose variable to flip
            # Weighted random: probability proportional to # unsat clauses containing var
            # This acts as a smooth greedy heuristic
            clause_v = clause_vars[clause_idx]
            weights = []
            for v in clause_v:
                w = 0
                for ci in var_clauses[v]:
                    if ci in unsat_set:
                        w += 1
                # Add 1 to avoid zero weight (though w>=1 for vars in unsat clause)
                weights.append(w)
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            var_to_flip = clause_v[0]
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    var_to_flip = clause_v[i]
                    break
            
            # Flip the variable
            old_val = assignment[var_to_flip]
            new_val = not old_val
            assignment[var_to_flip] = new_val
            
            # Incrementally update satisfaction for affected clauses
            for ci in var_clauses[var_to_flip]:
                was_sat = sat[ci]
                v_vals = assignment[clause_vars[ci]]
                is_now_sat = bool(np.any(v_vals == sign_bool[ci]))
                sat[ci] = is_now_sat
                
                if was_sat and not is_now_sat:
                    unsat_count += 1
                    unsat_set.add(ci)
                elif not was_sat and is_now_sat:
                    unsat_count -= 1
                    unsat_set.discard(ci)
        
        # Final check
        if unsat_count == 0:
            return assignment.tolist()
        
        return None
    
    # Main search loop with restarts and phase saving
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    phase_assignment = None
    
    # Initial phase: try a few quick random walks to find a good starting point
    for _ in range(3):
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        sat = compute_satisfaction(initial)
        unsat = int(np.sum(~sat))
        if phase_assignment is None or unsat < int(np.sum(~compute_satisfaction(phase_assignment))):
            phase_assignment = initial
    
    while time.time() - start_time < timeout - 10:
        restart_count += 1
        
        # Use phase assignment with some perturbation for better restarts
        if phase_assignment is not None and restart_count > 1:
            # Perturb the phase assignment slightly
            initial = phase_assignment.copy()
            num_flips = max(1, n // 10)
            flip_indices = rng.sample(range(n), min(num_flips, n))
            for idx in flip_indices:
                initial[idx] = not initial[idx]
        else:
            initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Adaptive flip budget: increase for later restarts
        base_flips = 8000
        adaptive_factor = 1.0 + 0.3 * min(restart_count, 10)
        max_flips = int(base_flips * adaptive_factor)
        
        result = walk_sat(initial, max_flips, seed=restart_count * 12345 + 67890)
        
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # If we didn't find a solution, return the best assignment we found
    if phase_assignment is not None:
        return phase_assignment.tolist()
    
    # Fallback: return a random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
