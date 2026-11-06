"""Improved 3-SAT solver using optimized WalkSAT with incremental tracking, clause weights, and adaptive strategy."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Preprocess clauses: convert to 0-based
    processed_clauses = []
    for clause in clauses:
        processed = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause]
        processed_clauses.append(processed)
    
    # Build variable-to-clause mapping for efficient gain computation
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause literal arrays
    clause_vars = []
    clause_signs = []
    for clause in processed_clauses:
        vs = [v for v, s in clause]
        ss = [s for v, s in clause]
        clause_vars.append(vs)
        clause_signs.append(ss)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    
    def lit_satisfied(var, sign, assignment):
        if sign == 1:
            return assignment[var]
        else:
            return not assignment[var]
    
    def init_tracking(assignment):
        """Initialize clause satisfaction tracking arrays."""
        clause_sat_count = [0] * num_clauses
        unsat_list = []
        for ci in range(num_clauses):
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            count = 0
            for i in range(len(vs)):
                if lit_satisfied(vs[i], ss[i], assignment):
                    count += 1
            clause_sat_count[ci] = count
            if count == 0:
                unsat_list.append(ci)
        return clause_sat_count, len(unsat_list), unsat_list
    
    def flip_var(assignment, var, clause_sat_count, unsat_count, unsat_list):
        """Flip a variable and update tracking incrementally. Returns (new_unsat_count, updated_unsat_list)."""
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        for ci, sign in var_to_clauses[var]:
            if sign == 1:
                before_sat = old_val
                after_sat = new_val
            else:
                before_sat = not old_val
                after_sat = not new_val
            
            if before_sat and not after_sat:
                clause_sat_count[ci] -= 1
                if clause_sat_count[ci] == 0:
                    unsat_count += 1
                    unsat_list.append(ci)
            elif not before_sat and after_sat:
                clause_sat_count[ci] += 1
                if clause_sat_count[ci] == 1:
                    unsat_count -= 1
                    # Remove from unsat_list (lazy deletion handled by check)
        return unsat_count, unsat_list
    
    def get_gain_fast(assignment, var, clause_sat_count, clause_weights):
        """Compute weighted gain of flipping var."""
        gain = 0.0
        for ci, sign in var_to_clauses[var]:
            w = clause_weights[ci]
            sc = clause_sat_count[ci]
            if sc > 1:
                continue
            if sc == 1:
                if sign == 1 and assignment[var]:
                    gain -= w
                elif sign == -1 and not assignment[var]:
                    gain -= w
            else:  # sc == 0
                if sign == 1 and not assignment[var]:
                    gain += w
                elif sign == -1 and assignment[var]:
                    gain += w
        return gain
    
    def pick_unsat_clause(unsat_list, clause_sat_count, rng):
        """Pick an unsat clause, filtering out stale entries."""
        valid = [ci for ci in unsat_list if clause_sat_count[ci] == 0]
        if not valid:
            return None
        return rng.choice(valid)
    
    # WalkSAT with clause weights and adaptive gain probability
    max_restarts = 300
    clause_weights = [1.0] * num_clauses
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_sat_count, unsat_count, unsat_list = init_tracking(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive steps based on unsat count
        if unsat_count <= 5:
            steps_per_restart = 30000
        elif unsat_count <= 20:
            steps_per_restart = 15000
        elif unsat_count <= 50:
            steps_per_restart = 8000
        else:
            steps_per_restart = 4000
        
        # Adaptive gain probability: higher when unsat is low (more focused search)
        base_gain_prob = 0.9 if unsat_count <= 20 else 0.8
        
        for step in range(steps_per_restart):
            # Time check every 100 steps to reduce overhead
            if step % 100 == 0 and time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            if step % 50 == 0 and unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Pick an unsatisfied clause
            ci = pick_unsat_clause(unsat_list, clause_sat_count, rng)
            if ci is None:
                break
            
            clause = processed_clauses[ci]
            
            # Adaptive gain probability: increase randomness when stuck
            gain_prob = base_gain_prob
            if step > steps_per_restart * 0.5 and unsat_count >= best_unsat_count:
                gain_prob = max(0.5, gain_prob - 0.15)
            
            if rng.random() < gain_prob:
                # Gain-based selection with clause weights
                best_var = None
                best_gain = -float('inf')
                for var, sign in clause:
                    gain = get_gain_fast(assignment, var, clause_sat_count, clause_weights)
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                if best_var is not None:
                    unsat_count, unsat_list = flip_var(assignment, best_var, clause_sat_count, unsat_count, unsat_list)
            else:
                var, sign = rng.choice(clause)
                unsat_count, unsat_list = flip_var(assignment, var, clause_sat_count, unsat_count, unsat_list)
        
        # Update clause weights: increase weight of clauses that remain unsatisfied
        for ci in range(num_clauses):
            if clause_sat_count[ci] == 0:
                clause_weights[ci] = min(clause_weights[ci] * 1.1, 10.0)
            else:
                clause_weights[ci] = max(clause_weights[ci] * 0.95, 1.0)
        
        # Prune unsat_list occasionally to keep it manageable
        if len(unsat_list) > num_clauses * 2:
            unsat_list = [ci for ci in unsat_list if clause_sat_count[ci] == 0]
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
