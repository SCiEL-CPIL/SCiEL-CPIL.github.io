"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Time budget guard: leave a safety margin before the 330s wall-clock kill.
    deadline = time.time() + 320.0
    
    # Preprocess: convert clauses to 0-based signed indices for speed.
    # literal l -> index abs(l)-1, positive flag (l > 0)
    # Store as list of (var_index, is_positive)
    pre_clauses = []
    for c in clauses:
        pre_c = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pre_c.append((var_idx, is_pos))
        pre_clauses.append(pre_c)
    
    m = len(pre_clauses)
    
    def is_satisfied_clause(cl, assign):
        for var_idx, is_pos in cl:
            if assign[var_idx] == is_pos:
                return True
        return False
    
    def count_unsatisfied(cl, assign):
        cnt = 0
        for var_idx, is_pos in cl:
            if assign[var_idx] != is_pos:
                cnt += 1
        return cnt
    
    def check_all(assign):
        for cl in pre_clauses:
            if not is_satisfied_clause(cl, assign):
                return False
        return True
    
    # WalkSAT with clause weighting and multiple restarts
    # Use a hybrid approach: WalkSAT with restarts and occasional DPLL-like unit propagation
    
    rng = random.Random(0)
    
    best_assign = None
    best_unsat_count = m + 1
    
    # Try multiple strategies/restarts within the time budget
    while time.time() < deadline:
        # Initialize assignment
        assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Check if already satisfied
        if check_all(assign):
            return assign
        
        # WalkSAT with adaptive parameters
        # Use a combination of random flips and weighted clause selection
        weight = [1.0] * m  # clause weights
        p = 0.55  # probability of random flip vs. most-frequent variable
        
        # Track unsatisfied count efficiently
        unsat_clauses = [i for i in range(m) if not is_satisfied_clause(pre_clauses[i], assign)]
        
        if not unsat_clauses:
            return assign
        
        # WalkSAT loop
        max_steps = 50000
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            
            if not unsat_clauses:
                return assign
            
            # Update best
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assign = assign[:]
            
            # Select a clause to focus on, weighted by weight
            # Use weighted random selection
            total_weight = sum(weight[i] for i in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0
            selected_cl_idx = unsat_clauses[0]
            for i in unsat_clauses:
                cumulative += weight[i]
                if r <= cumulative:
                    selected_cl_idx = i
                    break
            
            cl = pre_clauses[selected_cl_idx]
            
            # Decide which variable to flip
            if rng.random() < p:
                # Random flip among variables in the clause
                var_idx, is_pos = rng.choice(cl)
            else:
                # Flip the variable that appears most often in unsatisfied clauses
                # Count occurrences in unsat_clauses
                var_counts = {}
                for uc_idx in unsat_clauses:
                    uc = pre_clauses[uc_idx]
                    for var_idx2, _ in uc:
                        var_counts[var_idx2] = var_counts.get(var_idx2, 0) + 1
                
                # Find variable in the selected clause with highest count
                best_var = None
                best_count = -1
                for var_idx2, _ in cl:
                    cnt = var_counts.get(var_idx2, 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_var = var_idx2
                
                var_idx = best_var
            
            # Flip the variable
            old_val = assign[var_idx]
            assign[var_idx] = not old_val
            
            # Update unsat_clauses: only clauses containing var_idx need updating
            new_unsat = []
            for i in unsat_clauses:
                cl_i = pre_clauses[i]
                # Check if this clause contains var_idx
                contains_var = False
                for var_idx2, is_pos2 in cl_i:
                    if var_idx2 == var_idx:
                        contains_var = True
                        break
                if contains_var:
                    if not is_satisfied_clause(cl_i, assign):
                        new_unsat.append(i)
                    # else: now satisfied, remove
                else:
                    new_unsat.append(i)
            
            # Also need to check if flipping made any previously satisfied clause unsatisfied
            # This is expensive; for simplicity, rebuild unsat_clauses periodically
            if step % 100 == 0:
                unsat_clauses = [i for i in range(m) if not is_satisfied_clause(pre_clauses[i], assign)]
            else:
                unsat_clauses = new_unsat
            
            # Occasionally adjust weights
            if step % 1000 == 0:
                # Decay weights and boost unsatisfied clauses
                for i in range(m):
                    weight[i] *= 0.95
                for i in unsat_clauses:
                    weight[i] += 1.0
        
        # If we exit the inner loop without finding a solution, restart
        # Decay p slightly for next restart to increase exploration
        p = max(0.3, p - 0.01)
    
    # Return the best assignment found
    if best_assign is not None:
        return best_assign
    
    # Fallback: return current assignment
    return assign
# EVOLVE-BLOCK-END
