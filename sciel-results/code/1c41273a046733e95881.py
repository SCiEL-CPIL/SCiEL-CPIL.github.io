"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # Safety margin before 300s kill
    m = len(clauses)
    
    # Build clause matrix: shape (m, 3), each entry is a signed literal (1-based)
    # For efficiency, we'll use arrays
    clause_lits = np.array(clauses, dtype=np.int32)  # shape (m, 3)
    
    # Precompute variable-to-clause mapping for fast updates
    # var_to_clauses[v] = list of (clause_idx, literal_sign) where literal_sign is +1 or -1
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for lit in clause_lits[ci]:
            v = abs(lit) - 1
            if 0 <= v < n:
                var_to_clauses[v].append((ci, 1 if lit > 0 else -1))
    
    # Convert to numpy arrays for faster access
    var_clause_indices = [np.array([c[0] for c in lst], dtype=np.int32) for lst in var_to_clauses]
    var_clause_signs = [np.array([c[1] for c in lst], dtype=np.int32) for lst in var_to_clauses]
    
    def init_state(assignment):
        """Initialize clause satisfaction counts and per-variable impact scores."""
        # clause_sat_count[ci] = number of true literals in clause ci
        clause_sat_count = np.zeros(m, dtype=np.int32)
        
        for ci in range(m):
            for lit in clause_lits[ci]:
                v = abs(lit) - 1
                if 0 <= v < n:
                    lit_sign = 1 if lit > 0 else -1
                    if assignment[v] == (lit_sign == 1):
                        clause_sat_count[ci] += 1
        
        return clause_sat_count
    
    def update_state(assignment, clause_sat_count, var, new_val):
        """Update clause_sat_count after flipping variable var to new_val."""
        old_val = not new_val
        var_clause_idx = var_clause_indices[var]
        var_clause_sig = var_clause_signs[var]
        
        for i in range(len(var_clause_idx)):
            ci = var_clause_idx[i]
            sig = var_clause_sig[i]
            # Old contribution: was this literal true under old_val?
            old_true = (old_val == (sig == 1))
            # New contribution: is this literal true under new_val?
            new_true = (new_val == (sig == 1))
            
            if new_true and not old_true:
                clause_sat_count[ci] += 1
            elif old_true and not new_true:
                clause_sat_count[ci] -= 1
        
        return clause_sat_count
    
    def compute_impact(assignment, clause_sat_count, var):
        """
        Compute the impact of flipping variable var.
        Returns the net change in unsatisfied clause count.
        Negative = improvement (fewer unsatisfied clauses).
        """
        old_val = assignment[var]
        new_val = not old_val
        var_clause_idx = var_clause_indices[var]
        var_clause_sig = var_clause_signs[var]
        
        net_change = 0
        for i in range(len(var_clause_idx)):
            ci = var_clause_idx[i]
            sig = var_clause_sig[i]
            old_true = (old_val == (sig == 1))
            new_true = (new_val == (sig == 1))
            
            if new_true and not old_true:
                # This literal becomes true. Clause ci gains one true literal.
                # If clause was unsatisfied (sat_count == 0), it becomes satisfied: gain 1
                # If clause was already satisfied, no change in unsat count
                if clause_sat_count[ci] == 0:
                    net_change -= 1  # improvement
            elif old_true and not new_true:
                # This literal becomes false. Clause ci loses one true literal.
                # If clause had exactly 1 true literal, it becomes unsatisfied: loss 1
                # If clause had more, no change in unsat count
                if clause_sat_count[ci] == 1:
                    net_change += 1  # worsening
        
        return net_change
    
    def solve_one_restart(rng, max_flips, noise_prob):
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Initialize satisfaction state
        clause_sat_count = init_state(assignment)
        unsat_mask = clause_sat_count == 0
        unsat_count = int(np.sum(unsat_mask))
        
        if unsat_count == 0:
            return assignment, True
        
        for step in range(max_flips):
            if time.time() > deadline:
                return assignment, False
            
            if unsat_count == 0:
                return assignment, True
            
            # Pick a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment, True
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clause_lits[ci]
            
            # Calculate impact for each literal in the clause
            best_impact = 1e9
            best_var = -1
            candidates = []
            
            for lit in clause:
                v = abs(lit) - 1
                if v < 0 or v >= n:
                    continue
                
                impact = compute_impact(assignment, clause_sat_count, v)
                candidates.append((impact, v))
                
                if impact < best_impact:
                    best_impact = impact
                    best_var = v
            
            if not candidates:
                # Fallback: pick random literal from clause
                lit = rng.choice(clause)
                v = abs(lit) - 1
            else:
                # With probability noise_prob, pick random candidate
                # Otherwise, pick best impact (with random tie-breaking)
                if rng.random() < noise_prob:
                    _, v = rng.choice(candidates)
                else:
                    min_impact = best_impact
                    tied = [c for c in candidates if c[0] == min_impact]
                    _, v = rng.choice(tied)
            
            # Flip the variable
            new_val = not assignment[v]
            assignment[v] = new_val
            
            # Update satisfaction state
            clause_sat_count = update_state(assignment, clause_sat_count, v, new_val)
            
            # Update unsat mask and count
            old_unsat_mask = unsat_mask
            unsat_mask = clause_sat_count == 0
            
            # Efficiently update unsat_count
            # Clauses that were unsat and now satisfied: gain
            # Clauses that were sat and now unsat: loss
            newly_satisfied = np.sum(old_unsat_mask & ~unsat_mask)
            newly_unsatisfied = np.sum(~old_unsat_mask & unsat_mask)
            unsat_count += newly_unsatisfied - newly_satisfied
            
            if unsat_count == 0:
                return assignment, True
        
        return assignment, (unsat_count == 0)
    
    rng = random.Random(0)
    best_assign = None
    best_unsat = float('inf')
    
    restarts = 0
    base_flips = 500
    max_restarts = 10000
    
    while time.time() < deadline:
        restarts += 1
        cur_flips = base_flips + int(restarts * 10)
        if cur_flips > 10000:
            cur_flips = 10000
        
        noise = 0.1 + 0.05 * math.sin(restarts * 0.1)
        
        assignment, solved = solve_one_restart(rng, cur_flips, noise)
        
        if solved:
            return assignment.tolist()
        
        # Track best unsat count for fallback
        clause_sat_count = init_state(assignment)
        unsat_cnt = int(np.sum(clause_sat_count == 0))
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assign = assignment.copy()
        
    if best_assign is not None:
        return best_assign.tolist()
    
    # Fallback random
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
