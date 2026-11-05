"""Improved planted 3-SAT solver with WalkSAT + clause weighting + restarts + deadline guard."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: use 280s out of 330s to be safe
    deadline = time.time() + 280.0
    
    # Precompute clause literal indices and signs for speed
    clause_idx = np.array([[abs(l) - 1 for l in c] for c in clauses], dtype=np.int32)
    clause_sign = np.array([[(l > 0) for l in c] for c in clauses], dtype=bool)
    
    # Precompute variable-to-clause index map
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(i)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Multiple restarts with increasing step budgets
    restart_budgets = [3000, 7000, 15000, 30000, 60000, 120000, 250000]
    rng_seed = 42
    
    for restart in range(len(restart_budgets)):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(rng_seed + restart * 7919)
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Initialize clause weights for this restart
        w = np.ones(m, dtype=np.float64)
        
        step_budget = restart_budgets[restart]
        
        for step in range(step_budget):
            if time.time() > deadline - 2.0:
                break
            
            # Vectorized clause satisfaction check
            # For each clause, check if any literal is satisfied
            # assignment[clause_idx] gives us the truth values of all literals
            # We need to check if assignment matches the required sign
            lit_vals = assignment[clause_idx]  # shape (m, 3)
            # A literal is satisfied if (assignment == sign) for positive, or (not assignment == not sign) for negative
            # Actually: literal l is satisfied if assignment[var] == (l > 0)
            # So: satisfied = lit_vals == clause_sign
            sat_lits = (lit_vals == clause_sign)  # shape (m, 3), True if literal is satisfied
            clause_sat = sat_lits.any(axis=1)  # shape (m,)
            
            unsat_mask = ~clause_sat
            unsat_indices = np.where(unsat_mask)[0]
            
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            cur_unsat = len(unsat_indices)
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment.tolist()
            
            # Select clause weighted - use numpy for efficiency
            unsat_weights = w[unsat_indices]
            total_w = unsat_weights.sum()
            if total_w <= 0:
                total_w = 1.0
                unsat_weights = np.ones(len(unsat_indices))
            
            # Weighted random selection
            r = rng.random() * total_w
            cum = 0.0
            chosen_idx = 0
            for idx, wt in enumerate(unsat_weights):
                cum += wt
                if r <= cum:
                    chosen_idx = idx
                    break
            
            ci = unsat_indices[chosen_idx]
            
            # Count how many literals in chosen clause are satisfied
            sat_count = sat_lits[ci].sum()
            
            # Adaptive flip probability: higher when more unsat clauses
            # When close to solution, be more aggressive about fixing unsat literals
            if cur_unsat <= 5:
                p = 0.85
            elif cur_unsat <= 20:
                p = 0.80
            elif cur_unsat <= 50:
                p = 0.75
            else:
                p = 0.70
            
            if sat_count < 3:
                # There are unsatisfied literals
                unsat_lit_positions = []
                sat_lit_positions = []
                for j in range(3):
                    if sat_lits[ci][j]:
                        sat_lit_positions.append(j)
                    else:
                        unsat_lit_positions.append(j)
                
                if unsat_lit_positions and (rng.random() < p or not sat_lit_positions):
                    j = unsat_lit_positions[rng.randint(0, len(unsat_lit_positions) - 1)]
                else:
                    j = sat_lit_positions[rng.randint(0, len(sat_lit_positions) - 1)] if sat_lit_positions else unsat_lit_positions[0]
            else:
                j = rng.randint(0, 2)
            
            var = int(clause_idx[ci][j])
            old_val = assignment[var]
            assignment[var] = not old_val
            
            # Update weights: increase weight of clauses that became unsatisfied due to this flip
            # Only check clauses affected by this variable flip
            for ci2 in var_to_clauses[var]:
                # Check if clause ci2 is now unsatisfied
                c_sat = False
                for jj in range(3):
                    if assignment[int(clause_idx[ci2][jj])] == clause_sign[ci2][jj]:
                        c_sat = True
                        break
                if not c_sat:
                    w[ci2] *= 1.05
                else:
                    w[ci2] = max(1.0, w[ci2] * 0.98)
        
        # After each restart, if we have a good solution, do a polish pass
        if best_assignment is not None and best_unsat_count <= 10:
            # Polish pass: try to fix remaining unsat clauses
            polish_assignment = np.array(best_assignment, dtype=bool)
            polish_steps = min(5000, max(500, best_unsat_count * 100))
            for pstep in range(polish_steps):
                if time.time() > deadline - 2.0:
                    break
                
                lit_vals = polish_assignment[clause_idx]
                sat_lits = (lit_vals == clause_sign)
                clause_sat = sat_lits.any(axis=1)
                unsat_indices = np.where(~clause_sat)[0]
                
                if len(unsat_indices) == 0:
                    return polish_assignment.tolist()
                
                # Try to fix one unsat clause
                ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
                sat_count = sat_lits[ci].sum()
                
                if sat_count < 3:
                    unsat_lit_positions = [j for j in range(3) if not sat_lits[ci][j]]
                    # Flip an unsatisfied literal to try to satisfy this clause
                    j = unsat_lit_positions[rng.randint(0, len(unsat_lit_positions) - 1)]
                    var = int(clause_idx[ci][j])
                    polish_assignment[var] = not polish_assignment[var]
                    
                    # Check if this improved things
                    new_lit_vals = polish_assignment[clause_idx]
                    new_sat_lits = (new_lit_vals == clause_sign)
                    new_clause_sat = new_sat_lits.any(axis=1)
                    new_unsat = (~new_clause_sat).sum()
                    
                    # If it made things worse, revert
                    if new_unsat > best_unsat_count:
                        polish_assignment[var] = not polish_assignment[var]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
