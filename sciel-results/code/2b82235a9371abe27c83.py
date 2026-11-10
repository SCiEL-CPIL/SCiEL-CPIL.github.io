import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute clause literal arrays
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8) # 1 for pos, -1 for neg
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    # Build M_pos and M_neg: (n, m)
    # M_pos[v, c] = 1 if clause c contains +v
    # M_neg[v, c] = 1 if clause c contains -v
    M_pos = np.zeros((n, m), dtype=np.float32)
    M_neg = np.zeros((n, m), dtype=np.float32)
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                M_pos[v, i] = 1
            else:
                M_neg[v, i] = 1

    rng = np.random.RandomState(42)
    assignment = rng.rand(n) < 0.5

    # Compute initial clause_sat
    # lit_sat[i, j] = (assignment[var] == (sign==1))
    a = assignment[clause_vars] # (m, 3)
    s = (clause_signs == 1) # (m, 3)
    lit_sat = (a == s)
    clause_sat = lit_sat.any(axis=1)
    
    num_unsat = np.sum(~clause_sat)

    if num_unsat == 0:
        return assignment.tolist()

    max_iters = 100000
    p_walk = 0.5 # Probability to do a random walk step when best_gain <= 0
    
    for _ in range(max_iters):
        if num_unsat == 0:
            break
        
        unsat_mask = ~clause_sat
        sat_mask = clause_sat
        
        # Compute gains
        # gain_true[v] = unsat clauses with +v - sat clauses with -v
        # gain_false[v] = unsat clauses with -v - sat clauses with +v
        unsat_counts_pos = M_pos @ unsat_mask
        unsat_counts_neg = M_neg @ unsat_mask
        sat_counts_pos = M_pos @ sat_mask
        sat_counts_neg = M_neg @ sat_mask
        
        gain_true = unsat_counts_pos - sat_counts_neg
        gain_false = unsat_counts_neg - sat_counts_pos
        
        best_gain = -np.inf
        best_var = -1
        best_val = False
        
        max_gain_true = np.max(gain_true)
        max_gain_false = np.max(gain_false)
        
        if max_gain_true >= max_gain_false:
            best_gain = max_gain_true
            candidates = np.where(gain_true == max_gain_true)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = True
        else:
            best_gain = max_gain_false
            candidates = np.where(gain_false == max_gain_false)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = False

        # Escape mechanism: If best_gain <= 0, we are not improving.
        # With probability p_walk, do a standard WalkSAT step.
        if best_gain <= 0 and rng.rand() < p_walk:
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                break
            ci = unsat_indices[rng.randint(len(unsat_indices))]
            # Pick a random variable in this clause
            j = rng.randint(3)
            v = int(clause_vars[ci, j])
            best_var = v
            # With prob 0.5, set to value that satisfies clause, else random
            if rng.rand() < 0.5:
                best_val = (clause_signs[ci, j] == 1)
            else:
                best_val = not assignment[v]

        # Apply flip
        if best_val != assignment[best_var]:
            assignment[best_var] = best_val
            
            # Update clause_sat for clauses containing best_var
            # Find affected clauses
            affected_clauses = np.where((M_pos[best_var, :] == 1) | (M_neg[best_var, :] == 1))[0]
            
            if len(affected_clauses) > 0:
                # Vectorized update of clause_sat for affected clauses
                aff_vars = clause_vars[affected_clauses]
                aff_signs = clause_signs[affected_clauses]
                
                # Get current assignment for these variables
                aff_assign = assignment[aff_vars]
                
                # Check satisfaction for each literal
                # lit_sat is True if assignment matches sign
                # sign is 1 for True, -1 for False. 
                # assignment is bool. 
                # We need (assignment == (sign == 1))
                aff_s = (aff_signs == 1)
                aff_lit_sat = (aff_assign == aff_s)
                
                new_clause_sat = aff_lit_sat.any(axis=1)
                
                # Update global clause_sat
                old_clause_sat = clause_sat[affected_clauses]
                clause_sat[affected_clauses] = new_clause_sat
                
                # Update num_unsat
                num_unsat += np.sum(~new_clause_sat) - np.sum(~old_clause_sat)

    return assignment.tolist()
