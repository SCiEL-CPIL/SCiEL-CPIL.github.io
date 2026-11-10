import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

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
    a = assignment[clause_vars] # (m, 3)
    s = (clause_signs == 1) # (m, 3)
    lit_sat = (a == s)
    clause_sat = lit_sat.any(axis=1)
    
    num_unsat = m - np.sum(clause_sat)
    
    # Initialize impact scores
    unsat_counts_pos = np.zeros(n, dtype=np.int32)
    unsat_counts_neg = np.zeros(n, dtype=np.int32)
    sat_counts_pos = np.zeros(n, dtype=np.int32)
    sat_counts_neg = np.zeros(n, dtype=np.int32)
    
    unsat_mask = ~clause_sat
    sat_mask = clause_sat
    
    unsat_counts_pos = (M_pos * unsat_mask).sum(axis=1).astype(np.int32)
    unsat_counts_neg = (M_neg * unsat_mask).sum(axis=1).astype(np.int32)
    sat_counts_pos = (M_pos * sat_mask).sum(axis=1).astype(np.int32)
    sat_counts_neg = (M_neg * sat_mask).sum(axis=1).astype(np.int32)

    if num_unsat == 0:
        return assignment.tolist()

    max_iters = 500000
    p_walk = 0.5
    
    for _ in range(max_iters):
        if num_unsat == 0:
            break
            
        # Compute gains
        gain_true = unsat_counts_pos - sat_counts_neg
        gain_false = unsat_counts_neg - sat_counts_pos
        
        max_gain_true = np.max(gain_true)
        max_gain_false = np.max(gain_false)
        
        best_gain = -np.inf
        best_var = -1
        best_val = False
        
        if max_gain_true > max_gain_false:
            best_gain = max_gain_true
            candidates = np.where(gain_true == max_gain_true)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = True
        elif max_gain_false > max_gain_true:
            best_gain = max_gain_false
            candidates = np.where(gain_false == max_gain_false)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = False
        else:
            if max_gain_true == max_gain_false:
                if rng.rand() < 0.5:
                    candidates = np.where(gain_true == max_gain_true)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = True
                    best_gain = max_gain_true
                else:
                    candidates = np.where(gain_false == max_gain_false)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = False
                    best_gain = max_gain_false
            else:
                best_var = int(rng.randint(0, n))
                best_val = not assignment[best_var]
                best_gain = 0

        # Escape mechanism
        if best_gain <= 0 and rng.rand() < p_walk:
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                break
            ci = unsat_indices[rng.randint(len(unsat_indices))]
            j = rng.randint(3)
            v = int(clause_vars[ci, j])
            best_var = v
            if rng.rand() < 0.5:
                best_val = (clause_signs[ci, j] == 1)
            else:
                best_val = not assignment[v]

        # Apply flip
        if best_val != assignment[best_var]:
            assignment[best_var] = best_val
            
            # Identify affected clauses
            affected_clauses = np.where((M_pos[best_var, :] == 1) | (M_neg[best_var, :] == 1))[0]
            
            for ci in affected_clauses:
                # Determine if clause is now sat
                is_sat = False
                for j in range(3):
                    v = int(clause_vars[ci, j])
                    s = (clause_signs[ci, j] == 1)
                    if assignment[v] == s:
                        is_sat = True
                        break
                
                was_sat = clause_sat[ci]
                clause_sat[ci] = is_sat
                
                if was_sat != is_sat:
                    # Update global num_unsat
                    if was_sat and not is_sat:
                        num_unsat += 1
                    elif not was_sat and is_sat:
                        num_unsat -= 1
                    
                    # Update counts for all literals in this clause
                    delta_unsat = 1 if not is_sat else -1
                    delta_sat = -1 if not is_sat else 1
                    
                    for j in range(3):
                        v = int(clause_vars[ci, j])
                        if clause_signs[ci, j] == 1:
                            unsat_counts_pos[v] += delta_unsat
                            sat_counts_pos[v] += delta_sat
                        else:
                            unsat_counts_neg[v] += delta_unsat
                            sat_counts_neg[v] += delta_sat

    return assignment.tolist()
