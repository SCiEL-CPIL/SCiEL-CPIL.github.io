import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precompute clause structure as numpy arrays for vectorized operations
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    clause_lens = np.zeros(m, dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
        clause_lens[i] = len(c)
    
    # Build var_to_clauses mapping with numpy-friendly structure
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            if clause_signs[i, j] != 0:
                var_to_clauses[v].append(i)
    
    # Precompute for each variable: list of (clause_idx, position_in_clause)
    var_clause_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            if clause_signs[i, j] != 0:
                var_clause_pos[v].append((i, j))
    
    # Unit propagation preprocessing
    assignment = [None] * n
    changed = True
    while changed:
        changed = False
        for i in range(m):
            unassigned_count = 0
            satisfied = False
            forced_lit = None
            forced_val = None
            for j in range(3):
                v = int(clause_vars[i, j])
                s = clause_signs[i, j]
                if s == 0:
                    continue
                if assignment[v] is not None:
                    if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                        satisfied = True
                        break
                else:
                    unassigned_count += 1
                    forced_lit = v
                    forced_val = (s == 1)
            
            if satisfied:
                continue
                
            if unassigned_count == 1 and forced_lit is not None:
                if assignment[forced_lit] is None:
                    assignment[forced_lit] = forced_val
                    changed = True
                elif assignment[forced_lit] != forced_val:
                    break
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    def make_initial_assign():
        assign = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if assignment[i] is not None:
                assign[i] = assignment[i]
            else:
                assign[i] = rng.random() < 0.5
        return assign
    
    def get_clause_sat(assign):
        sat = np.zeros(m, dtype=np.bool_)
        for j in range(3):
            vars_j = clause_vars[:, j]
            signs_j = clause_signs[:, j]
            valid = signs_j != 0
            if not np.any(valid):
                continue
            vars_valid = vars_j[valid]
            signs_valid = signs_j[valid]
            lit_sat = np.zeros(m, dtype=np.bool_)
            lit_sat[valid] = (assign[vars_valid] == (signs_valid > 0))
            sat |= lit_sat
        return sat
    
    def compute_gain(var_idx, assign, clause_sat):
        """Compute gain (change in unsat count) if we flip var_idx. Lower is better."""
        current_val = assign[var_idx]
        new_val = not current_val
        gain = 0
        
        for ci, j in var_clause_pos[var_idx]:
            s = clause_signs[ci, j]
            lit_sat_new = (new_val == (s > 0))
            lit_sat_old = (current_val == (s > 0))
            
            is_sat_now = clause_sat[ci]
            
            if is_sat_now:
                if lit_sat_old and not lit_sat_new:
                    # Check if any other literal in this clause is satisfied
                    others_sat = False
                    for k in range(3):
                        if k == j or clause_signs[ci, k] == 0:
                            continue
                        v = int(clause_vars[ci, k])
                        sk = clause_signs[ci, k]
                        if assign[v] == (sk > 0):
                            others_sat = True
                            break
                    if not others_sat:
                        gain += 1
            else:
                if lit_sat_new:
                    gain -= 1
        
        return gain
    
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 200
    base_max_steps = 1500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assign = make_initial_assign()
        clause_sat = get_clause_sat(assign)
        unsat_count = int(m - np.sum(clause_sat))
        
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # Adaptive walk probability
        walk_prob = 0.25
        max_steps = base_max_steps + restart * 300
        if restart > 30:
            max_steps = int(max_steps * 1.5)
        if restart > 80:
            max_steps = int(max_steps * 2.0)
        
        step = 0
        no_improve_steps = 0
        last_unsat = unsat_count
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assign.tolist()
            
            # Select an unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assign.tolist()
            
            # Least-constrained clause selection: prefer clauses with fewer unassigned/unsatisfied vars
            # Use a simple heuristic: pick randomly but weight by clause "difficulty"
            c_idx = int(unsat_indices[np_rng.randint(len(unsat_indices))])
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
            # Select variable to flip
            if rng.random() < walk_prob:
                # Walk step: random literal from clause
                valid_lits = np.where(c_signs != 0)[0]
                j = int(valid_lits[np_rng.randint(len(valid_lits))])
                var_idx = int(c_vars[j])
            else:
                # Greedy step: find var with minimum gain
                best_gain = float('inf')
                best_var = None
                
                valid_lits = np.where(c_signs != 0)[0]
                for j in valid_lits:
                    var_idx = int(c_vars[j])
                    gain = compute_gain(var_idx, assign, clause_sat)
                    if gain < best_gain:
                        best_gain = gain
                        best_var = var_idx
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    valid_lits = np.where(c_signs != 0)[0]
                    j = int(valid_lits[np_rng.randint(len(valid_lits))])
                    var_idx = int(c_vars[j])
            
            # Perform the flip
            assign[var_idx] = not assign[var_idx]
            
            # Update clause statuses for all clauses containing var_idx
            for ci, j in var_clause_pos[var_idx]:
                old_sat = clause_sat[ci]
                new_sat = False
                for k in range(3):
                    if clause_signs[ci, k] == 0:
                        continue
                    v = int(clause_vars[ci, k])
                    s = clause_signs[ci, k]
                    if assign[v] == (s > 0):
                        new_sat = True
                        break
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                clause_sat[ci] = new_sat
            
            step += 1
            
            # Track progress and adapt walk probability
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
                if walk_prob > 0.15:
                    walk_prob = max(0.15, walk_prob * 0.95)
            else:
                no_improve_steps += 1
                if no_improve_steps > 150:
                    walk_prob = min(0.6, walk_prob * 1.08)
                elif no_improve_steps > 400:
                    walk_prob = min(0.8, walk_prob * 1.1)
                elif no_improve_steps > 800:
                    walk_prob = 0.9
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
            
            if no_improve_steps > 2500:
                break
    
    # Final polish: try to fix remaining unsatisfied clauses
    if best_assign is not None and best_unsat > 0:
        polish_deadline = deadline - 5.0
        assign = best_assign.copy()
        clause_sat = get_clause_sat(assign)
        unsat_count = int(m - np.sum(clause_sat))
        
        polish_steps = 0
        max_polish_steps = 5000
        
        while polish_steps < max_polish_steps and time.time() < polish_deadline:
            if unsat_count == 0:
                return assign.tolist()
            
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assign.tolist()
            
            c_idx = int(unsat_indices[np_rng.randint(len(unsat_indices))])
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
            # Try each variable in the clause, pick best gain
            best_gain = float('inf')
            best_var = None
            valid_lits = np.where(c_signs != 0)[0]
            for j in valid_lits:
                var_idx = int(c_vars[j])
                gain = compute_gain(var_idx, assign, clause_sat)
                if gain < best_gain:
                    best_gain = gain
                    best_var = var_idx
            
            if best_var is None:
                break
            
            var_idx = best_var
            assign[var_idx] = not assign[var_idx]
            
            for ci, j in var_clause_pos[var_idx]:
                old_sat = clause_sat[ci]
                new_sat = False
                for k in range(3):
                    if clause_signs[ci, k] == 0:
                        continue
                    v = int(clause_vars[ci, k])
                    s = clause_signs[ci, k]
                    if assign[v] == (s > 0):
                        new_sat = True
                        break
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                clause_sat[ci] = new_sat
            
            polish_steps += 1
        
        if unsat_count == 0:
            return assign.tolist()
        if unsat_count < best_unsat:
            best_assign = assign
            best_unsat = unsat_count
    
    if best_assign is not None:
        final_sat = get_clause_sat(best_assign)
        if np.all(final_sat):
            return best_assign.tolist()
    
    return best_assign.tolist() if best_assign is not None else [False] * n
