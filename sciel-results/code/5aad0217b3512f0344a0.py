import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precompute clause structure
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    clause_lens = np.zeros(m, dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
        clause_lens[i] = len(c)
        
    # Handle short clauses (though task says 3-SAT, safety check)
    # If clause len < 3, we pad with -1 (sign 0) in the loop above.
    
    # Build var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            if clause_signs[i, j] != 0:
                var_to_clauses[v].append(i)
                
    # Precompute var_clause_pos for fast gain calculation
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
    
    # Frequency-based initialization
    # Count occurrences of each literal
    lit_freq = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            s = clause_signs[i, j]
            if s != 0:
                if s == 1:
                    lit_freq[v] += 1
                else:
                    lit_freq[v] -= 1
    
    def make_initial_assign():
        assign = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if assignment[i] is not None:
                assign[i] = assignment[i]
            else:
                # Bias towards more frequent literal
                # If lit_freq > 0, prefer True. If < 0, prefer False.
                # Add some noise
                bias = lit_freq[i] / (np.sum(np.abs(lit_freq)) + 1.0)
                # Map bias to probability
                # If bias is 0.5, prob 0.5.
                # Simple heuristic:
                if rng.random() < 0.8:
                    assign[i] = (lit_freq[i] > 0)
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
    
    # Luby restart sequence
    def luby(k):
        # Returns the k-th term of the Luby sequence
        # 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 8, ...
        # Implementation:
        # Find the largest power of 2 that divides k+1? No.
        # Luby sequence: l(2n+1) = 1, l(2n) = l(n).
        # Let's use a standard implementation or just simple increasing.
        # For simplicity and robustness, we use a simple increasing strategy with a cap,
        # or a geometric restart.
        # Let's use a simple geometric restart: 100, 200, 400, 800...
        # Actually, Luby is better.
        # Here is a quick Luby implementation:
        def luby_seq(k):
            # k is 1-based
            # l(1)=1, l(2)=1, l(3)=2, l(4)=1, l(5)=1, l(6)=4...
            # l(2n) = l(n)
            # l(2n+1) = 1
            if k % 2 == 1:
                return 1
            else:
                return luby_seq(k // 2)
        return luby_seq(k)

    restart_idx = 1
    max_restarts = 500
    base_steps = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Determine steps for this restart
        # Use Luby scaled by base_steps
        steps = luby(restart_idx) * base_steps
        # Cap steps to avoid extremely long restarts if not solving
        if steps > 10000:
            steps = 10000
        restart_idx += 1
        
        assign = make_initial_assign()
        clause_sat = get_clause_sat(assign)
        unsat_count = int(m - np.sum(clause_sat))
        
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # Adaptive walk probability
        walk_prob = 0.20
        step = 0
        no_improve_steps = 0
        last_unsat = unsat_count
        
        while step < steps:
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assign.tolist()
            
            # Select an unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assign.tolist()
            
            # Pick random unsat clause
            c_idx = int(unsat_indices[np_rng.randint(len(unsat_indices))])
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
            valid_lits = np.where(c_signs != 0)[0]
            if len(valid_lits) == 0:
                step += 1
                continue
                
            # Select variable to flip
            if rng.random() < walk_prob:
                # Walk step
                j = int(valid_lits[np_rng.randint(len(valid_lits))])
                var_idx = int(c_vars[j])
            else:
                # Greedy step: find var with minimum gain
                best_gain = float('inf')
                best_var = None
                
                for j in valid_lits:
                    var_idx = int(c_vars[j])
                    gain = compute_gain(var_idx, assign, clause_sat)
                    if gain < best_gain:
                        best_gain = gain
                        best_var = var_idx
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    j = int(valid_lits[np_rng.randint(len(valid_lits))])
                    var_idx = int(c_vars[j])
            
            # Perform the flip
            assign[var_idx] = not assign[var_idx]
            
            # Update clause statuses
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
                if walk_prob > 0.10:
                    walk_prob = max(0.10, walk_prob * 0.9)
            else:
                no_improve_steps += 1
                if no_improve_steps > 100:
                    walk_prob = min(0.5, walk_prob * 1.05)
                elif no_improve_steps > 300:
                    walk_prob = min(0.8, walk_prob * 1.1)
                elif no_improve_steps > 500:
                    walk_prob = 0.95
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
                
            if no_improve_steps > 2000:
                break

    # Final polish
    if best_assign is not None and best_unsat > 0:
        polish_deadline = deadline - 5.0
        assign = best_assign.copy()
        clause_sat = get_clause_sat(assign)
        unsat_count = int(m - np.sum(clause_sat))
        
        polish_steps = 0
        max_polish_steps = 20000
        
        while polish_steps < max_polish_steps and time.time() < polish_deadline:
            if unsat_count == 0:
                return assign.tolist()
            
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assign.tolist()
            
            c_idx = int(unsat_indices[np_rng.randint(len(unsat_indices))])
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
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
