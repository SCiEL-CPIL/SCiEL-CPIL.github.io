import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    if m == 0:
        return [True] * n
    
    # Build clause structure as numpy arrays
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # Build variable occurrence: var_occ[v] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for ci in range(m):
        for k in range(3):
            v = clause_vars[ci, k]
            s = clause_signs[ci, k]
            var_occ[v].append((ci, s))
    
    # --- Preprocessing: Unit Propagation ---
    assignment = [None] * n
    clause_sat = np.zeros(m, dtype=np.int32)
    
    def compute_clause_sat():
        for ci in range(m):
            sc = 0
            for k in range(3):
                v = clause_vars[ci, k]
                s = clause_signs[ci, k]
                if assignment[v] is not None:
                    if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                        sc += 1
            clause_sat[ci] = sc
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for ci in range(m):
                if clause_sat[ci] > 0:
                    continue
                unassigned = []
                for k in range(3):
                    v = clause_vars[ci, k]
                    if assignment[v] is None:
                        unassigned.append((v, clause_signs[ci, k]))
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    assignment[v] = (s == 1)
                    changed = True
            if changed:
                compute_clause_sat()
    
    compute_clause_sat()
    propagate()
    
    # Check if solved
    if np.all(clause_sat > 0):
        return [a if a is not None else True for a in assignment]
    
    # Initialize unassigned variables randomly
    rng = random.Random(42)
    for v in range(n):
        if assignment[v] is None:
            assignment[v] = rng.random() < 0.5
    
    # Convert to numpy bool array for fast operations
    assign_np = np.array(assignment, dtype=bool)
    
    # Phase saving array
    phase = assign_np.copy()
    
    # Recompute clause satisfaction
    def recompute_clause_sat():
        for ci in range(m):
            sc = 0
            for k in range(3):
                v = clause_vars[ci, k]
                s = clause_signs[ci, k]
                if (s == 1 and assign_np[v]) or (s == -1 and not assign_np[v]):
                    sc += 1
            clause_sat[ci] = sc
    
    recompute_clause_sat()
    unsat_clauses = np.where(clause_sat == 0)[0].tolist()
    
    if not unsat_clauses:
        return [bool(a) for a in assign_np]
    
    best_unsat = len(unsat_clauses)
    best_assignment = assign_np.copy()
    
    # Clause weights for WMinOnes
    clause_weights = np.ones(m, dtype=np.float64)
    
    max_restarts = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Adaptive step budget based on remaining time
        steps_per_restart = max(200, int(remaining * 150))
        
        # Initialize or perturb
        if restart == 0:
            pass
        elif restart % 20 == 0:
            # Perturb from best
            assign_np = best_assignment.copy()
            phase = best_assignment.copy()
            num_flips = min(n // 4, 8)
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                assign_np[v] = not assign_np[v]
            recompute_clause_sat()
        else:
            # Random restart with phase saving bias
            for v in range(n):
                if rng.random() < 0.85:
                    assign_np[v] = phase[v]
                else:
                    assign_np[v] = rng.random() < 0.5
            recompute_clause_sat()
        
        unsat_clauses = np.where(clause_sat == 0)[0].tolist()
        unsat_count = len(unsat_clauses)
        
        if unsat_count == 0:
            return [bool(a) for a in assign_np]
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assign_np.copy()
            phase = assign_np.copy()
        
        # Decay weights
        clause_weights *= 0.93
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
            
            if not unsat_clauses:
                return [bool(a) for a in assign_np]
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assign_np.copy()
                phase = assign_np.copy()
            
            # Pick a random unsat clause
            c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            
            # Increase weight of this clause
            clause_weights[c_idx] += 1.0
            
            # WMinOnes Heuristic
            best_gain = -1000.0
            best_var = -1
            
            for k in range(3):
                v = clause_vars[c_idx, k]
                s = clause_signs[c_idx, k]
                gain = 0.0
                
                # Iterate over all clauses containing v
                for (ci2, s2) in var_occ[v]:
                    w = clause_weights[ci2]
                    v_val = assign_np[v]
                    
                    # Does current v satisfy the literal in ci2?
                    cur_lit_true = (s2 == 1 and v_val) or (s2 == -1 and not v_val)
                    new_lit_true = not cur_lit_true
                    
                    if cur_lit_true and not new_lit_true:
                        # We are falsifying a currently true literal
                        if clause_sat[ci2] == 1:
                            gain -= w
                    elif not cur_lit_true and new_lit_true:
                        # We are satisfying a currently false literal
                        if clause_sat[ci2] == 0:
                            gain += w
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = v
            
            # With small probability, pick randomly to escape local optima
            if rng.random() < 0.1:
                best_var = clause_vars[c_idx, rng.randint(0, 2)]
            
            if best_var < 0:
                best_var = clause_vars[c_idx, rng.randint(0, 2)]
            
            # Flip the best variable
            v_to_flip = best_var
            assign_np[v_to_flip] = not assign_np[v_to_flip]
            phase[v_to_flip] = assign_np[v_to_flip]
            
            # Update clause_sat for all clauses containing v_to_flip
            newly_unsat = []
            for (ci2, s2) in var_occ[v_to_flip]:
                old_sat = clause_sat[ci2]
                sc = 0
                for k in range(3):
                    v2 = clause_vars[ci2, k]
                    s3 = clause_signs[ci2, k]
                    if (s3 == 1 and assign_np[v2]) or (s3 == -1 and not assign_np[v2]):
                        sc += 1
                clause_sat[ci2] = sc
                if sc == 0 and old_sat > 0:
                    newly_unsat.append(ci2)
            
            # Update unsat_clauses list
            unsat_clauses = [ci for ci in unsat_clauses if clause_sat[ci] == 0]
            unsat_clauses.extend(newly_unsat)
    
    # Final polish: try to fix remaining unsat clauses with targeted flips
    if time.time() < deadline - 2:
        unsat_clauses = np.where(clause_sat == 0)[0].tolist()
        for _ in range(min(500, int((deadline - time.time()) * 50))):
            if time.time() > deadline - 1:
                break
            if not unsat_clauses:
                return [bool(a) for a in assign_np]
            
            c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            # Try flipping each variable in the clause
            best_var = -1
            best_new_unsat = len(unsat_clauses)
            
            for k in range(3):
                v = clause_vars[c_idx, k]
                old_val = assign_np[v]
                assign_np[v] = not old_val
                
                new_unsat_count = 0
                affected = []
                for (ci2, s2) in var_occ[v]:
                    old_sat = clause_sat[ci2]
                    sc = 0
                    for kk in range(3):
                        v2 = clause_vars[ci2, kk]
                        s3 = clause_signs[ci2, kk]
                        if (s3 == 1 and assign_np[v2]) or (s3 == -1 and not assign_np[v2]):
                            sc += 1
                    clause_sat[ci2] = sc
                    if sc == 0:
                        new_unsat_count += 1
                    affected.append(ci2)
                
                if new_unsat_count < best_new_unsat:
                    best_new_unsat = new_unsat_count
                    best_var = v
                    best_affected = affected[:]
                    best_old_sat = {ci2: clause_sat[ci2] for ci2 in affected}
                
                # Restore
                assign_np[v] = old_val
                for ci2 in affected:
                    clause_sat[ci2] = best_old_sat[ci2] if 'best_old_sat' in dir() else clause_sat[ci2]
            
            if best_var >= 0 and best_new_unsat < len(unsat_clauses):
                assign_np[best_var] = not assign_np[best_var]
                for ci2 in best_affected:
                    sc = 0
                    for kk in range(3):
                        v2 = clause_vars[ci2, kk]
                        s3 = clause_signs[ci2, kk]
                        if (s3 == 1 and assign_np[v2]) or (s3 == -1 and not assign_np[v2]):
                            sc += 1
                    clause_sat[ci2] = sc
                unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
                if len(unsat_clauses) < best_unsat:
                    best_unsat = len(unsat_clauses)
                    best_assignment = assign_np.copy()
                    phase = assign_np.copy()
            else:
                # Random flip
                v = rng.randint(0, n - 1)
                assign_np[v] = not assign_np[v]
                for ci2 in var_occ[v]:
                    sc = 0
                    for kk in range(3):
                        v2 = clause_vars[ci2, kk]
                        s3 = clause_signs[ci2, kk]
                        if (s3 == 1 and assign_np[v2]) or (s3 == -1 and not assign_np[v2]):
                            sc += 1
                    clause_sat[ci2] = sc
                unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
    
    if np.all(clause_sat > 0):
        return [bool(a) for a in assign_np]
    
    return [bool(a) for a in best_assignment]
