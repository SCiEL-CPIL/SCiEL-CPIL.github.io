import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # --- Preprocessing: Pure Literal Elimination ---
    pos_counts = [0] * n
    neg_counts = [0] * n
    
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    assignment = [None] * n
    
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if assignment[v] is not None:
                continue
            if pos_counts[v] > 0 and neg_counts[v] == 0:
                assignment[v] = True
                changed = True
            elif neg_counts[v] > 0 and pos_counts[v] == 0:
                assignment[v] = False
                changed = True
                
    # Apply assignments to simplify clauses
    new_clauses = []
    for c in clauses:
        new_c = []
        satisfied = False
        for lit in c:
            v = abs(lit) - 1
            if assignment[v] is not None:
                if assignment[v] == (lit > 0):
                    satisfied = True
                    break
                else:
                    continue
            else:
                new_c.append(lit)
                
        if not satisfied:
            if len(new_c) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            new_clauses.append(new_c)
            
    clauses = new_clauses
    
    if not clauses:
        final_assignment = [False] * n
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
            else:
                final_assignment[i] = random.random() < 0.5
        return final_assignment

    # Map to compact variable range
    var_map = {}
    new_n = 0
    for v in range(n):
        if assignment[v] is None:
            var_map[v] = new_n
            new_n += 1
            
    # Remap clauses to 0-based, separate pos/neg
    remapped_clauses = []
    for c in clauses:
        new_c = []
        for lit in c:
            v = abs(lit) - 1
            new_v = var_map[v]
            if lit > 0:
                new_c.append(new_v)  # positive: variable index
            else:
                new_c.append(-(new_v + 1))  # negative: -(var_index+1)
        remapped_clauses.append(new_c)
    
    m = len(remapped_clauses)
    
    # Build NumPy arrays for fast evaluation
    # For each clause, store: var indices (0-based) and signs (+1 for pos, -1 for neg)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int32)
    clause_lens = np.zeros(m, dtype=np.int32)
    
    for i, c in enumerate(remapped_clauses):
        for j, lit in enumerate(c):
            if lit >= 0:
                clause_vars[i, j] = lit
                clause_signs[i, j] = 1
            else:
                clause_vars[i, j] = -lit - 1
                clause_signs[i, j] = -1
        clause_lens[i] = len(c)
    
    # var_to_clauses index
    var_to_clauses = [[] for _ in range(new_n)]
    for i in range(m):
        for j in range(clause_lens[i]):
            var_to_clauses[clause_vars[i, j]].append(i)
    
    rng = random.Random(42)
    
    def evaluate_curr(curr):
        """Return (unsat_count, unsat_mask) where unsat_mask is boolean array of size m."""
        # curr is a numpy bool array of size new_n
        # For each clause, check if any literal is satisfied
        # A literal is satisfied if: sign * (curr[var] if sign>0 else not curr[var]) > 0
        # i.e., for pos: curr[var] == True; for neg: curr[var] == False
        
        # Build a matrix: for each clause i and position j, check satisfaction
        # Use broadcasting: curr[clause_vars] gives (m,3) array of bools
        # For pos literals: satisfied = curr[var]
        # For neg literals: satisfied = ~curr[var]
        # So: satisfied = (clause_signs == 1) & curr[var] | (clause_signs == -1) & ~curr[var]
        
        cv = clause_vars  # (m, 3) int32
        cs = clause_signs  # (m, 3) int32
        
        # Only consider valid positions (j < clause_lens[i])
        # Create a mask for valid positions
        pos_mask = np.arange(3)[None, :] < clause_lens[:, None]  # (m, 3)
        
        cv_valid = np.where(pos_mask, cv, 0)  # safe indexing
        curr_vals = curr[cv_valid]  # (m, 3) bool
        
        # Satisfied: for pos (sign=1), need curr_val=True; for neg (sign=-1), need curr_val=False
        sat = np.where(cs == 1, curr_vals, ~curr_vals)  # (m, 3)
        sat = np.where(pos_mask, sat, False)  # zero out invalid positions
        
        # Clause is satisfied if any literal is satisfied
        clause_sat = sat.any(axis=1)  # (m,)
        unsat_mask = ~clause_sat
        unsat_count = int(unsat_mask.sum())
        return unsat_count, unsat_mask
    
    def compute_gains(curr, unsat_mask):
        """For each variable in an unsat clause, compute the gain (number of clauses that would become unsat)."""
        # This is expensive; we'll compute it on-the-fly per clause
        pass
    
    best_assignment = None
    best_unsat = float('inf')
    
    # Phase 1: Try multiple random initializations with limited WalkSAT
    max_total_time = deadline - start_time
    
    while time.time() < deadline:
        # Initialize
        curr = np.array([rng.random() < 0.5 for _ in range(new_n)], dtype=bool)
        
        curr_unsat, unsat_mask = evaluate_curr(curr)
        
        if curr_unsat == 0:
            best_assignment = curr.copy()
            break
        
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = curr.copy()
        
        if best_unsat == 0:
            break
        
        # WalkSAT with adaptive walk probability
        walk_prob = 0.3
        steps_since_improvement = 0
        local_best_unsat = curr_unsat
        local_best_assign = curr.copy()
        
        max_steps = 50000  # per restart
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            if step % 1000 == 0 and step > 0:
                # Periodic check
                pass
            
            if local_best_unsat == 0:
                break
            
            # Choose an unsat clause
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                break
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Choose a literal to flip
            c_len = clause_lens[clause_idx]
            if rng.random() < walk_prob:
                j = rng.randint(0, int(c_len) - 1)
            else:
                # Min-conflicts: pick the literal whose flip causes fewest new unsats
                min_gain = float('inf')
                best_j = 0
                for j in range(int(c_len)):
                    v = int(clause_vars[clause_idx, j])
                    # Compute gain: how many currently-satisfied clauses would become unsat
                    gain = 0
                    for c_idx in var_to_clauses[v]:
                        if c_idx == clause_idx:
                            continue
                        # Check if c_idx is currently satisfied and only satisfied by this variable
                        c_len2 = int(clause_lens[c_idx])
                        all_sat = True
                        only_this = True
                        for j2 in range(c_len2):
                            v2 = int(clause_vars[c_idx, j2])
                            s2 = int(clause_signs[c_idx, j2])
                            if s2 == 1:
                                is_sat = curr[v2]
                            else:
                                is_sat = not curr[v2]
                            if not is_sat:
                                all_sat = False
                                break
                            if v2 != v:
                                only_this = False
                        if all_sat and only_this:
                            gain += 1
                    if gain < min_gain:
                        min_gain = gain
                        best_j = j
                j = best_j
            
            v = int(clause_vars[clause_idx, j])
            curr[v] = not curr[v]
            
            # Re-evaluate
            new_unsat, unsat_mask = evaluate_curr(curr)
            
            if new_unsat < local_best_unsat:
                local_best_unsat = new_unsat
                local_best_assign = curr.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            if new_unsat == 0:
                best_assignment = curr.copy()
                best_unsat = 0
                break
            
            # Adaptive walk prob
            if steps_since_improvement > 300:
                walk_prob = min(0.8, walk_prob * 1.02)
            elif steps_since_improvement > 2000:
                # Restart this walk
                break
        
        if best_unsat == 0:
            break
        
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assign
    
    # Map best_assignment back to original
    orig_to_new = [-1] * n
    for orig_v, new_v in var_map.items():
        orig_to_new[orig_v] = new_v
        
    final_assignment = [False] * n
    for i in range(n):
        if assignment[i] is not None:
            final_assignment[i] = assignment[i]
        else:
            new_v = orig_to_new[i]
            final_assignment[i] = bool(best_assignment[new_v])
            
    return final_assignment
