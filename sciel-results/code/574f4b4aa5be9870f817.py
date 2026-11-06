import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 280  # safety margin before 330s
    
    # Precompute clause data as numpy arrays for speed
    # Each clause: list of signed literals (1-based)
    # Convert to 0-based and sign encoding: +1 for positive, -1 for negative
    # clause_vars[i] = array of variable indices (0-based)
    # clause_signs[i] = array of +1/-1
    clause_vars = np.array([[abs(l) - 1 for l in c] for c in clauses], dtype=np.int32)
    clause_signs = np.array([[1 if l > 0 else -1 for l in c] for c in clauses], dtype=np.int8)
    
    # For each variable, precompute which clauses contain it and with what sign
    # var_clause_map[v] = list of (clause_idx, sign)
    var_clause_map = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_vars[ci, j]
            s = clause_signs[ci, j]
            var_clause_map[v].append((ci, s))
    
    def eval_assignment(assignment):
        """Return number of unsatisfied clauses."""
        # assignment: numpy bool array of length n
        # For each clause, check if any literal is satisfied
        # literal satisfied if (assignment[var] == True) iff sign==+1, or (assignment[var]==False) iff sign==-1
        # i.e., assignment[var] * sign == True  (True*1=True, True*-1=False, False*1=False, False*-1=True)
        # Wait: sign=+1 means literal is +v, satisfied when v=True
        # sign=-1 means literal is -v, satisfied when v=False
        # So satisfied = (assignment[var] == (sign == +1))
        # = (assignment[var] == True) when sign==1, (assignment[var]==False) when sign==-1
        # = assignment[var] XOR (sign==-1) ... let me think differently
        # satisfied = assignment[var] if sign==1, else not assignment[var]
        # = assignment[var] ^ (sign==-1)  ... no
        # Let me just compute: for each clause, OR over its 3 literals
        # lit_val = assignment[var] if sign==1 else (not assignment[var])
        # = assignment[var] ^ (sign==-1) ... 
        # sign==1: assignment[var] ^ False = assignment[var] ✓
        # sign==-1: assignment[var] ^ True = not assignment[var] ✓
        # So lit_val = assignment[var] ^ (sign==-1)
        # But sign==-1 means sign is -1, so (sign==-1) is True when sign is -1
        # Actually in numpy: (clause_signs == -1) gives bool array
        # lit_val = assignment[clause_vars] ^ (clause_signs == -1)
        # satisfied_clause = lit_val.any(axis=1)
        # unsat_count = (~satisfied_clause).sum()
        
        lit_vals = assignment[clause_vars] ^ (clause_signs == -1)
        satisfied = lit_vals.any(axis=1)
        return int((~satisfied).sum())
    
    def get_unsat_clauses(assignment):
        """Return indices of unsatisfied clauses."""
        lit_vals = assignment[clause_vars] ^ (clause_signs == -1)
        satisfied = lit_vals.any(axis=1)
        return np.where(~satisfied)[0]
    
    best_unsat = m + 1
    best_assignment = None
    
    for attempt in range(200):
        if time.time() >= deadline:
            break
        
        rng = random.Random(attempt * 7919 + 42)
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_count = eval_assignment(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
        
        # WalkSAT with lookahead
        max_flips = 3000
        for flip_i in range(max_flips):
            if time.time() >= deadline:
                break
            
            unsat_idx = get_unsat_clauses(assignment)
            if len(unsat_idx) == 0:
                return assignment.tolist()
            
            if len(unsat_idx) < best_unsat:
                best_unsat = len(unsat_idx)
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            ci = unsat_idx[rng.randrange(len(unsat_idx))]
            
            # For each of the 3 variables in this clause, compute delta:
            # if we flip variable v, how does the unsat count change?
            # We need: for each variable in clause ci, compute net change in unsat clauses
            best_delta = 999999
            best_var = -1
            
            for j in range(3):
                v = int(clause_vars[ci, j])
                # Count how many clauses are currently satisfied only by this literal
                # and how many would become satisfied
                # More precisely: flipping v changes satisfaction of all clauses containing v
                # For each clause containing v:
                #   Before: clause is satisfied if any literal is true
                #   After: v's literal value flips, so clauses where v's literal was the only true literal become unsat
                #          clauses where v's literal was false and no other literal is true become sat
                # 
                # Simple approach: just count current unsat among v's clauses, and estimate
                # Actually, let's do a proper 1-step lookahead:
                # For variable v, consider all clauses in var_clause_map[v]
                # Current state: for each such clause, is it satisfied?
                # After flip: v's contribution flips, recheck
                
                v_clauses = var_clause_map[v]
                # Current unsat count for these clauses
                cur_unsat = 0
                new_unsat = 0
                for (cci, csign) in v_clauses:
                    # Check if clause cci is currently satisfied
                    lit_vals_c = assignment[clause_vars[cci]] ^ (clause_signs[cci] == -1)
                    cur_sat = lit_vals_c.any()
                    if not cur_sat:
                        cur_unsat += 1
                    # After flipping v:
                    # v's new value: not assignment[v]
                    # v's literal value after: (not assignment[v]) ^ (csign == -1)
                    # Other literals unchanged
                    new_v_lit = (not assignment[v]) ^ (csign == -1)
                    other_satisfied = False
                    for k in range(3):
                        if int(clause_vars[cci, k]) == v:
                            continue
                        other_lit = assignment[int(clause_vars[cci, k])] ^ (clause_signs[cci, k] == -1)
                        if other_lit:
                            other_satisfied = True
                            break
                    new_sat = new_v_lit or other_satisfied
                    if not new_sat:
                        new_unsat += 1
                
                delta = new_unsat - cur_unsat
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
        
        # End of this attempt
        final_unsat = eval_assignment(assignment)
        if final_unsat == 0:
            return assignment.tolist()
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = assignment.copy()
    
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
