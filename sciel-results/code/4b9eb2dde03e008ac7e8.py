import time
import numpy as np
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [True] * n

    start_time = time.time()
    # 330s limit. Use 320s for search, 10s buffer.
    deadline = start_time + 320.0

    # Preprocess: Remove tautologies, duplicates.
    processed_clauses = []
    for c in clauses:
        seen = set()
        new_c = []
        has_pos = False
        has_neg = False
        for l in c:
            v = abs(l)
            if v in seen:
                continue
            seen.add(v)
            if l > 0:
                has_pos = True
            else:
                has_neg = True
            new_c.append(l)
        if has_pos and has_neg:
            continue # Tautology
        if len(new_c) == 0:
            return [False] * n # UNSAT
        processed_clauses.append(tuple(new_c))
    
    clauses = processed_clauses
    m = len(clauses)
    if m == 0:
        return [True] * n

    # Build NumPy arrays for efficient vectorized operations
    # var_idx[i, j] is the 0-based variable index for literal j in clause i
    # sign[i, j] is 1 if literal is positive, 0 if negative
    var_idx = np.zeros((m, 3), dtype=np.int32)
    sign = np.zeros((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            var_idx[i, j] = abs(l) - 1
            sign[i, j] = 1 if l > 0 else 0

    # Precompute variable -> list of (clause_idx, pos_idx, sign)
    # We need this for fast delta calculation in Min-Conflict
    var_info = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            s = int(sign[i, j])
            var_info[v].append((i, j, s))

    # Convert to numpy arrays for vectorized access if possible, or keep as lists for small n
    # For n=150, lists of tuples are fast enough for the inner loop if we vectorize the check.
    # However, to get the speedup mentioned in knowledge, we should vectorize the "score" calculation.
    
    # Let's use a simpler approach for Min-Conflict that is highly optimized:
    # For each variable, we store:
    # - clauses: array of clause indices
    # - pos: array of positions in clause
    # - signs: array of signs
    var_clauses_arr = []
    var_pos_arr = []
    var_signs_arr = []
    
    for v in range(n):
        if var_info[v]:
            vc = np.array([x[0] for x in var_info[v]], dtype=np.int32)
            vp = np.array([x[1] for x in var_info[v]], dtype=np.int32)
            vs = np.array([x[2] for x in var_info[v]], dtype=np.int32)
        else:
            vc = np.array([], dtype=np.int32)
            vp = np.array([], dtype=np.int32)
            vs = np.array([], dtype=np.int32)
        var_clauses_arr.append(vc)
        var_pos_arr.append(vp)
        var_signs_arr.append(vs)

    rng = np.random.default_rng(42)
    
    def check_solution(assignment):
        """Returns True if assignment satisfies all clauses."""
        # assignment: [n] 0/1
        # var_idx: [m, 3]
        # sign: [m, 3]
        # satisfied[i, j] = (assignment[var_idx[i, j]] == sign[i, j])
        # clause_sat[i] = any(satisfied[i, :])
        sat_lits = (assignment[var_idx] == sign)
        clause_sat = sat_lits.any(axis=1)
        return clause_sat.all()

    def count_unsat(assignment):
        """Returns number of unsatisfied clauses."""
        sat_lits = (assignment[var_idx] == sign)
        clause_sat = sat_lits.any(axis=1)
        return int((~clause_sat).sum())

    def update_unsat_count(assignment, num_unsat_per_clause, v, new_val):
        """
        Update num_unsat_per_clause when variable v flips to new_val.
        num_unsat_per_clause[i] is 0 if clause i is satisfied, >0 if not? 
        Better: track num_satisfied_lits per clause.
        """
        # This function is tricky to vectorize perfectly without rebuilding.
        # Instead, we will recalculate the delta for the specific variable's clauses.
        pass # We will use a different strategy: recalculate cost for candidates directly.

    # Strategy: WalkSAT with Min-Conflict Heuristic
    # State: assignment [n] (0/1)
    # Metric: number of unsatisfied clauses.
    
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 2000
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize randomly
        assignment = (rng.random(n) < 0.5).astype(np.int32)
        
        # Calculate initial unsat count
        sat_lits = (assignment[var_idx] == sign)
        num_sat_lits_per_clause = sat_lits.sum(axis=1)
        unsat_clauses = (num_sat_lits_per_clause == 0).sum()
        
        if unsat_clauses == 0:
            return assignment.astype(bool).tolist()
            
        if unsat_clauses < best_unsat:
            best_unsat = unsat_clauses
            best_assign = assignment.copy()
            
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            # If solved, return
            if (num_sat_lits_per_clause == 0).sum() == 0:
                return assignment.astype(bool).tolist()
                
            # Track best
            current_unsat = int((num_sat_lits_per_clause == 0).sum())
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assignment.copy()
                
            # Select an unsatisfied clause
            unsat_indices = np.where(num_sat_lits_per_clause == 0)[0]
            if len(unsat_indices) == 0:
                return assignment.astype(bool).tolist()
            c_idx = int(unsat_indices[rng.integers(len(unsat_indices))])
            
            # Get variables in this clause
            c_vars = var_idx[c_idx]
            c_signs = sign[c_idx]
            
            # Heuristic: Min-Conflict
            # For each variable in the clause, calculate the change in unsat count if flipped.
            # Delta = (New Unsat Count) - (Old Unsat Count)
            # We want to minimize Delta.
            
            best_v = -1
            best_delta = float('inf')
            
            # There are at most 3 variables in a 3-SAT clause.
            # We can calculate delta for each of the 3 candidates.
            for k in range(3):
                v = int(c_vars[k])
                s = int(c_signs[k])
                old_val = assignment[v]
                new_val = 1 - old_val
                
                # Get all clauses containing v
                vc = var_clauses_arr[v]
                vp = var_pos_arr[v]
                vs = var_signs_arr[v]
                
                if len(vc) == 0:
                    continue
                    
                # Current state of these clauses
                # num_sat_lits_per_clause[vc]
                
                # For each clause in vc:
                # 1. Does v currently satisfy it?
                #    v_satisfies_old = (old_val == vs)
                # 2. Will v satisfy it after flip?
                #    v_satisfies_new = (new_val == vs)
                
                v_sat_old = (old_val == vs)
                v_sat_new = (new_val == vs)
                
                # Clauses that are currently unsatisfied (num_sat_lits == 0)
                # If v_satisfies_new is True, this clause becomes satisfied (Unsat count -1)
                # If v_satisfies_old is True and num_sat_lits == 1, this clause becomes unsatisfied (Unsat count +1)
                
                # Gain: Clauses that were unsat (0 sat lits) and v satisfies them now.
                # Condition: num_sat_lits_per_clause[vc] == 0 AND v_sat_new
                gain = ((num_sat_lits_per_clause[vc] == 0) & v_sat_new).sum()
                
                # Loss: Clauses that were sat (num_sat_lits >= 1) and v was the ONLY one satisfying them?
                # No, if v was satisfying it, and we flip it, it loses one sat lit.
                # It becomes unsat ONLY IF it had exactly 1 sat lit (which was v).
                # Condition: num_sat_lits_per_clause[vc] == 1 AND v_sat_old
                loss = ((num_sat_lits_per_clause[vc] == 1) & v_sat_old).sum()
                
                delta = int(loss - gain)
                
                # Tie-breaking: random or prefer smaller delta.
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
            # If no variable found (should not happen), pick random
            if best_v == -1:
                best_v = int(c_vars[rng.integers(3)])
                
            # Apply flip
            old_val = assignment[best_v]
            new_val = 1 - old_val
            assignment[best_v] = new_val
            
            # Update num_sat_lits_per_clause for all clauses containing best_v
            vc = var_clauses_arr[best_v]
            vs = var_signs_arr[best_v]
            
            if len(vc) > 0:
                # Determine change in sat count for each clause
                # If old_val satisfied, subtract 1. If new_val satisfies, add 1.
                # old_sat = (old_val == vs)
                # new_sat = (new_val == vs)
                # delta_sat = new_sat.astype(int) - old_sat.astype(int)
                
                old_sat = (old_val == vs).astype(np.int32)
                new_sat = (new_val == vs).astype(np.int32)
                delta_sat = new_sat - old_sat
                
                num_sat_lits_per_clause[vc] += delta_sat

    if best_assign is not None:
        return best_assign.astype(bool).tolist()
        
    return (rng.random(n) < 0.5).astype(bool).tolist()
