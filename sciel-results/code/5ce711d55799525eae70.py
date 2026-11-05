"""Planted 3-SAT solver with vectorized WalkSAT + adaptive noise + restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: aim to finish well before 330s
    deadline = time.time() + 320.0

    # Precompute literal arrays: shape (m, 3)
    # Each literal is signed 1-based.
    lit_arr = np.array([[l for l in c] for c in clauses], dtype=np.int32)  # (m, 3)
    var_arr = np.abs(lit_arr) - 1  # 0-based variable indices
    sign_arr = np.sign(lit_arr).astype(np.int8)  # +1 or -1

    # We'll track assignment as a numpy array of booleans (True = positive literal true)
    # sat_count[i] = number of literals in clause i that are satisfied

    def compute_sat_counts(assign):
        # assign: bool array of length n
        # For each literal, it's satisfied if assign[var] == (sign > 0)
        # i.e., assign[var] == (sign == 1)
        # Let's compute: for each literal (m,3), satisfied = assign[var_arr] == (sign_arr == 1)
        # sign_arr == 1 means positive literal; assign[var] should be True
        # sign_arr == -1 means negative literal; assign[var] should be False
        # So satisfied = assign[var_arr] if sign>0 else ~assign[var_arr]
        # Equivalently: satisfied = (assign[var_arr].astype(int8) == sign_arr)
        # Because assign True=1, False=0; sign +1 for positive, -1 for negative
        # Wait: if sign=+1, we need assign[var]=True=1 -> match
        # If sign=-1, we need assign[var]=False=0 -> but sign is -1, so 0 != -1
        # So that doesn't work directly. Let me think again.
        # 
        # Let's use: val = assign[var_arr] (bool array shape (m,3))
        # For positive literal (sign=1): satisfied if val is True
        # For negative literal (sign=-1): satisfied if val is False
        # So satisfied = (val == (sign_arr == 1))
        val = assign[var_arr]  # (m, 3)
        target = (sign_arr == 1)  # (m, 3), True for positive literals
        sat = (val == target)  # (m, 3)
        return sat.sum(axis=1)  # (m,)

    # We'll use a smarter representation for speed:
    # Keep assign as int8 array: 1 for True, 0 for False
    # sat_count: int32 array of length m
    # When we flip variable v:
    #   - For each clause containing v, check if the literal was satisfied before and after
    #   - Precompute for each variable, which clauses and positions it appears in

    # Build variable-to-clause-position mapping
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = var_arr[i, j]
            var_clauses[v].append((i, j))

    # Convert to numpy arrays for each variable for faster access
    var_clause_idx = []
    var_pos_idx = []
    for v in range(n):
        if var_clauses[v]:
            idxs, poss = zip(*var_clauses[v])
            var_clause_idx.append(np.array(idxs, dtype=np.int32))
            var_pos_idx.append(np.array(poss, dtype=np.int32))
        else:
            var_clause_idx.append(np.array([], dtype=np.int32))
            var_pos_idx.append(np.array([], dtype=np.int32))

    # Also need sign for each occurrence: sign_arr[i, j]
    # We can precompute sign per variable occurrence too
    var_sign_arr = []
    for v in range(n):
        if var_clauses[v]:
            signs = np.array([sign_arr[i, j] for i, j in var_clauses[v]], dtype=np.int8)
            var_sign_arr.append(signs)
        else:
            var_sign_arr.append(np.array([], dtype=np.int8))

    def flip_variable(assign, sat_count, v):
        """Flip variable v, update sat_count in place. Returns number of newly satisfied clauses."""
        ci = var_clause_idx[v]
        if len(ci) == 0:
            return 0
        pos = var_pos_idx[v]
        signs = var_sign_arr[v]
        
        # Current value of variable v
        old_val = assign[v]
        # For each occurrence, check if literal was satisfied
        # satisfied_old = (old_val == (signs == 1))
        # After flip, new_val = ~old_val
        # satisfied_new = (new_val == (signs == 1))
        
        # A literal's satisfaction changes iff sign matches the old value in a specific way
        # If signs[k] == 1: satisfied when assign[v] == True. Flipping changes satisfaction iff assign[v] was True (becomes False, now unsat) or False (becomes True, now sat)
        # Actually: satisfied_old = (old_val == (signs[k]==1))
        # satisfied_new = ((~old_val) == (signs[k]==1))
        # These differ for all k? No:
        # If signs[k]==1: sat_old = old_val, sat_new = ~old_val. They always differ.
        # If signs[k]==-1: sat_old = ~old_val, sat_new = old_val. They always differ.
        # So flipping always changes the satisfaction status of every occurrence of v.
        # 
        # So: for each occurrence, if it was satisfied, it becomes unsat (decrement); if unsat, becomes sat (increment)
        
        # satisfied_old for each occurrence:
        # signs==1: sat_old = old_val (True=1, False=0)
        # signs==-1: sat_old = ~old_val
        # Combined: sat_old = (old_val == (signs == 1))
        
        old_val_bool = bool(old_val)
        sat_old = (old_val_bool == (signs == 1))  # bool array
        # sat_new = ~sat_old (since flipping always inverts)
        
        # Delta for sat_count at clauses ci:
        # For each k: if sat_old[k] is True, delta = -1; else delta = +1
        delta = np.where(sat_old, -1, 1).astype(np.int32)
        
        sat_count[ci] += delta
        assign[v] = not old_val_bool
        return delta.sum()

    def is_solved(sat_count):
        return np.all(sat_count >= 1)

    # Main search loop with restarts and adaptive noise
    best_assignment = None
    best_min_sat = -1  # We want to maximize the minimum sat_count (or minimize unsat)
    
    rng = np.random.default_rng(42)
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial assignment
        assign = rng.random(n) < 0.5
        sat_count = compute_sat_counts(assign)
        
        if is_solved(sat_count):
            return assign.tolist()
        
        # Track best seen
        current_unsat = np.sum(sat_count < 1)
        if current_unsat < best_min_sat or best_min_sat == -1:
            best_min_sat = current_unsat
            best_assignment = assign.copy()
        
        if current_unsat == 0:
            return assign.tolist()
        
        # WalkSAT phase
        # Adaptive parameters:
        # - Walk length: geometric distribution with mean ~5-10
        # - Noise: probability of random flip
        walk_mean = 5.0
        max_steps = 20000
        noise_prob = 0.05
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            # Find unsatisfied clauses
            unsat_mask = sat_count < 1
            n_unsat = np.sum(unsat_mask)
            
            if n_unsat == 0:
                return assign.tolist()
            
            # Update best
            if n_unsat < best_min_sat:
                best_min_sat = n_unsat
                best_assignment = assign.copy()
            
            # Choose an unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            clause_idx = rng.integers(len(unsat_indices))
            ci = unsat_indices[clause_idx]
            
            # Choose a variable from this clause
            # With probability noise_prob, pick a random variable from the clause
            # Otherwise, pick the variable that maximizes gain (fewest other unsat clauses affected)
            # For simplicity and speed, let's use a weighted approach
            
            # Get the 3 variables in this clause
            clause_vars = var_arr[ci]  # shape (3,)
            
            if rng.random() < noise_prob:
                # Random flip within clause
                v = clause_vars[rng.integers(3)]
            else:
                # Pick variable that minimizes number of other unsat clauses it's in
                # For each of the 3 vars, count how many unsat clauses they appear in
                best_v = None
                best_score = float('inf')
                for v in clause_vars:
                    # Count unsat clauses containing v
                    v_ci = var_clause_idx[v]
                    if len(v_ci) == 0:
                        score = 0
                    else:
                        score = np.sum(sat_count[v_ci] < 1)
                    if score < best_score:
                        best_score = score
                        best_v = v
                v = best_v
            
            # Flip variable v
            flip_variable(assign, sat_count, v)
        
        # After walk phase, if not solved, we'll restart with new random assignment
        # Optionally, we can do a "walk" restart: keep current assignment but add noise
        # For now, simple restart
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    
    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
