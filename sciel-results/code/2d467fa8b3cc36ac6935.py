import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill
    
    m = len(clauses)
    
    # Preprocess: convert to 0-based, store as arrays for speed
    # clause_vars[i] = array of 3 variable indices (0-based)
    # clause_pols[i] = array of 3 polarities (1 for positive, -1 for negative)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_pols = np.zeros((m, 3), dtype=np.int8)
    
    # var_clauses[v] = list of (clause_idx, polarity_sign) where polarity_sign is +1 or -1
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            v = abs(l) - 1
            p = 1 if l > 0 else -1
            clause_vars[i][j] = v
            clause_pols[i][j] = p
            var_clauses[v].append((i, p))
    
    # Convert var_clauses to numpy arrays for faster access
    var_clause_arrays = []
    for v in range(n):
        if var_clauses[v]:
            vc = np.array(var_clauses[v], dtype=np.int32)
            # vc[:, 0] = clause indices, vc[:, 1] = polarity signs
            var_clause_arrays.append(vc)
        else:
            var_clause_arrays.append(np.empty((0, 2), dtype=np.int32))
    
    # Precompute for each variable: the list of (clause_idx, polarity) as two separate arrays
    var_clause_idx = []
    var_clause_pol = []
    for v in range(n):
        if var_clause_arrays[v].shape[0] > 0:
            var_clause_idx.append(var_clause_arrays[v][:, 0])
            var_clause_pol.append(var_clause_arrays[v][:, 1])
        else:
            var_clause_idx.append(np.empty(0, dtype=np.int32))
            var_clause_pol.append(np.empty(0, dtype=np.int8))
    
    def evaluate_assignment(assignment):
        """Return number of unsatisfied clauses."""
        # assignment is a numpy bool array of length n
        # For each clause, check if any literal is satisfied
        # literal is satisfied if assignment[v] == (pol > 0)
        # i.e., assignment[v] * pol > 0
        # Convert assignment to int: True->1, False->-1
        assign_int = np.where(assignment, 1, -1).astype(np.int8)
        # For each clause, each literal: assign_int[var] * pol
        # Satisfied if product > 0
        products = assign_int[clause_vars] * clause_pols  # shape (m, 3)
        # A clause is satisfied if any product > 0
        sat_mask = (products > 0).any(axis=1)
        return int((~sat_mask).sum())
    
    def walksat_search(assignment, max_time, rng_seed):
        """
        Run WalkSAT starting from assignment.
        Returns (best_assignment, best_unsat, solved).
        """
        rng = random.Random(rng_seed)
        
        # Work on a copy
        x = assignment.copy()
        assign_int = np.where(x, 1, -1).astype(np.int8)
        
        # Compute initial state
        sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            cnt = 0
            for j in range(3):
                if assign_int[clause_vars[i][j]] == clause_pols[i][j]:
                    cnt += 1
            sat_count[i] = cnt
        
        unsat_count = int((sat_count == 0).sum())
        
        if unsat_count == 0:
            return x, 0, True
        
        best_unsat = unsat_count
        best_x = x.copy()
        best_assign_int = assign_int.copy()
        
        # WalkSAT parameters
        # Adaptive: for 150 vars, typical planted 3-SAT
        flip_prob = 0.25  # probability of random flip (noise)
        
        # Track unsatisfied clause indices for fast selection
        unsat_indices = np.where(sat_count == 0)[0].tolist()
        
        steps_per_restart = 500
        max_restarts = 10000  # will be limited by time
        
        restart = 0
        while time.time() < max_time:
            if unsat_count == 0:
                return x, 0, True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_x = x.copy()
                best_assign_int = assign_int.copy()
            
            # If we've done enough steps in this restart, restart
            if restart > 0 and restart % (steps_per_restart) == 0:
                # Random restart with perturbation
                for _ in range(5):
                    v = rng.randrange(n)
                    x[v] = not x[v]
                    assign_int[v] = -assign_int[v]
                    # Update sat counts
                    for c_idx, pol in var_clauses[v]:
                        if assign_int[v] == pol:
                            sat_count[c_idx] += 1
                            if sat_count[c_idx] == 1 and c_idx not in unsat_indices:
                                # was unsat, now sat
                                if c_idx in unsat_indices:
                                    unsat_indices.remove(c_idx)
                                unsat_count -= 1
                        else:
                            sat_count[c_idx] -= 1
                            if sat_count[c_idx] == 0 and c_idx not in unsat_indices:
                                unsat_indices.append(c_idx)
                                unsat_count += 1
                # Recompute to be safe
                unsat_indices = np.where(sat_count == 0)[0].tolist()
                unsat_count = len(unsat_indices)
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_x = x.copy()
                    best_assign_int = assign_int.copy()
                if unsat_count == 0:
                    return x, 0, True
            
            # Pick a random unsatisfied clause
            if not unsat_indices:
                unsat_indices = np.where(sat_count == 0)[0].tolist()
                unsat_count = len(unsat_indices)
                if unsat_count == 0:
                    return x, 0, True
            
            c = unsat_indices[rng.randrange(len(unsat_indices))]
            
            # Try each of the 3 variables in this clause
            best_var = -1
            best_score = float('inf')
            
            for j in range(3):
                v = clause_vars[c][j]
                pol = clause_pols[c][j]
                
                # Flipping v: if currently assign_int[v] == pol, flipping makes it -pol (breaks this clause)
                # if currently assign_int[v] != pol, flipping makes it pol (fixes this clause)
                
                broken = 0
                fixed = 0
                
                vc_idx = var_clause_idx[v]
                vc_pol = var_clause_pol[v]
                
                for k in range(len(vc_idx)):
                    ci = vc_idx[k]
                    cp = vc_pol[k]
                    if assign_int[v] == cp:
                        # Currently satisfied by v
                        if sat_count[ci] == 1:
                            # v is the only satisfying literal -> flipping breaks it
                            broken += 1
                    else:
                        # Currently not satisfied by v
                        if sat_count[ci] == 0:
                            # Flipping will satisfy this clause
                            fixed += 1
                
                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v
            
            # Decide: random flip with probability flip_prob, else best_var
            if rng.random() < flip_prob:
                v = clause_vars[c][rng.randrange(3)]
            else:
                v = best_var
            
            # Apply flip
            old_val = assign_int[v]
            assign_int[v] = -old_val
            x[v] = not x[v]
            
            # Update sat counts
            vc_idx = var_clause_idx[v]
            vc_pol = var_clause_pol[v]
            
            for k in range(len(vc_idx)):
                ci = vc_idx[k]
                cp = vc_pol[k]
                if old_val == cp:
                    # Was satisfying, now not
                    sat_count[ci] -= 1
                    if sat_count[ci] == 0:
                        if ci not in unsat_indices:
                            unsat_indices.append(ci)
                            unsat_count += 1
                else:
                    # Was not satisfying, now is
                    sat_count[ci] += 1
                    if sat_count[ci] == 1:
                        if ci in unsat_indices:
                            unsat_indices.remove(ci)
                            unsat_count -= 1
            
            restart += 1
        
        # Return best found
        if best_unsat == 0:
            return best_x, 0, True
        return best_x, best_unsat, False
    
    # Strategy: multiple independent WalkSAT runs with different seeds
    # Use time budget: run several attempts, keep best
    
    best_assignment = np.array([random.Random(42).random() < 0.5 for _ in range(n)], dtype=bool)
    best_unsat = evaluate_assignment(best_assignment)
    
    # Phase 1: Quick random assignments + short walks
    t0 = time.time()
    num_quick = 20
    for i in range(num_quick):
        if time.time() > deadline - 30:
            break
        rng = random.Random(1000 + i)
        x0 = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        u = evaluate_assignment(x0)
        if u < best_unsat:
            best_unsat = u
            best_assignment = x0.copy()
        if u == 0:
            return best_assignment.tolist()
        
        # Short walk from this starting point
        t_limit = min(deadline - 30, time.time() + 2.0)
        x, u, solved = walksat_search(x0, t_limit, 2000 + i)
        if u < best_unsat:
            best_unsat = u
            best_assignment = x.copy()
        if solved:
            return x.tolist()
    
    # Phase 2: Longer walks from best assignment
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        # Split time among several longer runs
        num_long = 5
        time_per_run = remaining / num_long
        for i in range(num_long):
            if time.time() > deadline - 10:
                break
            # Start from best assignment with small perturbation
            rng = random.Random(5000 + i * 100)
            x0 = best_assignment.copy()
            for _ in range(3):
                v = rng.randrange(n)
                x0[v] = not x0[v]
            
            t_limit = min(deadline - 10, time.time() + time_per_run)
            x, u, solved = walksat_search(x0, t_limit, 6000 + i * 100)
            if u < best_unsat:
                best_unsat = u
                best_assignment = x.copy()
            if solved:
                return x.tolist()
    
    # Final: one more attempt with remaining time
    remaining = deadline - 5 - time.time()
    if remaining > 2:
        x, u, solved = walksat_search(best_assignment, deadline - 5, 99999)
        if u < best_unsat:
            best_unsat = u
            best_assignment = x.copy()
        if solved:
            return x.tolist()
    
    # Verify
    if best_unsat == 0:
        return best_assignment.tolist()
    
    # Last resort: try to fix remaining unsatisfied clauses greedily
    x = best_assignment.copy()
    assign_int = np.where(x, 1, -1).astype(np.int8)
    
    # Recompute sat counts
    sat_count = np.zeros(m, dtype=np.int32)
    for i in range(m):
        for j in range(3):
            if assign_int[clause_vars[i][j]] == clause_pols[i][j]:
                sat_count[i] += 1
    
    unsat_indices = np.where(sat_count == 0)[0].tolist()
    
    # Try to fix each unsatisfied clause by flipping one of its variables
    for c in unsat_indices[:]:
        if time.time() > deadline - 2:
            break
        if sat_count[c] > 0:
            continue
        best_v = -1
        best_score = float('inf')
        for j in range(3):
            v = clause_vars[c][j]
            broken = 0
            fixed = 1  # fixing this clause
            for ci, cp in var_clauses[v]:
                if ci == c:
                    continue
                if assign_int[v] == cp:
                    if sat_count[ci] == 1:
                        broken += 1
                else:
                    if sat_count[ci] == 0:
                        fixed += 1
            score = broken - fixed
            if score < best_score:
                best_score = score
                best_v = v
        if best_v >= 0 and best_score <= 0:
            v = best_v
            old_val = assign_int[v]
            assign_int[v] = -old_val
            x[v] = not x[v]
            for ci, cp in var_clauses[v]:
                if old_val == cp:
                    sat_count[ci] -= 1
                else:
                    sat_count[ci] += 1
    
    final_unsat = int((sat_count == 0).sum())
    if final_unsat == 0:
        return x.tolist()
    
    return best_assignment.tolist()
