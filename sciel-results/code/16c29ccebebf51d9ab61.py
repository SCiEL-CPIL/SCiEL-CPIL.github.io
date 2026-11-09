"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # Pre-process: build clause variables, signs, and variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_signs = []
    
    for i, c in enumerate(clauses):
        cv = []
        cs = []
        for lit in c:
            v = abs(lit) - 1
            s = (lit > 0)
            cv.append(v)
            cs.append(s)
            var_to_clauses[v].append(i)
        clause_vars.append(cv)
        clause_signs.append(cs)
        
    num_clauses = len(clauses)
    
    # Phase initialization based on literal occurrence counts
    pos_counts = np.zeros(n, dtype=np.int32)
    neg_counts = np.zeros(n, dtype=np.int32)
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    phase = pos_counts > neg_counts
    phase_list = phase.tolist()
    
    # Spectral initialization: build incidence matrix and use SVD
    # The clause-variable incidence matrix M has shape (num_clauses, n)
    # M[i, v] = +1 if variable v appears positively in clause i
    #           -1 if variable v appears negatively in clause i
    #            0 if variable v does not appear in clause i
    # For a planted instance, the true assignment should satisfy:
    #   M @ x >= 1 for all rows, where x is the assignment vector (+1/-1)
    # We can try to find x by looking at the sign pattern of the
    # principal eigenvector of M^T M or via SVD of M.
    
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 10s margin before 300s limit
    
    # Try DPLL first for easy instances (limited time)
    dpll_deadline = start_time + 5.0
    
    def dpll_limited(n, clauses, deadline):
        var_to_clauses_local = [[] for _ in range(n)]
        for i, c in enumerate(clauses):
            for lit in c:
                var_to_clauses_local[abs(lit) - 1].append(i)
        
        active_clauses = [list(c) for c in clauses]
        assignment = {}
        
        def propagate():
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for c in active_clauses:
                    sat = False
                    for lit in c:
                        v = abs(lit) - 1
                        if v in assignment:
                            if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                                sat = True
                                break
                    if sat:
                        continue
                    
                    unassigned = []
                    for lit in c:
                        v = abs(lit) - 1
                        if v not in assignment:
                            unassigned.append(lit)
                    
                    if len(unassigned) == 0:
                        return False
                    
                    if len(unassigned) == 1:
                        lit = unassigned[0]
                        v = abs(lit) - 1
                        val = (lit > 0)
                        if v in assignment:
                            if assignment[v] != val:
                                return False
                        else:
                            assignment[v] = val
                            changed = True
                        continue
                    
                    new_clauses.append(unassigned)
                
                active_clauses[:] = new_clauses
                return True
        
        def dpll(depth=0):
            if time.time() > deadline:
                return None
            
            if not propagate():
                return None
            
            if not active_clauses:
                return [assignment.get(i, False) for i in range(n)]
            
            var_counts = [0] * n
            for c in active_clauses:
                for lit in c:
                    var_counts[abs(lit) - 1] += 1
            
            best_var = -1
            best_count = -1
            for v in range(n):
                if v not in assignment and var_counts[v] > best_count:
                    best_count = var_counts[v]
                    best_var = v
            
            if best_var == -1:
                for c in active_clauses:
                    sat = False
                    for lit in c:
                        v = abs(lit) - 1
                        if (assignment.get(v, False) and lit > 0) or (not assignment.get(v, True) and lit < 0):
                            sat = True
                            break
                    if not sat:
                        return None
                return [assignment.get(i, False) for i in range(n)]
            
            assignment[best_var] = True
            saved_clauses = [c[:] for c in active_clauses]
            saved_assignment = dict(assignment)
            
            result = dpll(depth + 1)
            if result is not None:
                return result
            
            assignment.clear()
            assignment.update(saved_assignment)
            del assignment[best_var]
            active_clauses[:] = [c[:] for c in saved_clauses]
            
            assignment[best_var] = False
            saved_clauses2 = [c[:] for c in active_clauses]
            saved_assignment2 = dict(assignment)
            
            result = dpll(depth + 1)
            if result is not None:
                return result
            
            assignment.clear()
            assignment.update(saved_assignment2)
            del assignment[best_var]
            active_clauses[:] = [c[:] for c in saved_clauses2]
            
            return None
        
        return dpll(0)
    
    dpll_result = dpll_limited(n, clauses, dpll_deadline)
    if dpll_result is not None:
        # Verify
        all_sat = True
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (dpll_result[v] and lit > 0) or (not dpll_result[v] and lit < 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return dpll_result
    
    # Spectral initialization via SVD of the incidence matrix
    # Build the incidence matrix M (num_clauses x n)
    M = np.zeros((num_clauses, n), dtype=np.float64)
    for i in range(num_clauses):
        for k in range(3):
            v = clause_vars[i][k]
            s = clause_signs[i][k]
            M[i, v] = 1.0 if s else -1.0
    
    # Compute SVD: M = U S V^T
    # The right singular vectors V give directions in variable space
    # For a planted 3-SAT, the true assignment x (as +1/-1 vector)
    # should be approximately aligned with the first few right singular vectors
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        # Vt has shape (min(num_clauses, n), n), rows are right singular vectors
        # Take the first few right singular vectors and combine them
        # The sign of the combination with the largest singular value gives an initial guess
        # Use a weighted combination of the top k right singular vectors
        k = min(5, Vt.shape[0])
        # Weight by singular values
        weights = S[:k]
        weights = weights / weights.sum()
        # Combine: x_spectral = sum_i w_i * Vt[i]
        x_spectral = np.zeros(n, dtype=np.float64)
        for i in range(k):
            x_spectral += weights[i] * Vt[i]
        
        # Convert to boolean assignment: True if x_spectral[v] > 0
        spectral_assignment = (x_spectral > 0).tolist()
    except:
        spectral_assignment = phase_list.copy()
    
    # Evaluate which initialization is better
    def count_unsat(assignment):
        unsat = 0
        for i in range(num_clauses):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for v, s in zip(cv, cs):
                if assignment[v] == s:
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat
    
    phase_unsat = count_unsat(phase_list)
    spectral_unsat = count_unsat(spectral_assignment)
    
    # Use the better initialization
    if spectral_unsat < phase_unsat:
        initial_assignment = spectral_assignment
        initial_unsat = spectral_unsat
    else:
        initial_assignment = phase_list
        initial_unsat = phase_unsat
    
    # WalkSAT with optimized clause tracking
    rng = random.Random(42)
    
    # Clause weights
    clause_weights = np.ones(num_clauses, dtype=np.float64)
    
    # Track unsat clauses incrementally
    # clause_satisfied[i] = True if clause i is satisfied under current assignment
    clause_satisfied = np.zeros(num_clauses, dtype=bool)
    clause_unsat_count = np.zeros(num_clauses, dtype=np.int32)  # number of unassigned/unmet literals
    
    def init_clause_status(assignment):
        for i in range(num_clauses):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            unmet = 0
            for v, s in zip(cv, cs):
                if assignment[v] == s:
                    sat = True
                    break
                else:
                    unmet += 1
            clause_satisfied[i] = sat
            clause_unsat_count[i] = unmet if not sat else 0
    
    def update_clause(i, old_val, new_val, var):
        """Update clause status after flipping var from old_val to new_val"""
        cv = clause_vars[i]
        cs = clause_signs[i]
        # Find position of var in clause
        for k in range(3):
            if cv[k] == var:
                old_sign = cs[k]
                old_met = (old_val == old_sign)
                new_met = (new_val == old_sign)
                break
        
        if clause_satisfied[i]:
            if old_met and not new_met:
                # Clause was satisfied only by this literal, now may be unsat
                # Check if any other literal satisfies
                other_sat = False
                for k2 in range(3):
                    if cv[k2] != var:
                        if assignment[cv[k2]] == cs[k2]:
                            other_sat = True
                            break
                if not other_sat:
                    clause_satisfied[i] = False
                    # Count unmet
                    unmet = 0
                    for k2 in range(3):
                        if not (assignment[cv[k2]] == cs[k2]):
                            unmet += 1
                    clause_unsat_count[i] = unmet
            # If old_met and new_met, or not old_met and new_met (but was already sat), no change
        else:
            if not old_met and new_met:
                # Clause becomes satisfied
                clause_satisfied[i] = True
                clause_unsat_count[i] = 0
            elif not old_met and not new_met:
                # Still unsat, count unchanged
                pass
            elif old_met and not new_met:
                # Wasn't satisfied before (contradiction, shouldn't happen)
                pass
    
    # Initialize clause status
    init_clause_status(initial_assignment)
    
    # Build unsat clause list
    unsat_clauses = [i for i in range(num_clauses) if not clause_satisfied[i]]
    unsat_count = len(unsat_clauses)
    
    best_assignment = initial_assignment.copy()
    best_unsat_count = unsat_count
    
    max_restarts = 300
    steps_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # For restarts > 0, randomize the assignment
        if restart == 0:
            assignment = initial_assignment.copy()
            # Reinitialize clause status
            init_clause_status(assignment)
            unsat_clauses = [i for i in range(num_clauses) if not clause_satisfied[i]]
            unsat_count = len(unsat_clauses)
        else:
            # Random assignment with bias toward phase
            assignment = []
            for v in range(n):
                if rng.random() < 0.7:
                    assignment.append(phase_list[v])
                else:
                    assignment.append(not phase_list[v])
            init_clause_status(assignment)
            unsat_clauses = [i for i in range(num_clauses) if not clause_satisfied[i]]
            unsat_count = len(unsat_clauses)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive noise: start high, decrease over restarts
        noise_prob = 0.20 * (0.92 ** restart) + 0.05
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment
                
            if unsat_count == 0:
                return assignment
                
            # Pick an unsat clause with probability proportional to weight
            if len(unsat_clauses) == 0:
                break
                
            if len(unsat_clauses) == 1:
                clause_idx = unsat_clauses[0]
            else:
                weights = np.array([clause_weights[i] for i in unsat_clauses], dtype=np.float64)
                total_weight = weights.sum()
                if total_weight <= 0:
                    clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
                else:
                    r = rng.random() * total_weight
                    cumulative = 0.0
                    clause_idx = unsat_clauses[-1]
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            clause_idx = unsat_clauses[i]
                            break
            
            cv = clause_vars[clause_idx]
            cs = clause_signs[clause_idx]
            
            # Evaluate flip for each literal in this clause
            best_var = -1
            best_delta = float('inf')
            
            for k in range(3):
                v = cv[k]
                delta = 0
                for c_idx in var_to_clauses[v]:
                    c_cv = clause_vars[c_idx]
                    c_cs = clause_signs[c_idx]
                    
                    # Find sign of v in this clause
                    v_sign = None
                    for j in range(3):
                        if c_cv[j] == v:
                            v_sign = c_cs[j]
                            break
                    if v_sign is None:
                        continue
                    
                    v_current_val = assignment[v]
                    v_satisfies_now = (v_current_val == v_sign)
                    
                    other_sat = False
                    for j in range(3):
                        if c_cv[j] != v:
                            if assignment[c_cv[j]] == c_cs[j]:
                                other_sat = True
                                break
                                
                    if not other_sat:
                        if v_satisfies_now:
                            delta += clause_weights[c_idx]
                        else:
                            delta -= clause_weights[c_idx]
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                elif delta == best_delta and rng.random() < 0.5:
                    best_var = v
            
            # Adaptive noise: with probability noise_prob, pick random var from clause
            if rng.random() < noise_prob:
                best_var = cv[rng.randint(0, 2)]
            
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            # Update clause statuses incrementally
            for c_idx in var_to_clauses[best_var]:
                old_sat = clause_satisfied[c_idx]
                update_clause(c_idx, old_val, new_val, best_var)
                new_sat = clause_satisfied[c_idx]
                
                if old_sat and not new_sat:
                    # Clause became unsat
                    unsat_count += 1
                    unsat_clauses.append(c_idx)
                    clause_weights[c_idx] *= 1.5
                elif not old_sat and new_sat:
                    # Clause became sat
                    unsat_count -= 1
                    # Remove from unsat_clauses (lazy deletion: mark and filter later)
                    clause_weights[c_idx] *= 0.7
                elif not old_sat and not new_sat:
                    # Still unsat, increase weight slightly
                    clause_weights[c_idx] *= 1.1
                else:
                    # Still sat, decrease weight slightly
                    clause_weights[c_idx] *= 0.9
            
            # Periodically clean up unsat_clauses list (remove satisfied clauses)
            if step % 100 == 0 and len(unsat_clauses) > 0:
                new_unsat = []
                for i in unsat_clauses:
                    if not clause_satisfied[i]:
                        new_unsat.append(i)
                    else:
                        # This was a stale entry
                        pass
                unsat_clauses = new_unsat
                # Recount to be safe
                actual_unsat = sum(1 for i in range(num_clauses) if not clause_satisfied[i])
                if actual_unsat != unsat_count:
                    unsat_count = actual_unsat
                    unsat_clauses = [i for i in range(num_clauses) if not clause_satisfied[i]]
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
        
        # Decay clause weights after each restart
        clause_weights *= 0.85
    
    return best_assignment
# EVOLVE-BLOCK-END
