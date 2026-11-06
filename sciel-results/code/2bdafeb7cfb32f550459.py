"""Optimized WalkSAT with incremental clause tracking and adaptive noise."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Precompute: for each variable, list of (clause_idx, sign) where sign=+1 means lit is var
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[vi].append((ci, sign))
    
    # Build clause -> literals as (var_idx, sign) tuples for fast access
    clause_lits = []
    for clause in clauses:
        lits = []
        for lit in clause:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            lits.append((vi, sign))
        clause_lits.append(lits)
    
    def init_state(assignment):
        """Compute clause satisfaction counts and unsat count from scratch."""
        clause_sat_count = [0] * m
        unsat_count = 0
        for ci, lits in enumerate(clause_lits):
            cnt = 0
            for vi, sign in lits:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat_count += 1
        return clause_sat_count, unsat_count
    
    def flip_var(assignment, vi, clause_sat_count, unsat_count):
        """Flip variable vi and update clause_sat_count and unsat_count incrementally."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        
        delta_unsat = 0
        for ci, sign in var_clauses[vi]:
            old_lit_satisfied = (old_val == (sign == 1))
            new_lit_satisfied = (new_val == (sign == 1))
            
            old_sat = clause_sat_count[ci]
            
            if old_lit_satisfied and not new_lit_satisfied:
                clause_sat_count[ci] = old_sat - 1
                if old_sat == 1:
                    delta_unsat += 1
            elif not old_lit_satisfied and new_lit_satisfied:
                clause_sat_count[ci] = old_sat + 1
                if old_sat == 0:
                    delta_unsat -= 1
        
        return unsat_count + delta_unsat
    
    def compute_delta(assignment, vi, clause_sat_count):
        """Compute change in unsat count if we flip variable vi, without modifying state."""
        old_val = assignment[vi]
        new_val = not old_val
        delta = 0
        
        for ci, sign in var_clauses[vi]:
            old_lit_sat = (old_val == (sign == 1))
            new_lit_sat = (new_val == (sign == 1))
            old_sat = clause_sat_count[ci]
            
            if old_lit_sat and not new_lit_sat:
                if old_sat == 1:
                    delta += 1
            elif not old_lit_sat and new_lit_sat:
                if old_sat == 0:
                    delta -= 1
        
        return delta
    
    def local_search(assignment, clause_sat_count, unsat_count, deadline):
        """Run WalkSAT local search with incremental updates."""
        rng = random.Random()
        
        max_steps = 5000
        stall_limit = 300
        stall_counter = 0
        best_local = unsat_count
        best_local_assign = assignment[:]
        
        for step in range(max_steps):
            if time.time() >= deadline:
                return None, best_local, best_local_assign
            
            if unsat_count == 0:
                return assignment, 0, assignment
            
            if unsat_count < best_local:
                best_local = unsat_count
                best_local_assign = assignment[:]
                stall_counter = 0
            else:
                stall_counter += 1
            
            if stall_counter > stall_limit:
                break
            
            # Find an unsatisfied clause
            ci = rng.randrange(m)
            found = False
            for _ in range(20):
                if clause_sat_count[ci] == 0:
                    found = True
                    break
                ci = rng.randrange(m)
            
            if not found:
                unsat_indices = [i for i in range(m) if clause_sat_count[i] == 0]
                if not unsat_indices:
                    return assignment, 0, assignment
                ci = rng.choice(unsat_indices)
            
            # Choose variable to flip from this clause using delta heuristic
            lits = clause_lits[ci]
            
            best_var = -1
            best_score = float('inf')
            
            # Adaptive noise
            noise_param = 0.3 + 0.7 * min(stall_counter / stall_limit, 1.0)
            
            for vi, sign in lits:
                delta = compute_delta(assignment, vi, clause_sat_count)
                score = delta + rng.random() * noise_param
                
                if score < best_score:
                    best_score = score
                    best_var = vi
            
            if best_var >= 0:
                unsat_count = flip_var(assignment, best_var, clause_sat_count, unsat_count)
        
        return best_local_assign, best_local, best_local_assign
    
    # Main search loop with restarts
    best_unsat = float('inf')
    best_assignment = None
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        rng_seed = attempt * 7919 + 42
        rng_init = random.Random(rng_seed)
        assignment = [rng_init.random() < 0.5 for _ in range(n)]
        
        clause_sat_count, unsat_count = init_state(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # Run local search
        result, local_best, local_assign = local_search(assignment, clause_sat_count, unsat_count, deadline)
        
        if result is not None and local_best == 0:
            return local_assign
        
        if local_best < best_unsat:
            best_unsat = local_best
            best_assignment = local_assign[:]
        
        if best_unsat == 0:
            return best_assignment
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
