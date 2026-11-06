import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 290 seconds (safety margin for 300s limit)
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-compute clause literal arrays for faster access
    # Each clause stored as list of (var_idx, sign) where sign is +1 or -1
    clause_vars = []  # clause_vars[i] = list of (var_idx, sign)
    for clause in clauses:
        cv = []
        for l in clause:
            var_idx = abs(l) - 1
            sign = 1 if l > 0 else -1
            cv.append((var_idx, sign))
        clause_vars.append(cv)
    
    m = len(clauses)
    
    # Build variable-to-clause mapping: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, sign in cv:
            var_to_clauses[var_idx].append((i, sign))
    
    # Precompute literal satisfaction: lit_sat[clause_idx][var_idx] = sign
    # Actually, we'll compute clause satisfaction incrementally
    
    def is_clause_satisfied(clause_idx, assignment):
        """Check if a single clause is satisfied."""
        cv = clause_vars[clause_idx]
        for var_idx, sign in cv:
            if assignment[var_idx] == (sign > 0):
                return True
        return False
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for i in range(m):
            if not is_clause_satisfied(i, assignment):
                cnt += 1
        return cnt
    
    def get_unsat_clause_indices(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for i in range(m):
            if not is_clause_satisfied(i, assignment):
                unsat.append(i)
        return unsat
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    max_steps_per_run = 3000
    max_restarts = 2000
    
    rng = random.Random(0)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Local search with adaptive WalkSAT
        steps = 0
        last_improvement_step = 0
        current_unsat = unsat_count
        
        while steps < max_steps_per_run:
            if time.time() > deadline:
                return best_assignment if best_assignment is not None else assignment
                
            if current_unsat == 0:
                return assignment
                
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                last_improvement_step = steps
                
            # Stagnation check: if no improvement for 800 steps, break and restart
            if steps - last_improvement_step > 800:
                break
                
            # Get unsatisfied clauses
            unsat_idx = get_unsat_clause_indices(assignment)
            if not unsat_idx:
                return assignment
                
            # Adaptive WalkSAT parameter: increase randomization when stuck
            stagnation = steps - last_improvement_step
            p = min(0.8, 0.3 + 0.1 * (stagnation / 100.0))
            
            # Score each variable by how many unsatisfied clauses it appears in
            # and prefer flipping variables that would satisfy more clauses
            var_score = {}
            for ci in unsat_idx:
                cv = clause_vars[ci]
                for var_idx, sign in cv:
                    if var_idx not in var_score:
                        var_score[var_idx] = 0
                    # If flipping this variable would satisfy the clause, give it a bonus
                    current_val = assignment[var_idx]
                    would_satisfy = (current_val != (sign > 0))
                    var_score[var_idx] += 1 + (2 if would_satisfy else 0)
            
            if not var_score:
                break
                
            # Find max score
            max_score = max(var_score.values())
            candidates = [v for v, s in var_score.items() if s == max_score]
            
            # With probability p, pick a random variable from unsat clauses (WalkSAT)
            # With probability 1-p, pick the best-scored variable (Min-Conflicts)
            if rng.random() < p:
                # Pick random variable from a random unsat clause
                ci = rng.choice(unsat_idx)
                cv = clause_vars[ci]
                var_idx, sign = rng.choice(cv)
            else:
                # Pick best variable
                var_idx = rng.choice(candidates)
                
            # Flip variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Incrementally update unsat count
            # Only clauses containing var_idx can change status
            delta = 0
            for ci, sign in var_to_clauses[var_idx]:
                was_sat = is_clause_satisfied(ci, assignment) if old_val else False
                # We need to check before and after flip
                # Since we already flipped, let's check current state
                # and compare with what it was before
            
            # Simpler: just recount affected clauses
            new_unsat = 0
            for ci, sign in var_to_clauses[var_idx]:
                if not is_clause_satisfied(ci, assignment):
                    new_unsat += 1
            
            # But we need to account for clauses that were already unsat and remain unsat
            # Easiest for N=50: just recount all
            current_unsat = count_unsat(assignment)
            steps += 1
            
    # If loop finishes without finding solution, return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
