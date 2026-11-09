"""Planted 3-SAT solver using WalkSAT with efficient clause tracking and DLS restarts."""
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # safety margin before 300s kill

    m = len(clauses)
    
    # Precompute clause literals as (var_idx, polarity) for faster access
    # var_idx is 0-based, polarity is True for positive, False for negative
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    
    # Build variable-to-clause index for efficient impact calculation
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_opt):
        for var_idx, pol in c:
            var_to_clauses[var_idx].append(ci)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        unsat = 0
        for c in clauses_opt:
            sat = False
            for var_idx, pol in c:
                if assignment[var_idx] == pol:
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for c in clauses_opt:
            sat = False
            for var_idx, pol in c:
                if assignment[var_idx] == pol:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # Try a few random assignments first
    for trial in range(10):
        if time.time() > deadline:
            break
        rng = random.Random(trial * 987654321 + 111)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        uc = count_unsat(assignment)
        if uc == 0:
            return assignment
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]
    
    # WalkSAT with DLS (Dynamic Large-Sum) restart strategy
    # DLS: restart when the number of flips since last restart exceeds a threshold
    # based on the current unsat count
    
    max_restarts = 200
    base_steps = 10000
    noise_prob = 0.25
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 123456789 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track which clauses are unsatisfied
        unsat_set = set()
        for ci, c in enumerate(clauses_opt):
            sat = False
            for var_idx, pol in c:
                if assignment[var_idx] == pol:
                    sat = True
                    break
            if not sat:
                unsat_set.add(ci)
        
        uc = len(unsat_set)
        if uc == 0:
            return assignment
        
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]
        
        # DLS: steps allowed is proportional to unsat count
        # More unsat -> more steps before restart
        steps_allowed = max(500, int(uc * 50 + 500))
        
        for step in range(steps_allowed):
            if time.time() > deadline:
                return best_assignment if best_assignment is not None else assignment
            
            if not unsat_set:
                return assignment
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            clause_idx = next(iter(unsat_set))
            clause = clauses_opt[clause_idx]
            
            # Score each variable in the clause:
            # For each variable, compute how many other unsat clauses would be satisfied if we flip it
            best_var = None
            best_score = float('inf')
            
            for var_idx, pol in clause:
                # If we flip this variable:
                # - This clause becomes satisfied (the literal becomes true)
                # - Other clauses containing this variable may be affected
                
                # Count how many currently unsat clauses (other than this one) 
                # would be satisfied by this flip
                score = 0
                for other_ci in var_to_clauses[var_idx]:
                    if other_ci == clause_idx:
                        continue
                    if other_ci in unsat_set:
                        other_clause = clauses_opt[other_ci]
                        # Check if flipping var_idx would satisfy this other clause
                        # The literal for var_idx in other_clause:
                        for v, p in other_clause:
                            if v == var_idx:
                                # After flip, assignment[var_idx] becomes not assignment[var_idx]
                                if (not assignment[var_idx]) == p:
                                    score += 1
                                break
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                continue
            
            # With some probability, pick a random variable from the clause (noise)
            if rng.random() < noise_prob:
                var_idx, pol = clause[rng.randrange(len(clause))]
            else:
                var_idx = best_var
            
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_set:
            # Clauses containing var_idx may change status
            for ci in var_to_clauses[var_idx]:
                c = clauses_opt[ci]
                # Check if clause is now satisfied
                sat = False
                for v, p in c:
                    if assignment[v] == p:
                        sat = True
                        break
                
                if sat:
                    unsat_set.discard(ci)
                else:
                    # Check if it was previously satisfied (need to re-add if not in unsat_set)
                    if ci not in unsat_set:
                        # It was satisfied before, now it's unsat
                        unsat_set.add(ci)
                    # else it was already unsat, stays unsat
            
            if not unsat_set:
                return assignment
        
        # After steps_allowed steps, restart with DLS
        # The restart is implicit in the outer loop
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [False] * n
