"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure for fast updates
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if assignment[vi] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def update_clause_sat(assignment, var_idx, clause_idx):
        """Check if a clause is satisfied given current assignment."""
        cv = clause_vars[clause_idx]
        for vi, pos in cv:
            if assignment[vi] == pos:
                return True
        return False
    
    def flip_variable(assignment, var_idx):
        """Flip a variable and return the change in unsat count."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        
        delta = 0
        for ci, pos in var_clauses[var_idx]:
            cv = clause_vars[ci]
            # Check if clause was satisfied before flip
            was_sat = False
            for vi, p in cv:
                if vi == var_idx:
                    continue
                if assignment[vi] == p:
                    was_sat = True
                    break
            
            # Check if clause is satisfied after flip
            is_sat_now = False
            for vi, p in cv:
                if vi == var_idx:
                    if new_val == p:
                        is_sat_now = True
                        break
                else:
                    if assignment[vi] == p:
                        is_sat_now = True
                        break
            
            if was_sat and not is_sat_now:
                delta += 1
            elif not was_sat and is_sat_now:
                delta -= 1
        
        return delta
    
    def weighted_walksat(assignment, weights, max_steps, deadline):
        """Weighted WalkSAT with stagnation-based weight boosting."""
        rng = random.Random(int(time.time() * 1000) % 2**31)
        
        # Compute initial unsat count
        unsat_count = 0
        unsat_clauses = []
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if assignment[vi] == pos:
                    sat = True
                    break
            if not sat:
                unsat_count += 1
                unsat_clauses.append(ci)
        
        if unsat_count == 0:
            return True
        
        best_unsat = unsat_count
        stagnation = 0
        p = 0.55  # Walk probability
        min_p = 0.3
        max_p = 0.9
        stagnation_threshold = 1000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return True
            
            # Update best and check stagnation
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                stagnation = 0
            else:
                stagnation += 1
            
            # Adjust walk probability based on stagnation
            if stagnation > stagnation_threshold:
                p = min(max_p, p + 0.05)
            elif stagnation < 100:
                p = max(min_p, p - 0.01)
            
            # Choose an unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Decide: greedy flip or random walk
            if rng.random() < p:
                # Random walk: pick a random literal from the clause
                vi, pos = rng.choice(cv)
            else:
                # Greedy: pick the variable that minimizes the increase in unsat
                best_delta = float('inf')
                best_vi = cv[0][0]
                for vi, pos in cv:
                    # Estimate delta by counting how many other literals in this clause
                    # would be satisfied if we flip vi
                    # A flip of vi to pos satisfies this clause if no other literal is satisfied
                    # We want to minimize the net change
                    other_sat = 0
                    for vi2, pos2 in cv:
                        if vi2 == vi:
                            continue
                        if assignment[vi2] == pos2:
                            other_sat += 1
                    
                    # If we set vi to pos, this clause is satisfied (since vi=pos)
                    # The change depends on whether this clause was the only unsat one
                    # and whether flipping vi breaks other clauses
                    # Approximate: prefer variables that appear in fewer unsat clauses
                    # or that have more satisfied literals in their clauses
                    # Simple heuristic: count how many clauses vi appears in that are unsat
                    delta_est = 0
                    for ci2, pos2 in var_clauses[vi]:
                        cv2 = clause_vars[ci2]
                        sat2 = False
                        for vi3, p3 in cv2:
                            if vi3 == vi:
                                continue
                            if assignment[vi3] == p3:
                                sat2 = True
                                break
                        if not sat2:
                            delta_est += 1
                    
                    if delta_est < best_delta:
                        best_delta = delta_est
                        best_vi = vi
                
                vi = best_vi
            
            # Flip the variable
            old_val = assignment[vi]
            new_val = not old_val
            assignment[vi] = new_val
            
            # Update unsat count and unsat_clauses list
            # We need to update all clauses containing vi
            new_unsat_clauses = []
            for ci2, cv2 in enumerate(clause_vars):
                # Check if clause contains vi
                contains_vi = any(vi3 == vi for vi3, _ in cv2)
                if contains_vi:
                    # Re-evaluate this clause
                    sat = False
                    for vi3, p3 in cv2:
                        if assignment[vi3] == p3:
                            sat = True
                            break
                    if not sat:
                        new_unsat_clauses.append(ci2)
                else:
                    # Clause unchanged, keep if it was unsat
                    pass
            
            # Rebuild unsat_clauses properly
            unsat_clauses = []
            for ci2, cv2 in enumerate(clause_vars):
                sat = False
                for vi3, p3 in cv2:
                    if assignment[vi3] == p3:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci2)
            
            unsat_count = len(unsat_clauses)
        
        return unsat_count == 0
    
    # Multi-restart strategy
    best_assignment = None
    best_score = m + 1
    
    while time.time() < deadline:
        # Initialize random assignment
        rng = random.Random()
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize weights (all 1.0)
        weights = [1.0] * m
        
        # Run weighted WalkSAT
        max_steps = 50000
        found = weighted_walksat(assignment, weights, max_steps, deadline)
        
        if found:
            return assignment
        
        # Track best solution
        current_unsat = 0
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if assignment[vi] == pos:
                    sat = True
                    break
            if not sat:
                current_unsat += 1
        
        if current_unsat < best_score:
            best_score = current_unsat
            best_assignment = assignment[:]
        
        # If we've found a near-solution, try to polish
        if best_score <= 5 and time.time() < deadline:
            # Local search on the best assignment
            polish_rng = random.Random()
            for _ in range(10000):
                if time.time() > deadline:
                    break
                if best_score == 0:
                    break
                
                # Find an unsatisfied clause
                unsat_clauses = []
                for ci, cv in enumerate(clause_vars):
                    sat = False
                    for vi, pos in cv:
                        if best_assignment[vi] == pos:
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                
                if not unsat_clauses:
                    best_score = 0
                    break
                
                ci = polish_rng.choice(unsat_clauses)
                cv = clause_vars[ci]
                vi = polish_rng.choice(cv)[0]
                
                # Flip
                best_assignment[vi] = not best_assignment[vi]
                
                # Recount
                new_unsat = 0
                for ci2, cv2 in enumerate(clause_vars):
                    sat = False
                    for vi3, p3 in cv2:
                        if best_assignment[vi3] == p3:
                            sat = True
                            break
                    if not sat:
                        new_unsat += 1
                
                if new_unsat < best_score:
                    best_score = new_unsat
                else:
                    # Revert
                    best_assignment[vi] = not best_assignment[vi]
        
        if best_score == 0:
            return best_assignment
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
