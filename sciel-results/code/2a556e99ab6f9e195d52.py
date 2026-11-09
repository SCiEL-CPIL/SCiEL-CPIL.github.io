"""Rewritten program using Weighted Min-Conflicts with Restarts and Phase Bias."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Time limit guard: 300s limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-process clauses for faster access
    # Store clauses as lists of (var, polarity) where var is 0-indexed
    # polarity: +1 if literal is positive, -1 if negative
    preprocessed_clauses = []
    for c in clauses:
        pre_c = []
        for lit in c:
            var = abs(lit) - 1
            pol = 1 if lit > 0 else -1
            pre_c.append((var, pol))
        preprocessed_clauses.append(pre_c)
        
    # Build variable index: list of (clause_idx, polarity) for each variable
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(preprocessed_clauses):
        for var, pol in c:
            var_clauses[var].append((i, pol))
            
    # Helper to check if a clause is satisfied
    # Returns True if satisfied, False otherwise
    def is_satisfied(clause_idx, assignment):
        c = preprocessed_clauses[clause_idx]
        for var, pol in c:
            if (assignment[var] == 1) == (pol == 1):
                return True
        return False
        
    # State: assignment is list of 0s and 1s
    # false_counts[var] = number of unsatisfied clauses where this variable is "critical" or just involved?
    # In Min-Conflicts, we usually count how many constraints are violated.
    # For 3-SAT, a clause is violated if all literals are false.
    # Flipping a variable v changes the status of all clauses containing v.
    
    # We will use a "Weighted WalkSAT" approach.
    # weight[var] = number of unsatisfied clauses containing var.
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Phase history for biasing restarts
    # If a variable was set to 1 in the last few failed attempts, bias towards 1
    phase_history = [0] * n  # 0 means neutral, +1 bias to 1, -1 bias to 0
    
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialization
        # Use phase bias: if phase_history[v] > 0, init to 1; if < 0, init to 0; else random
        assignment = [0] * n
        for v in range(n):
            if phase_history[v] > 0:
                assignment[v] = 1
            elif phase_history[v] < 0:
                assignment[v] = 0
            else:
                assignment[v] = 1 if random.random() < 0.5 else 0
                
        # Calculate initial unsat count and weights
        # unsat_clauses: set of clause indices that are not satisfied
        # weights: array of size n, counts how many unsat clauses contain this var
        
        unsat_clauses = set()
        weights = [0] * n
        
        for i, c in enumerate(preprocessed_clauses):
            sat = False
            for var, pol in c:
                if (assignment[var] == 1) == (pol == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.add(i)
                for var, pol in c:
                    weights[var] += 1
                    
        if not unsat_clauses:
            return [bool(x) for x in assignment]
            
        current_unsat = len(unsat_clauses)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # Search loop
        no_improve_count = 0
        max_no_improve = 2000 # Restart if stuck for this many steps
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if not unsat_clauses:
                return [bool(x) for x in assignment]
                
            # Update best
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break # Break to restart
            
            # Select a random unsatisfied clause
            clause_idx = random.choice(list(unsat_clauses))
            c = preprocessed_clauses[clause_idx]
            
            # Heuristic: Pick a variable from this clause with high weight
            # Or just random? Weighted is better.
            # Let's pick the variable with the highest weight in this clause
            best_var = -1
            best_weight = -1
            for var, pol in c:
                if weights[var] > best_weight:
                    best_weight = weights[var]
                    best_var = var
            
            # Random tie-breaking or slight randomness to escape local minima
            if random.random() < 0.3:
                # Random choice from clause
                var, pol = random.choice(c)
            else:
                var = best_var
                
            # Flip the variable
            old_val = assignment[var]
            new_val = 1 - old_val
            assignment[var] = new_val
            
            # Update unsat_clauses and weights
            # Clauses affected are those in var_clauses[var]
            for c_idx, pol in var_clauses[var]:
                # Check status before flip? No, we need to know if it was unsat before
                # and if it is unsat after.
                
                # Let's re-evaluate the clause status
                # Efficient way:
                # If clause was unsat, flipping a var might satisfy it (if the flipped var now satisfies it)
                # or keep it unsat.
                # If clause was sat, flipping a var might make it unsat (if the flipped var was the only satisfying one).
                
                # We need to know the state of the clause BEFORE the flip for variables other than var?
                # Actually, it's easier to just check the clause status after flip and compare to before.
                
                # Optimization:
                # We maintain a count of true literals in each clause?
                # Let's try a simpler update:
                # For each clause containing var:
                #   old_sat = is_satisfied(c_idx, assignment_with_old_val) -> We don't have old assignment easily.
                #   Let's just re-check satisfaction for the affected clauses.
                
                # To avoid full re-check, we can track true_count per clause.
                # But for now, simple re-check of the 3 literals is fast enough (3 ops).
                
                # Let's implement true_count tracking for efficiency.
                pass # Placeholder, let's use the simple re-check for clarity first, then optimize if needed.
                
                # Actually, let's stick to the simple re-check but only for affected clauses.
                # It's O(3) per clause. Number of clauses per var is small (avg 3000*3/150 = 60).
                # So ~180 checks per flip. 5000 flips * 180 = 900k checks. Very fast.
                
                was_unsat = c_idx in unsat_clauses
                now_sat = False
                for v, p in preprocessed_clauses[c_idx]:
                    if (assignment[v] == 1) == (p == 1):
                        now_sat = True
                        break
                
                if was_unsat and now_sat:
                    unsat_clauses.remove(c_idx)
                    # Decrement weights for all vars in this clause
                    for v, p in preprocessed_clauses[c_idx]:
                        weights[v] -= 1
                elif was_unsat and not now_sat:
                    # Still unsat, weights don't change for this clause
                    pass
                elif not was_unsat and now_sat:
                    # Still sat, weights don't change
                    pass
                elif not was_unsat and not now_sat:
                    # Became unsat
                    unsat_clauses.add(c_idx)
                    # Increment weights for all vars in this clause
                    for v, p in preprocessed_clauses[c_idx]:
                        weights[v] += 1
                        
            # End of step
            
        # End of search loop for this restart
        
        # Update phase history
        # If we found a solution, we would have returned.
        # If we didn't, bias towards the values in the best assignment found in this restart?
        # Or just bias towards the current assignment?
        # Let's bias towards the current assignment's values for variables that were stable?
        # Simple heuristic: add current assignment to phase history
        for v in range(n):
            phase_history[v] += 1 if assignment[v] == 1 else -1
            
    # Return best assignment found
    if best_assignment:
        return [bool(x) for x in best_assignment]
    else:
        # Fallback: random assignment
        return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
