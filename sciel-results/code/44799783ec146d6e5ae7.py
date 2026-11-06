"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: leave 5s safety margin from 330s limit
    deadline = time.time() + 325.0
    
    # Precompute clause indices for faster access
    # var_to_clauses[var] = list of (clause_idx, literal_in_clause)
    # literal_in_clause is the signed literal as it appears in the clause
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append((i, lit))
            
    def is_satisfied(assignment):
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def walk_sat(assignment, max_steps, rng, p):
        """
        Optimized WalkSAT with incremental updates.
        """
        # Initialize clause satisfaction counts
        clause_sat_count = [0] * m
        unsat_clauses = []
        
        for i, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_clauses.append(i)
                
        if not unsat_clauses:
            return True
            
        # Use a set for O(1) lookup of unsatisfied clauses if we need to remove/add frequently
        # But for simplicity and memory, a list with linear search or a set is better.
        # Given m is likely small (3-SAT, 50 vars -> ~150-200 clauses), list is fine.
        # However, removing from a list is O(m). Let's use a set for unsat_clauses indices.
        unsat_set = set(unsat_clauses)
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False
                
            if not unsat_set:
                return True
                
            # Pick a random unsatisfied clause
            c_idx = next(iter(unsat_set)) # Arbitrary pick from set, or rng.choice(list(unsat_set))
            # To be strictly random:
            unsat_list = list(unsat_set)
            c_idx = rng.choice(unsat_list)
            
            clause = clauses[c_idx]
            
            # Evaluate flips for literals in this clause
            best_gain = -100
            best_var = -1
            flip_candidates = []
            
            for lit in clause:
                var = abs(lit) - 1
                current_val = assignment[var]
                desired_val = (lit > 0)
                
                # If the literal is already satisfied, flipping it would make it unsatisfied.
                # In standard WalkSAT, we usually consider flipping any variable in the clause.
                # But if it's already satisfied, the gain is likely negative.
                # Let's calculate gain for all variables in the clause.
                
                gain = 0
                # Flipping 'var' will change the satisfaction of 'c_idx'.
                # If current_val == desired_val, it was contributing 1 to sat_count. Flipping makes it 0.
                # If current_val != desired_val, it was contributing 0. Flipping makes it 1.
                if current_val == desired_val:
                    # We are breaking a satisfied literal in the target clause
                    # The clause c_idx will lose 1 satisfaction.
                    # But wait, if c_idx is unsat, it has 0 satisfied literals.
                    # So current_val != desired_val MUST be true for all literals in an unsat clause?
                    # No, an unsat clause has 0 satisfied literals.
                    # So for every lit in an unsat clause, assignment[var] != (lit > 0).
                    # Therefore, flipping ANY var in the clause will satisfy that literal, increasing sat_count of c_idx by 1.
                    # So gain starts at +1 for the target clause.
                    pass # This branch shouldn't happen if c_idx is truly unsat and we only pick unsat clauses.
                    
                # Since c_idx is unsat, all its literals are currently unsatisfied.
                # Flipping 'var' will satisfy 'lit' in c_idx. Gain += 1.
                gain = 1
                
                # Check impact on other clauses containing 'var'
                for (ci2, lit2) in var_to_clauses[var]:
                    if ci2 == c_idx:
                        continue
                    
                    # Determine if flipping 'var' changes the satisfaction of ci2
                    # Current state of lit2 in ci2:
                    var2 = abs(lit2) - 1 # Same as var
                    is_sat2 = assignment[var] == (lit2 > 0)
                    
                    if is_sat2:
                        # Currently satisfied. Flipping 'var' will make it unsatisfied.
                        # We need to know if it was the ONLY satisfied literal in ci2.
                        if clause_sat_count[ci2] == 1:
                            # It becomes unsatisfied
                            gain -= 1
                    else:
                        # Currently unsatisfied. Flipping 'var' will make it satisfied.
                        # Check if it was the ONLY unsatisfied literal? No, if it becomes satisfied,
                        # does the clause become satisfied?
                        # If clause_sat_count[ci2] == 0 (unsat), and we satisfy one literal, it becomes sat.
                        # If clause_sat_count[ci2] > 0, it was already sat, and remains sat.
                        if clause_sat_count[ci2] == 0:
                            # It becomes satisfied
                            gain += 1
                        # Else: it was already satisfied (by other literals), and remains satisfied. No change.
                    
                flip_candidates.append((var, gain))
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            # Select variable to flip
            if rng.random() < p:
                # Random flip
                var = flip_candidates[rng.randrange(len(flip_candidates))][0]
            else:
                # Best gain flip
                var = best_var
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update clause_sat_count and unsat_set
            for (ci2, lit2) in var_to_clauses[var]:
                old_count = clause_sat_count[ci2]
                
                # Determine new satisfaction status of lit2
                # assignment[var] has just changed.
                new_is_sat = assignment[var] == (lit2 > 0)
                old_is_sat = not new_is_sat # Since we flipped it
                
                if new_is_sat and not old_is_sat:
                    # Became satisfied
                    clause_sat_count[ci2] = old_count + 1
                    if old_count == 0:
                        # Was unsat, now sat
                        unsat_set.discard(ci2)
                elif not new_is_sat and old_is_sat:
                    # Became unsatisfied
                    clause_sat_count[ci2] = old_count - 1
                    if old_count == 1:
                        # Was sat (only this literal), now unsat
                        unsat_set.add(ci2)
                # Else: no change in satisfaction count (either both sat or both unsat, which is impossible for a single literal flip? 
                # Wait. If a clause has 2 satisfied literals, and we flip one to unsat, count goes 2->1. 
                # If a clause has 0 satisfied literals, and we flip one to sat, count goes 0->1.
                # The logic above handles the transitions that affect the unsat_set (0<->1).
                # If old_count > 1 and new_is_sat=False, count decreases but stays > 0, so not added to unsat_set. Correct.
                # If old_count == 0 and new_is_sat=True, count increases to 1, removed from unsat_set. Correct.
                # If old_count == 1 and new_is_sat=False, count decreases to 0, added to unsat_set. Correct.
                # If old_count > 0 and new_is_sat=True, count increases, not added. Correct.
            
        return not unsat_set

    # Strategy: Multiple restarts with varying parameters
    # For planted SAT, typical ratio is near threshold.
    # Base steps per restart
    base_steps = max(1000, n * 50)
    
    restart_count = 0
    # Vary p slightly to escape local minima
    p_values = [0.2, 0.3, 0.5, 0.6, 0.8]
    
    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(restart_count * 1000003 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if is_satisfied(assignment):
            return assignment
            
        p = p_values[restart_count % len(p_values)]
        
        # Run WalkSAT
        if walk_sat(assignment, base_steps, rng, p):
            return assignment
            
    # Fallback: return last assignment or random
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
