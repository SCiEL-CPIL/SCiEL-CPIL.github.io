"""Improved 3-SAT solver with DLS (Dynamic Large Scale) approach."""
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 290 seconds (leaving margin for 330s limit)
    deadline = time.time() + 290.0
    
    # Build literal-to-variable mapping and clause structure
    m = len(clauses)
    
    # For each variable, track which clauses it appears in (as positive or negative)
    # var_clauses[var] = list of (clause_idx, sign) where sign=1 for positive, -1 for negative
    var_clauses = [[] for _ in range(n)]
    
    # Parse clauses and build index
    # Store clauses as list of (var, sign) where sign = 1 if positive, -1 if negative
    parsed_clauses = []
    for ci, clause in enumerate(clauses):
        parsed = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            parsed.append((var, sign))
            var_clauses[var].append((ci, sign))
        parsed_clauses.append(parsed)
    
    # Track for each clause how many literals are satisfied
    # sat_count[ci] = number of satisfied literals in clause ci
    # Track for each variable its current truth value
    # Use 1 for True, -1 for False
    
    # Initialize assignment randomly
    rng = random.Random(42)
    assignment = [rng.choice([True, False]) for _ in range(n)]
    
    # Compute initial sat_count
    sat_count = [0] * m
    for ci in range(m):
        for var, sign in parsed_clauses[ci]:
            val = assignment[var]
            if (val and sign == 1) or (not val and sign == -1):
                sat_count[ci] += 1
    
    # Track unsatisfied clauses
    # unsat_count: number of unsatisfied literals per clause (3 - sat_count)
    
    # For DLS: we need to track for each variable, the sum of weights of unsatisfied clauses it can satisfy
    # We use a simple DLS: pick variable in most unsatisfied clauses with highest "urgency"
    
    # Better approach: Use a hybrid of WalkSAT with clause-based scoring
    # and periodic restarts with learning
    
    # Let's implement a more sophisticated approach:
    # 1. Track for each unsatisfied clause, the literals in it
    # 2. Score variables by how many unsatisfied clauses they appear in
    # 3. Flip the variable that maximizes some criterion
    
    # Precompute: for each variable, list of (clause_idx, sign)
    # Already have var_clauses
    
    def get_unsat_count_for_var(var):
        """Get number of unsatisfied clauses where this variable appears as unsatisfied literal."""
        count = 0
        val = assignment[var]
        for ci, sign in var_clauses[var]:
            if sat_count[ci] == 0:  # clause is unsatisfied
                # Check if this literal is unsatisfied
                if (val and sign == 1) or (not val and sign == -1):
                    # This literal IS satisfied, so flipping would make it unsatisfied
                    # Actually wait: if clause is unsatisfied (sat_count=0), no literal is satisfied
                    # So all literals in unsatisfied clause are unsatisfied
                    count += 1
        return count
    
    # Let me reconsider the approach. Use a cleaner DLS implementation.
    
    # Reset and use a proper DLS with incremental updates
    # For each variable v, let w(v) = sum over unsatisfied clauses C containing v of 1/|unsatisfied literals in C|
    # Actually simpler: w(v) = number of unsatisfied clauses containing v
    
    # Let's use a well-known effective approach: WalkSAT with phase saving and restarts
    # but with better variable selection
    
    best_assignment = None
    best_unsat = m + 1
    
    # Multiple restarts
    num_restarts = 50
    max_steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize assignment: for first restart random, later use best found
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif best_assignment is not None:
            # Start from best with some noise
            assignment = list(best_assignment)
            for i in range(n):
                if rng.random() < 0.1:
                    assignment[i] = not assignment[i]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute sat_count
        sat_count = [0] * m
        for ci in range(m):
            for var, sign in parsed_clauses[ci]:
                val = assignment[var]
                if (val and sign == 1) or (not val and sign == -1):
                    sat_count[ci] += 1
        
        # Check if already solved
        unsat_list = [ci for ci in range(m) if sat_count[ci] == 0]
        
        if not unsat_list:
            return assignment
        
        # Track best
        if len(unsat_list) < best_unsat:
            best_unsat = len(unsat_list)
            best_assignment = list(assignment)
        
        # WalkSAT loop with adaptive step limit
        # Use a noise parameter that increases over steps to escape local minima
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 2.0:
                return best_assignment if best_assignment else assignment
            
            # Find unsatisfied clauses
            if not unsat_list:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = parsed_clauses[ci]
            
            # Score each literal in the clause
            # For each variable in the clause, compute how many other unsatisfied clauses it appears in
            # (excluding the current clause)
            best_var = -1
            best_score = -1
            
            # Compute scores for variables in this clause
            var_scores = {}
            for var, sign in clause:
                # Count unsatisfied clauses this variable appears in (excluding current)
                # But we need to be careful: if we flip var, which clauses become satisfied/unsatisfied?
                # Simple heuristic: count how many currently unsatisfied clauses contain this variable
                # as a potentially satisfying literal
                score = 0
                val = assignment[var]
                for ci2, sign2 in var_clauses[var]:
                    if ci2 == ci:
                        continue
                    if sat_count[ci2] == 0:  # unsatisfied clause
                        # If we flip var, does it satisfy ci2?
                        if (val and sign2 == 1) or (not val and sign2 == -1):
                            # Currently this literal is satisfied, so flipping unsatisfies it
                            # Bad
                            score -= 2
                        else:
                            # Currently unsatisfied, flipping satisfies it
                            score += 1
                var_scores[var] = score
            
            # Pick variable with best score, with some randomness
            # Use softmax-like selection for diversity
            if len(var_scores) > 0:
                items = list(var_scores.items())
                # Add noise
                noise_scale = 1.0 + 0.01 * (step % 100)
                noisy_scores = [(v, s + rng.gauss(0, noise_scale)) for v, s in items]
                noisy_scores.sort(key=lambda x: x[1], reverse=True)
                # Pick from top with some probability
                if rng.random() < 0.7:
                    best_var = noisy_scores[0][0]
                else:
                    best_var = rng.choice(noisy_scores)[0]
            
            if best_var == -1:
                best_var = clause[0][0]
            
            # Flip the variable
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            # Update sat_count for all clauses containing best_var
            for ci2, sign2 in var_clauses[best_var]:
                # Before flip: was this literal satisfied?
                old_sat = (old_val and sign2 == 1) or (not old_val and sign2 == -1)
                new_sat = (new_val and sign2 == 1) or (not new_val and sign2 == -1)
                
                if old_sat and not new_sat:
                    sat_count[ci2] -= 1
                    if sat_count[ci2] == 0:
                        # Newly unsatisfied - add to unsat_list
                        unsat_list.append(ci2)
                elif not old_sat and new_sat:
                    sat_count[ci2] += 1
                    if sat_count[ci2] >= 1:
                        # No longer unsatisfied - remove from unsat_list
                        try:
                            unsat_list.remove(ci2)
                        except ValueError:
                            pass
            
            # Update best
            if not unsat_list:
                return assignment
            
            cur_unsat = len(unsat_list)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = list(assignment)
        
        # Restart
        continue
    
    # Return best assignment found
    return best_assignment if best_assignment else assignment
