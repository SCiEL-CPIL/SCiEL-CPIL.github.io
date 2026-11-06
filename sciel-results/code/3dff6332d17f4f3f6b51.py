<<<<<<< SEARCH
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280.0

    # Precompute clause variable sets for fast evaluation
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]

    # WalkSAT with adaptive noise and clause weighting
    assignment = [rng.random() < 0.5 for _ in range(n)]

    def is_satisfied(c_idx):
        c = clauses[c_idx]
        for i in range(len(c)):
            v = c[i]
            if assignment[abs(v) - 1] == (v > 0):
                return True
        return False

    def num_unsat(c_idx):
        c = clauses[c_idx]
        cnt = 0
        for i in range(len(c)):
            v = c[i]
            if assignment[abs(v) - 1] != (v > 0):
                cnt += 1
        return cnt

    def count_unsat_clauses():
        cnt = 0
        for ci in range(len(clauses)):
            if not is_satisfied(ci):
                cnt += 1
        return cnt

    # Try multiple restarts with increasing effort
    max_restarts = 50
    base_steps = 5000
    noise = 0.15

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reset assignment with slight bias
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Track clause weights for weighting selection
        clause_weights = [1.0] * len(clauses)

        steps = base_steps + restart * 2000
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break

            # Find unsatisfied clauses
            unsat_indices = []
            for ci in range(len(clauses)):
                if not is_satisfied(ci):
                    unsat_indices.append(ci)

            if not unsat_indices:
                return assignment

            # Weighted selection of a clause
            total_w = sum(clause_weights[ci] for ci in unsat_indices)
            r = rng.random() * total_w
            cum = 0.0
            chosen_ci = unsat_indices[0]
            for ci in unsat_indices:
                cum += clause_weights[ci]
                if r <= cum:
                    chosen_ci = ci
                    break

            # Choose literal to flip
            clause = clauses[chosen_ci]
            # Count how many literals are false (all are false since clause is unsat)
            # Pick a literal to flip
            # With some probability, pick a literal that appears in few other unsat clauses
            lit_indices = [abs(v) - 1 for v in clause]
            if rng.random() < noise:
                # Random flip
                var_idx = rng.choice(lit_indices)
            else:
                # Heuristic: pick variable that appears in fewest other unsat clauses
                best_var = lit_indices[0]
                best_score = float('inf')
                for vi in lit_indices:
                    score = 0
                    for other_ci in unsat_indices:
                        if other_ci != chosen_ci:
                            if vi in clause_vars[other_ci]:
                                score += 1
                    if score < best_score:
                        best_score = score
                        best_var = vi
                var_idx = best_var

            assignment[var_idx] = not assignment[var_idx]

            # Update clause weights
            # Decrease weight of satisfied clauses, increase unsat
            for ci in unsat_indices:
                if ci == chosen_ci:
                    # This clause is now potentially satisfied
                    if is_satisfied(ci):
                        clause_weights[ci] = max(0.5, clause_weights[ci] * 0.9)
                    else:
                        clause_weights[ci] = min(10.0, clause_weights[ci] * 1.1)
                else:
                    # Check if flipping made this clause more or less satisfied
                    # Simple: just decay weights slowly
                    clause_weights[ci] = min(10.0, clause_weights[ci] * 1.001)

            # Occasionally decay all weights
            if step % 1000 == 0:
                for ci in range(len(clauses)):
                    clause_weights[ci] = max(0.5, clause_weights[ci] * 0.99)

    # Final check
    for ci in range(len(clauses)):
        if not is_satisfied(ci):
            break
    else:
        return assignment

    return assignment
=======
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280.0
    
    # Precompute clause data
    m = len(clauses)
    # clause_vars[i] is a list of variable indices (0-based) for clause i
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    # clause_sigs[i] is a list of boolean flags: True if literal is positive
    clause_sigs = [[l > 0 for l in c] for c in clauses]
    
    # Helper to check if a clause is satisfied given current assignment
    def is_satisfied(ci):
        cv = clause_vars[ci]
        cs = clause_sigs[ci]
        for j in range(3):
            if assignment[cv[j]] == cs[j]:
                return True
        return False

    best_unsat = float('inf')
    best_assignment = None

    # WalkSAT parameters
    noise = 0.05
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize state
        unsat_count = 0
        unsat_indices = []
        # var_unsat_count[v] = number of unsatisfied clauses containing variable v
        var_unsat_count = [0] * n
        
        for ci in range(m):
            if not is_satisfied(ci):
                unsat_count += 1
                unsat_indices.append(ci)
                for v in clause_vars[ci]:
                    var_unsat_count[v] += 1
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
        # WalkSAT loop
        # Adjust steps based on remaining time or fixed budget per restart
        # We use a fixed budget per restart but break early if time is low
        steps = 10000
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                # Save current state if it's better
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment
            
            if unsat_count == 0:
                return assignment
            
            # Select a random unsatisfied clause
            # Using a list is O(1) for random choice
            ci = rng.choice(unsat_indices)
            
            # Select a variable to flip
            # With probability 'noise', pick random variable from clause
            # Otherwise, pick variable that appears in fewest other unsatisfied clauses
            if rng.random() < noise:
                v = rng.choice(clause_vars[ci])
            else:
                # Heuristic: minimize sum of var_unsat_count for variables in clause
                # Since all vars in clause are in at least 1 unsat clause (the current one),
                # we look for the one with lowest count.
                best_v = -1
                best_count = float('inf')
                for v in clause_vars[ci]:
                    c = var_unsat_count[v]
                    if c < best_count:
                        best_count = c
                        best_v = v
                v = best_v
            
            # Flip variable v
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val
            
            # Update unsat counts and lists
            # We need to update all clauses containing v
            # To do this efficiently, we should have a list of clauses for each variable
            # But we don't have that precomputed. 
            # For 150 vars, M is likely ~1000-2000. 
            # Scanning all clauses to find those with v is O(M).
            # Total cost per step: O(M) for update + O(1) for flip.
            # This is much faster than O(M^2).
            
            # Optimization: Precompute var_to_clauses
            # We will do this outside the restart loop for efficiency
            
            # Actually, let's just iterate through all clauses containing v.
            # We need var_to_clauses. Let's add it.
            
            pass # Placeholder for logic below
            
            # We'll restructure to use var_to_clauses
            
            # Update logic:
            # For each clause c containing v:
            #   Check if c was satisfied before.
            #   Check if c is satisfied now.
            #   Update unsat_count, unsat_indices, var_unsat_count for all vars in c.
            
            # This is getting complex to edit in-place without breaking structure.
            # I will replace the whole function body in the next block properly.
            
            break # Break out of this dummy loop to use the proper replacement below

    # Fallback
    return best_assignment if best_assignment else [False] * n
>>>>>>> REPLACE

Wait, the replacement above is a mess because I started thinking about `var_to_clauses` but didn't implement it. I need to provide a clean, complete replacement for the `solve` function.

