import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin
    
    # Pre-process: variable occurrences
    # var_occurrences[v] = list of (clause_idx, literal)
    var_occurrences = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append((i, l))
    
    # Helper: compute initial clause_sat_count and unsat_clauses set
    def init_state(assignment):
        counts = [0] * len(clauses)
        unsat_set = set()
        for i, c in enumerate(clauses):
            count = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    count += 1
            counts[i] = count
            if count == 0:
                unsat_set.add(i)
        return counts, unsat_set

    # Helper: check if satisfied
    def is_satisfied(unsat_set):
        return len(unsat_set) == 0

    # Helper: apply flip and update state
    # Returns nothing, modifies assignment, counts, unsat_set
    def apply_flip(v, assignment, counts, unsat_set):
        assignment[v] = not assignment[v]
        for ci, lit in var_occurrences[v]:
            # Determine if the literal is satisfied AFTER the flip
            # Since we just flipped v, we can check the new state
            v_val = assignment[v]
            if (lit > 0 and v_val) or (lit < 0 and not v_val):
                # Literal is now satisfied
                old_count = counts[ci]
                new_count = old_count + 1
                counts[ci] = new_count
                if old_count == 0:
                    unsat_set.discard(ci)
            else:
                # Literal is now unsatisfied
                old_count = counts[ci]
                new_count = old_count - 1
                counts[ci] = new_count
                if old_count == 1 and new_count == 0:
                    unsat_set.add(ci)

    rng = random.Random(42)
    
    # Global best tracking
    best_assignment = None
    best_unsat_count = float('inf')
    
    # ILS (Iterated Local Search) parameters
    max_steps_per_run = 2000
    perturb_size = 10 # Number of random flips for perturbation
    
    # Main loop
    while time.time() < deadline:
        # 1. Initialize or Perturb
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # With high probability, perturb the best solution (ILS)
            # With low probability, start fresh
            if rng.random() < 0.8:
                assignment = best_assignment[:]
                # Perturb
                for _ in range(perturb_size):
                    v = rng.randint(0, n - 1)
                    apply_flip(v, assignment, *init_state(assignment)) # This is inefficient, re-init below
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]
                
        counts, unsat_set = init_state(assignment)
        
        # Check if satisfied
        if is_satisfied(unsat_set):
            return assignment
            
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # Local Search Phase
        steps = 0
        noise = 0.3 # Initial noise
        steps_since_improvement = 0
        run_best_unsat = current_unsat
        run_best_assignment = assignment[:]
        
        while steps < max_steps_per_run and time.time() < deadline:
            if is_satisfied(unsat_set):
                return assignment
            
            # Pick a random unsatisfied clause
            if not unsat_set:
                return assignment
                
            clause_idx = rng.choice(list(unsat_set))
            clause = clauses[clause_idx]
            
            # Adaptive Noise
            # If we haven't improved in a while, increase noise to escape local minima
            if steps_since_improvement > 50:
                noise = min(0.8, noise + 0.05)
            elif steps_since_improvement > 10:
                noise = max(0.1, noise - 0.01)
                
            # Heuristic: Min-Conflicts
            # Calculate score for each literal in the clause
            # Score = number of clauses that become unsatisfied if we flip this variable
            # We want to minimize this.
            
            best_v = None
            best_score = float('inf')
            best_tie_breaker = 0
            
            for l in clause:
                v = abs(l) - 1
                # Calculate the change in unsat count for all clauses involving v
                # Note: We must be careful not to double count or miss the clause itself
                # Actually, standard WalkSAT calculates:
                # newly_unsat = number of clauses containing v that are currently satisfied (count==1) 
                #               and will become unsatisfied (count==0) after flip?
                # No, that's not right.
                
                # Let's use the standard definition:
                # If we flip v, for each clause ci containing v:
                # If the literal in ci is currently satisfied, it becomes unsatisfied.
                #   If count[ci] was 1, it becomes 0 (newly unsat).
                #   If count[ci] was >1, it stays sat.
                # If the literal in ci is currently unsatisfied, it becomes satisfied.
                #   If count[ci] was 0, it becomes 1 (newly sat).
                #   If count[ci] was >0, it stays unsat (but count increases).
                
                # We want to minimize the number of NEWLY UNSAT clauses.
                newly_unsat = 0
                newly_sat = 0
                
                for ci, lit in var_occurrences[v]:
                    old_count = counts[ci]
                    # Check if literal is currently satisfied
                    v_val = assignment[v]
                    is_sat = (lit > 0 and v_val) or (lit < 0 and not v_val)
                    
                    if is_sat:
                        # Will become unsatisfied
                        if old_count == 1:
                            newly_unsat += 1
                    else:
                        # Will become satisfied
                        if old_count == 0:
                            newly_sat += 1
                
                # Score: primarily minimize newly_unsat, secondarily maximize newly_sat
                # We can combine them: score = newly_unsat * 10 - newly_sat
                # Or just minimize newly_unsat, then random tie break.
                # Let's use a weighted score to break ties deterministically towards "better" flips
                score = newly_unsat * 100 - newly_sat
                
                if score < best_score:
                    best_score = score
                    best_v = v
                    best_tie_breaker = rng.random() # Store random for tie breaking if needed
                elif score == best_score:
                    # Tie breaker: random
                    if rng.random() < 0.5:
                        best_v = v
                        best_tie_breaker = rng.random()
            
            # Apply Noise: With probability 'noise', pick a random variable from the clause
            if rng.random() < noise:
                l = rng.choice(clause)
                v = abs(l) - 1
            else:
                v = best_v
                
            # Apply flip
            apply_flip(v, assignment, counts, unsat_set)
            
            steps += 1
            
            # Update improvement tracking
            current_unsat = len(unsat_set)
            if current_unsat < run_best_unsat:
                run_best_unsat = current_unsat
                run_best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            # If we found a better global best, update it
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            # Early exit if we get stuck for a very long time and noise is maxed
            if steps_since_improvement > 1000 and noise >= 0.8:
                break

        # End of local search run
        # If we improved the global best, we keep it.
        # The loop will restart, potentially perturbing the new best.
        
    return best_assignment if best_assignment is not None else assignment
