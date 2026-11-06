import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute variable to clause index mapping for potential future use, 
    # though for MinConfl we scan unsat clauses directly.
    
    def is_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    best_assignment = None
    min_unsat = float('inf')
    phase = [None] * n  # Phase saving
    
    max_restarts = 5000
    steps_per_run = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            # Initialize phase to this random assignment
            for i in range(n):
                phase[i] = assignment[i]
        else:
            # Start from phase saved from best_assignment
            assignment = [phase[i] if phase[i] is not None else rng.random() < 0.5 for i in range(n)]
            # Perturb slightly to escape local minima
            # Flip a small fraction of variables
            for _ in range(n // 5):
                idx = rng.randint(0, n - 1)
                assignment[idx] = not assignment[idx]
        
        # Walk
        for step in range(steps_per_run):
            if time.time() > deadline:
                # Update best before returning
                if best_assignment is None:
                    return assignment
                # Return best found
                # We should check if current is better? 
                # We track best in the loop, so best_assignment is up to date.
                return best_assignment
            
            # Find unsatisfied clauses
            unsat_indices = []
            for i, c in enumerate(clauses):
                if not is_satisfied(c, assignment):
                    unsat_indices.append(i)
                    
            if not unsat_indices:
                # Solved! Update phase
                for i in range(n):
                    phase[i] = assignment[i]
                return assignment
                
            current_unsat = len(unsat_indices)
            if current_unsat < min_unsat:
                min_unsat = current_unsat
                best_assignment = list(assignment)
                # Save phase of this best assignment
                for i in range(n):
                    phase[i] = assignment[i]
                    
            # Pick a random unsatisfied clause
            target_clause_idx = rng.choice(unsat_indices)
            target_clause = clauses[target_clause_idx]
            
            # Heuristic: MinConfl
            # Pick the literal that appears in the fewest other unsatisfied clauses
            best_lit = None
            min_count = float('inf')
            
            for l in target_clause:
                var = abs(l) - 1
                count = 0
                for ci in unsat_indices:
                    if ci == target_clause_idx:
                        continue
                    clause = clauses[ci]
                    # Check if var is in this clause
                    for ll in clause:
                        if abs(ll) - 1 == var:
                            count += 1
                            break
                if count < min_count:
                    min_count = count
                    best_lit = l
                    if min_count == 0:
                        break # Can't do better than 0
                    
            # Flip the variable
            assignment[abs(best_lit) - 1] = not assignment[abs(best_lit) - 1]
            
    return best_assignment if best_assignment is not None else [False] * n
