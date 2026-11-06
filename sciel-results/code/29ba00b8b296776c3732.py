import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 280 seconds (safety margin for 300s limit)
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Pre-compute variable to clause index mapping for potential optimization, 
    # but for N=50, direct evaluation is likely fast enough.
    
    # We will use a Weighted WalkSAT approach with restarts.
    # Weights are assigned to unsatisfied clauses.
    
    def evaluate(assignment):
        """Returns the number of unsatisfied clauses."""
        unsat_count = 0
        for clause in clauses:
            sat = False
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count

    def get_unsat_clauses(assignment):
        """Returns list of unsatisfied clauses."""
        unsat = []
        for clause in clauses:
            sat = False
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(clause)
        return unsat

    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    max_steps_per_run = 2000
    max_restarts = 1000
    p = 0.5 # Probability of random flip (WalkSAT parameter)
    
    rng = random.Random(0)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat
        unsat_count = evaluate(assignment)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Local search
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
                
            # Stagnation check: if no improvement for 500 steps, break and restart
            if steps - last_improvement_step > 500:
                break
                
            # Select variable to flip
            unsat_clauses = get_unsat_clauses(assignment)
            if not unsat_clauses:
                return assignment
                
            # Heuristic: Pick a variable that appears in the most unsatisfied clauses
            # This is a "Min-Conflicts" inspired heuristic.
            var_counts = {}
            for clause in unsat_clauses:
                for l in clause:
                    var_idx = abs(l) - 1
                    var_counts[var_idx] = var_counts.get(var_idx, 0) + 1
            
            if not var_counts:
                # Should not happen if unsat_clauses is not empty
                break
                
            # Find max count
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            
            # With probability p, pick a random variable from the unsat clauses (WalkSAT style)
            # With probability 1-p, pick the best variable (Min-Conflicts style)
            if rng.random() < p:
                # Pick random variable from a random unsat clause
                clause = rng.choice(unsat_clauses)
                literal = rng.choice(clause)
                var_idx = abs(literal) - 1
            else:
                # Pick best variable
                var_idx = rng.choice(candidates)
                
            # Flip variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat count
            # For N=50, recalculating is cheap and less error-prone
            current_unsat = evaluate(assignment)
            steps += 1
            
    # If loop finishes without finding solution, return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
