# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 325s to leave 5s safety margin before 330s kill
    deadline = time.time() + 325.0
    rng = random.Random(0)
    
    # Preprocess: convert to 0-based, store signs
    # clause is list of signed literals (1-based)
    # We will store as list of (var_idx, sign) where sign is +1 or -1
    # var_idx is 0-based
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pc.append((var, sign))
        processed_clauses.append(pc)
    
    # Helper to check if a clause is satisfied
    def clause_satisfied(clause, assignment):
        for var, sign in clause:
            if assignment[var] == (sign == 1):
                return True
        return False
    
    # Helper to check if a clause is falsified
    def clause_falsified(clause, assignment):
        for var, sign in clause:
            if assignment[var] == (sign == 1):
                return False
        return True
    
    # Helper to count unsatisfied clauses
    def count_unsat(assignment):
        cnt = 0
        for c in processed_clauses:
            if not clause_satisfied(c, assignment):
                cnt += 1
        return cnt
    
    # Helper to get unsatisfied clauses
    def get_unsat(assignment):
        unsat = []
        for c in processed_clauses:
            if not clause_satisfied(c, assignment):
                unsat.append(c)
        return unsat

    best_assignment = None
    best_unsat = float('inf')

    # Strategy 1: Random restarts with WalkSAT
    # Use a dynamic time budget per restart
    num_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            unsat = get_unsat(assignment)
            if not unsat:
                return assignment # Found solution
            
            # Track best
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
            
            # Choose a random unsatisfied clause
            clause = rng.choice(unsat)
            
            # Choose a literal to flip
            # Weighted random choice: prefer literals that are false in the clause
            # But standard WalkSAT picks uniformly from the clause
            literal_var, literal_sign = rng.choice(clause)
            
            # Flip the variable
            assignment[literal_var] = not assignment[literal_var]
        
        # Check if the final state of this restart is better
        final_unsat = count_unsat(assignment)
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = assignment[:]

    # Strategy 2: If we haven't solved it, try a local search polish on the best assignment found
    # Or simply return the best assignment found so far.
    # Given the "planted" nature, WalkSAT is usually very effective.
    # Let's add a simple deterministic polish: try flipping each variable once to see if it reduces unsat count.
    if best_assignment is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_unsat = count_unsat(best_assignment)
            for var in range(n):
                if time.time() > deadline:
                    break
                # Flip
                best_assignment[var] = not best_assignment[var]
                new_unsat = count_unsat(best_assignment)
                if new_unsat < current_unsat:
                    current_unsat = new_unsat
                    improved = True
                    best_unsat = new_unsat
                else:
                    # Unflip
                    best_assignment[var] = not best_assignment[var]
            
            if best_unsat == 0:
                return best_assignment

    # If no assignment was found (unlikely), return random
    if best_assignment is None:
        return [rng.random() < 0.5 for _ in range(n)]
        
    return best_assignment
# EVOLVE-BLOCK-END
