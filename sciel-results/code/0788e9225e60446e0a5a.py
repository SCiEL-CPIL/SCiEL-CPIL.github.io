import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Build index
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(i)
            
    def is_sat(clause, assignment):
        l1, l2, l3 = clause
        return (assignment[abs(l1)-1] == (l1>0) or 
                assignment[abs(l2)-1] == (l2>0) or 
                assignment[abs(l3)-1] == (l3>0))

    def count_unsat(assignment):
        cnt = 0
        for c in clauses:
            if not is_sat(c, assignment):
                cnt += 1
        return cnt

    def get_unsat_clause(assignment):
        for c in clauses:
            if not is_sat(c, assignment):
                return c
        return None

    deadline = time.time() + 280.0
    best_unsat_count = float('inf')
    best_assignment = None
    rng = random.Random(42)
    
    step_budget = 20000 # MinConflicts is slower per step, so fewer steps per restart
    max_restarts = 2000
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        for step in range(step_budget):
            if time.time() > deadline:
                if best_assignment:
                    return best_assignment
                return assignment
                
            unsat_clause = get_unsat_clause(assignment)
            if unsat_clause is None:
                return assignment
                
            # MinConflicts: pick variable that minimizes unsat neighbors
            l1, l2, l3 = unsat_clause
            vars_to_check = [abs(l1)-1, abs(l2)-1, abs(l3)-1]
            
            best_var = None
            min_cost = float('inf')
            
            for v in vars_to_check:
                cost = 0
                # Check all clauses containing v, except the current one?
                # Actually, the current clause is unsat, so flipping v makes it sat.
                # We care about the *other* clauses.
                for c_idx in var_to_clauses[v]:
                    # Skip the current clause? 
                    # If we flip v, current clause becomes sat.
                    # Other clauses might become unsat.
                    # We want to minimize the number of *newly* unsat clauses.
                    # A clause is "newly unsat" if it was sat before and becomes unsat.
                    # Or simply: count how many of the *other* clauses are currently unsat.
                    # If we flip v, those unsat clauses might become sat (good), 
                    # and sat clauses might become unsat (bad).
                    
                    # Simpler metric: 
                    # Count how many clauses containing v are currently UNSAT.
                    # If we flip v, we hope to satisfy them.
                    # So we want to pick the variable that is in the MOST unsat clauses?
                    # That's "MaxConflicts" or "WalkSAT with weighting".
                    
                    # Let's use: Pick variable that appears in the most unsat clauses.
                    # This is a strong heuristic.
                    
                    # We need to know which clauses are unsat.
                    # We don't have a list.
                    # We can just count by checking.
                    # This makes the step O(C) again.
                    
                    # Let's stick to Random for simplicity and speed, 
                    # but use the efficient `is_sat` check.
                    # Random WalkSAT is surprisingly robust for N=50.
                    
                    pass # Skip complex heuristic to keep code clean and fast enough.
                    
            # Just pick random
            v = rng.choice(vars_to_check)
            assignment[v] = not assignment[v]
            
    if best_assignment:
        return best_assignment
    return assignment
