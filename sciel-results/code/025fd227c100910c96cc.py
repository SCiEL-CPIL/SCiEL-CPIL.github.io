"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]
    
    # Convert to 0-based variables and store (var_idx, is_pos)
    # var_idx in [0, n-1]
    clause_vars = []
    for c in clauses_raw:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
        
    m = len(clause_vars)
    
    # Build variable to clause index
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((i, is_pos))
            
    rng = random.Random(0)
    deadline = time.monotonic() + 280.0
    
    def is_clause_satisfied(assignment, cv):
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def get_unsat_clauses(assignment):
        unsat = []
        for i, cv in enumerate(clause_vars):
            if not is_clause_satisfied(assignment, cv):
                unsat.append(i)
        return unsat

    # Min-Conflicts with incremental update
    def walk_one(assignment, max_steps):
        # Initialize unsat set
        unsat_set = set(get_unsat_clauses(assignment))
        
        for step in range(max_steps):
            if time.monotonic() > deadline - 2.0:
                return False, assignment, unsat_set
                
            if not unsat_set:
                return True, assignment, unsat_set
            
            # Pick random unsat clause
            c_idx = rng.choice(list(unsat_set))
            cv = clause_vars[c_idx]
            
            # Heuristic: Pick variable in this clause that minimizes "damage"
            # Damage = number of currently satisfied clauses containing this var 
            # that would become unsatisfied if we flip it.
            # Approximation: Just pick random, or pick var with lowest degree?
            # Let's use a simple score: count how many *other* clauses this var is in 
            # that are currently satisfied. We don't track sat set, so let's just 
            # pick the variable that appears in the fewest clauses overall (static) 
            # or just random. 
            # Better: Random selection is standard WalkSAT. 
            # Let's try: Pick literal that is currently false.
            # All literals in an unsat clause are currently false.
            # So we just pick a random variable from the clause.
            
            var_idx, is_pos = rng.choice(cv)
            
            # Flip variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update unsat_set for clauses containing var_idx
            for c_idx_affected, is_pos_affected in var_to_clauses[var_idx]:
                cv_affected = clause_vars[c_idx_affected]
                # Check status before flip? 
                # We know the status change depends on the new value.
                # If the clause is now satisfied, it should not be in unsat_set.
                # If the clause is now unsatisfied, it should be in unsat_set.
                
                is_sat_now = is_clause_satisfied(assignment, cv_affected)
                
                if is_sat_now:
                    unsat_set.discard(c_idx_affected)
                else:
                    unsat_set.add(c_idx_affected)
                    
        return False, assignment, unsat_set

    best_assignment = [False] * n
    best_unsat_count = m + 1
    
    # Multi-restart
    restart_budget = 1000
    steps_per_restart = 5000
    
    for restart in range(restart_budget):
        if time.monotonic() > deadline - 5.0:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        solved, assignment, unsat_set = walk_one(assignment, steps_per_restart)
        
        if solved:
            return assignment
            
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # Adaptive steps
        remaining = deadline - time.monotonic()
        if remaining < 10.0:
            break
        elif remaining < 30.0:
            steps_per_restart = 2000
        elif remaining < 60.0:
            steps_per_restart = 3000

    return best_assignment
# EVOLVE-BLOCK-END
