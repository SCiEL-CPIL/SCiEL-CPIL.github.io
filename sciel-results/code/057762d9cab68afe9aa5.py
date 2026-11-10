"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variables in each clause for faster access
    # clauses is list of lists of signed ints
    clause_vars = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        clause_vars.append(vars_in_c)
        
    def is_satisfied(assignment, clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for c in clauses:
            if not is_satisfied(assignment, c):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        return [i for i, c in enumerate(clauses) if not is_satisfied(assignment, c)]

    def walksat(assignment, max_steps, p=0.5):
        # Returns (solved, assignment)
        unsat_indices = get_unsat_clauses(assignment)
        if not unsat_indices:
            return True, assignment
        
        for _ in range(max_steps):
            if not unsat_indices:
                return True, assignment
                
            # Pick a random unsat clause
            clause_idx = random.choice(unsat_indices)
            clause = clauses[clause_idx]
            vars_in_clause = clause_vars[clause_idx]
            
            # Determine best flip
            min_delta = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                # Calculate delta if we flip v
                # We need to check all clauses containing v
                # To do this efficiently, we should precompute variable -> clause indices
                # But for n=50 and small m, we can just recount or use a simpler heuristic
                # Let's use a simple heuristic: 
                # delta = (number of clauses satisfied by v) - (number of clauses unsatisfied by v)
                # This requires knowing which clauses contain v.
                
                # For simplicity and speed in this rewrite, we will use the standard WalkSAT 
                # which often just picks a random var from the clause with prob p, 
                # and the "best" var with prob 1-p.
                # To find "best" without full precomputation of var->clauses, 
                # we can estimate or just use the random pick if p is high enough.
                # However, a greedy step is powerful.
                
                # Let's implement a lightweight greedy check:
                # Flip v, count unsat, unflip v.
                # This is O(m) per var. 3 vars per clause. So O(3*m) per step.
                # m is likely around 40-50 for 3-SAT (ratio ~1).
                # 3*50 = 150 operations per step. 1000 steps = 150k ops. Very fast.
                
                old_val = assignment[v]
                assignment[v] = not old_val
                new_unsat = count_unsat(assignment)
                assignment[v] = old_val
                
                # We need to compare with current unsat count
                # We don't track current unsat count globally in this inner loop easily 
                # unless we update it.
                # Let's just calculate the delta relative to the current state.
                # Actually, just counting new_unsat and comparing to a baseline is tricky 
                # if we don't store the baseline.
                
                # Let's store current_unsat outside the var loop.
                pass
            
            # Optimization: Precompute variable to clause mapping for efficient delta calculation
            # This is better.
            pass
        
        return False, assignment

    # Precompute variable to clause indices mapping for efficient delta calculation
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append(i)

    def count_unsat_fast(assignment):
        # Count unsat clauses
        # This is O(m * 3)
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walksat_efficient(assignment, max_steps, p=0.5):
        current_unsat = count_unsat_fast(assignment)
        if current_unsat == 0:
            return True, assignment
        
        for _ in range(max_steps):
            if current_unsat == 0:
                return True, assignment
            
            # Get an unsat clause
            # We can pick a random clause and check if it's unsat, or maintain a list.
            # Maintaining a list is better but requires updates.
            # For n=50, scanning all clauses to find one unsat is O(m).
            # Let's just pick a random index and check.
            while True:
                ci = random.randint(0, len(clauses) - 1)
                if not is_satisfied(assignment, clauses[ci]):
                    break
            
            clause = clauses[ci]
            vars_in_clause = clause_vars[ci]
            
            # Find best flip
            min_delta = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                # Calculate change in unsat count if we flip v
                # A clause containing v changes state if v is the only satisfied literal 
                # (becomes unsat) or if v is the only unsatisfied literal in an unsat clause 
                # (becomes sat).
                
                # Let's just use the brute force count for simplicity and correctness, 
                # as m is small.
                old_val = assignment[v]
                assignment[v] = not old_val
                new_unsat = count_unsat_fast(assignment)
                assignment[v] = old_val
                
                delta = new_unsat - current_unsat
                
                if delta < min_delta:
                    min_delta = delta
                    best_vars = [v]
                elif delta == min_delta:
                    best_vars.append(v)
            
            # Choose variable to flip
            if random.random() < p:
                # Random walk: pick any var from the clause
                v_to_flip = random.choice(vars_in_clause)
            else:
                # Greedy: pick a var that minimizes unsat count
                v_to_flip = random.choice(best_vars)
            
            # Perform flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update current_unsat
            # Instead of recounting everything, we can update incrementally, 
            # but recounting is O(m*3) which is fine.
            current_unsat = count_unsat_fast(assignment)
            
        return False, assignment

    # Main loop with restarts
    start_time = time.time()
    max_time = 310.0  # Safety margin before 330s kill
    
    restart = 0
    best_unsat_count = float('inf')
    best_assignment = None
    
    while time.time() - start_time < max_time:
        restart += 1
        # Exponential increase in steps per restart
        # Start with 1000, then 2000, 4000, etc.
        steps = min(1000 * (2 ** (restart - 1)), 100000)
        
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Check if initial assignment is already satisfying
        init_unsat = count_unsat_fast(assignment)
        if init_unsat < best_unsat_count:
            best_unsat_count = init_unsat
            best_assignment = list(assignment)
        
        solved, final_assignment = walksat_efficient(assignment, steps, p=0.5)
        
        if solved:
            return final_assignment
        
        # Track best unsatisfied count for fallback
        final_unsat = count_unsat_fast(final_assignment)
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = list(final_assignment)
            
    # If time runs out, return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
