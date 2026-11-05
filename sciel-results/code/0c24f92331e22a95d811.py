"""
Improved 3-SAT solver using WalkSAT with clause-weighted variable selection.
"""
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Safety deadline: 290 seconds to ensure we return before the 300s kill.
    deadline = time.time() + 290.0
    
    # Pre-process: Map variables to the clauses they appear in
    # var_to_clauses[i] is a list of clause indices containing variable i
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(ci)

    def is_satisfied(assignment, clause):
        for lit in clause:
            var_idx = abs(lit) - 1
            if assignment[var_idx] == (lit > 0):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for clause in clauses:
            if not is_satisfied(assignment, clause):
                count += 1
        return count

    def walksat_step(assignment, unsat_clauses, rng, p):
        """
        Perform one step of WalkSAT.
        Returns the new assignment and updated list of unsatisfied clauses.
        p is the probability of choosing a variable randomly (to escape local minima).
        """
        if not unsat_clauses:
            return assignment, unsat_clauses
            
        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_clauses)
        clause = clauses[ci]
        
        # Calculate gain for each variable in the clause
        # gain = (# of currently unsatisfied clauses this var satisfies) - 
        #        (# of currently satisfied clauses this var unsatisfies)
        gains = []
        for lit in clause:
            var_idx = abs(lit) - 1
            new_val = lit > 0
            
            gain = 0
            for ci2 in var_to_clauses[var_idx]:
                clause2 = clauses[ci2]
                # Check if clause2 is currently satisfied
                sat = is_satisfied(assignment, clause2)
                
                # Check if clause2 would be satisfied after flipping var_idx to new_val
                would_be_sat = False
                for l2 in clause2:
                    v2 = abs(l2) - 1
                    if v2 == var_idx:
                        if new_val == (l2 > 0):
                            would_be_sat = True
                            break
                    else:
                        if assignment[v2] == (l2 > 0):
                            would_be_sat = True
                            break
                
                if sat and not would_be_sat:
                    gain -= 1
                elif not sat and would_be_sat:
                    gain += 1
            
            gains.append((var_idx, gain))
        
        # Choose variable to flip
        if rng.random() < p:
            # Random choice
            var_idx, _ = rng.choice(gains)
        else:
            # Choose variable with maximum gain (random tie-break)
            max_gain = max(g for _, g in gains)
            best_vars = [v for v, g in gains if g == max_gain]
            var_idx = rng.choice(best_vars)
        
        # Flip the variable
        assignment[var_idx] = not assignment[var_idx]
        
        # Update unsat_clauses list
        # We need to check which clauses are affected by this flip
        # A clause is affected if it contains var_idx
        # We can just re-evaluate the status of all clauses containing var_idx
        # And update the global unsat_clauses list
        
        # To keep it simple and efficient enough for n=50, we can just rebuild the unsat list
        # or update it incrementally. Rebuilding is O(num_clauses * clause_size) which is small.
        # However, incremental is better. Let's do incremental.
        
        # First, remove clauses that are now satisfied from unsat_clauses?
        # Actually, it's easier to just recalculate the list of unsat clauses if we don't want complex bookkeeping.
        # Given n=50, num_clauses ~ 200-400, recalculation is very fast.
        # Let's optimize by just checking the affected clauses.
        
        # Identify clauses affected by var_idx
        affected_clauses = set(var_to_clauses[var_idx])
        
        # We will create a new unsat list
        # We can filter the existing unsat_clauses and add any newly unsatisfied ones.
        # But we don't know which ones became unsatisfied without checking.
        
        # Simple approach: Rebuild unsat_clauses.
        # Since this is called potentially thousands of times, let's see if we can do better.
        # Better approach:
        # 1. Check all clauses in affected_clauses.
        # 2. Determine their new status.
        # 3. Update the set of unsat clauses.
        
        # Let's use a set for O(1) lookup of unsat clauses
        # But the function signature returns a list.
        
        # Let's stick to a slightly optimized rebuild for clarity and correctness, 
        # as n=50 is small.
        new_unsat = []
        for c in clauses:
            if not is_satisfied(assignment, c):
                new_unsat.append(c)
        
        return assignment, new_unsat

    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    # WalkSAT parameters
    # p is the probability of random choice. Typical values are 0.1 to 0.5.
    # We can tune this.
    p = 0.3 
    
    max_steps_per_restart = 10000
    max_restarts = 100
    
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unsat clauses
        unsat_clauses = [clause for clause in clauses if not is_satisfied(assignment, clause)]
        
        if not unsat_clauses:
            return assignment
        
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        steps = 0
        no_improve_steps = 0
        max_no_improve = 1000
        
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
                
            prev_unsat_count = len(unsat_clauses)
            
            assignment, unsat_clauses = walksat_step(assignment, unsat_clauses, rng, p)
            current_unsat = len(unsat_clauses)
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < prev_unsat_count:
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            steps += 1
            
            if no_improve_steps > max_no_improve:
                break
                
    if best_assignment is not None:
        return best_assignment
        
    # Fallback (should not happen if we always have an assignment)
    return [False] * n
