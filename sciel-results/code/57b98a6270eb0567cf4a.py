"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 320.0
    
    # Pre-compute literal indices for speed
    # Store clauses as tuples of (var_index, is_positive) for faster access
    # But we also need the original for checking. Let's use a faster representation.
    # Actually, let's stick to a hybrid: keep original clauses but use a fast check loop.
    
    # Optimization: Use a list of booleans for assignment.
    # Fast clause satisfaction check.
    
    # Strategy: 
    # 1. Try a few random restarts with WalkSAT.
    # 2. If that fails, try a more systematic approach or longer WalkSAT with better heuristics.
    # 3. Since it's "planted" SAT, it's likely satisfiable and structured.
    #    WalkSAT is usually very effective for planted 3-SAT.
    
    # We'll implement a robust WalkSAT with:
    # - Multiple restarts
    # - Weighted clause selection (prefer smaller unsat clauses)
    # - Weighted literal selection (prefer literals that appear in more unsat clauses)
    # - Time limit check
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    # lit > 0 means variable lit-1 is True.
    # lit < 0 means variable abs(lit)-1 is False.
    
    # For speed, we can pre-compute for each clause the list of (var_idx, sign)
    # where sign is +1 or -1.
    # assignment[var_idx] is True or False.
    # Clause is satisfied if any (assignment[var_idx] == (sign == 1)) is True.
    
    # Let's use a different representation for speed:
    # For each clause, store list of (var_idx, required_value)
    # required_value is True if literal is positive, False if negative.
    
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Also, for literal selection heuristic, we might want to count occurrences.
    # But let's start with a solid WalkSAT.
    
    def check_solution(assignment):
        for fc in fast_clauses:
            sat = False
            for var_idx, required in fc:
                if assignment[var_idx] == required:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        count = 0
        for fc in fast_clauses:
            sat = False
            for var_idx, required in fc:
                if assignment[var_idx] == required:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    # WalkSAT with restarts
    # Parameters:
    # - max_restarts
    # - max_steps_per_restart
    # - p (probability of choosing a literal from the clause vs random)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    num_restarts = 50  # We'll adjust based on time
    max_steps = 5000   # Steps per restart
    
    # Adaptive restarts: if we run out of time, we stop.
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        # For planted SAT, random initialization is usually fine.
        # But maybe a bias towards the planted solution? We don't know it.
        # Random is good.
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Track current unsat count
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # WalkSAT loop
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            # For efficiency, we could maintain a list of unsat clauses,
            # but for n=150 and reasonable clause counts, re-scanning is okay.
            # Let's optimize by keeping a list of unsat clause indices.
            
            # Actually, re-scanning 150 variables * num_clauses can be slow if num_clauses is large.
            # Let's estimate: if num_clauses is ~500, 150*500 = 75k ops per check.
            # 5000 steps * 75k = 375M ops. Might be too slow in Python.
            
            # Better: Maintain a list of unsat clauses.
            # When we flip a variable, update the status of clauses containing that variable.
            
            # Let's implement a faster WalkSAT with clause status tracking.
            
            # For this restart, let's build the tracking structures.
            # This might be overkill if the simple version works fast enough.
            # Let's try a simpler optimized version first:
            # 1. Find all unsat clauses (list of indices).
            # 2. Pick one.
            # 3. Pick a literal.
            # 4. Flip.
            
            # To speed up finding unsat clauses, we can just iterate.
            # But let's try to make the inner loop tight.
            
            # Find unsat clauses
            unsat_indices = []
            for i, fc in enumerate(fast_clauses):
                sat = False
                for var_idx, required in fc:
                    if assignment[var_idx] == required:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment
            
            if len(unsat_indices) < best_unsat_count:
                best_unsat_count = len(unsat_indices)
                best_assignment = assignment[:]
            
            # Choose a clause
            # Weighted by length? Or just random?
            # Random is standard.
            clause_idx = random.choice(unsat_indices)
            clause = fast_clauses[clause_idx]
            
            # Choose a literal to flip
            # Heuristic: prefer literals that are in more unsat clauses.
            # For simplicity, let's use random first, then add heuristic if needed.
            # Actually, a simple heuristic: pick the variable that appears in the most unsat clauses.
            # Let's compute a score for each variable in the clause.
            
            # For now, random choice from the clause.
            var_idx, required = random.choice(clause)
            
            # Flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat count?
            # We'll just re-scan in the next iteration.
            # This is O(num_clauses * clause_len) per step.
            # If num_clauses is 500, clause_len 3, that's 1500 ops per step.
            # 5000 steps * 1500 = 7.5M ops per restart.
            # 50 restarts * 7.5M = 375M ops.
            # Python can do ~10M-50M simple ops per second.
            # So 375M ops might take 10-40 seconds.
            # This should be within the time limit.
        
        # End of restart
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
