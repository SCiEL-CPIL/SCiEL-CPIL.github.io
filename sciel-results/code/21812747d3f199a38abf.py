"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42) # Fixed seed for reproducibility, or maybe 0
    
    # Preprocessing
    m = len(clauses)
    clause_vars = []
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_clauses[var_idx].append((i, is_pos))
        clause_vars.append(cv)
    
    # Time management
    start_time = time.time()
    time_limit = 295.0 # Safety margin for 330s kill
    
    def time_up():
        return time.time() - start_time > time_limit

    def init_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def calc_sat_counts(assignment):
        """Calculate initial satisfaction counts for all clauses."""
        counts = [0] * m
        for i in range(m):
            cnt = 0
            for var_idx, is_pos in clause_vars[i]:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts

    def get_unsat_clauses(counts):
        return [i for i in range(m) if counts[i] == 0]

    def walk_sat(assignment, max_steps):
        """
        Run WalkSAT with damage-minimizing heuristic.
        Returns (assignment, solved, unsat_count)
        """
        counts = calc_sat_counts(assignment)
        unsat = get_unsat_clauses(counts)
        
        if not unsat:
            return assignment, True, 0

        # Noise parameter: probability of choosing a random literal instead of best
        noise = 0.25 
        
        for step in range(max_steps):
            if time_up():
                return assignment, False, len(unsat)
            
            if not unsat:
                return assignment, True, 0
                
            # Pick a random unsatisfied clause
            # Weighting by "age" or just random is standard. Let's stick to random for simplicity 
            # as damage heuristic is the main driver.
            clause_idx = rng.choice(unsat)
            clause = clause_vars[clause_idx]
            
            # Heuristic: Pick variable with minimum damage
            best_var = -1
            best_flip_val = False
            best_damage = float('inf')
            
            for var_idx, is_pos in clause:
                # We want to flip var_idx to is_pos to satisfy this clause.
                # Current value: assignment[var_idx]
                # If assignment[var_idx] is already is_pos, it would be satisfied, 
                # but the clause is unsat, so this literal is not satisfied. 
                # So assignment[var_idx] != is_pos.
                
                # Calculate damage:
                # How many satisfied clauses containing (var_idx, NOT is_pos) will become unsatisfied?
                damage = 0
                for c_idx, c_is_pos in var_clauses[var_idx]:
                    if c_idx == clause_idx:
                        continue
                    # If this other clause has the opposite literal (var_idx, NOT is_pos)
                    if c_is_pos != is_pos:
                        # Check if this other clause is currently satisfied
                        if counts[c_idx] > 0:
                            # It is satisfied. If we flip var_idx, this literal becomes unsatisfied.
                            # Does the clause become unsatisfied?
                            # It becomes unsatisfied only if this was the ONLY satisfied literal.
                            if counts[c_idx] == 1:
                                damage += 1
                
                # We minimize damage. 
                # Tie-breaking: random
                if damage < best_damage or (damage == best_damage and rng.random() < 0.5):
                    best_damage = damage
                    best_var = var_idx
                    best_flip_val = is_pos
            
            # With some probability, pick a random variable from the clause to escape local minima
            if rng.random() < noise:
                var_idx, is_pos = rng.choice(clause)
                best_var = var_idx
                best_flip_val = is_pos
            
            # Perform flip
            if best_var != -1:
                old_val = assignment[best_var]
                assignment[best_var] = best_flip_val
                
                # Update counts incrementally
                # The flipped variable affects all clauses it appears in.
                # For each clause containing best_var:
                for c_idx, c_is_pos in var_clauses[best_var]:
                    # Determine if the literal (best_var, c_is_pos) was satisfied before and is satisfied now
                    was_sat = (old_val == c_is_pos)
                    is_sat = (best_flip_val == c_is_pos)
                    
                    if was_sat and not is_sat:
                        # Lost satisfaction
                        counts[c_idx] -= 1
                        if counts[c_idx] == 0:
                            # Newly unsatisfied
                            # We don't maintain a set of unsat clauses dynamically for simplicity of logic
                            # We recalculate unsat list at the end of the step or use a set.
                            # Given m is likely < 1000 for 150 vars (planted 3-SAT usually m ~ 4.26*n ~ 640),
                            # recalculating unsat list is O(m) which is fine.
                            pass
                    elif not was_sat and is_sat:
                        # Gained satisfaction
                        counts[c_idx] += 1
                        # If it was 0, it's now 1, so it's no longer unsat.
                        # We handle this by recalculating unsat list below.
                        pass
            
            # Recalculate unsatisfied clauses list
            # This is O(m). With m ~ 600 and 10k steps, this is 6M ops, which is fast in Python if optimized?
            # Actually, list comprehension is fast.
            unsat = [i for i in range(m) if counts[i] == 0]
            
            # Optimization: If we know we just satisfied a clause, we could remove it.
            # But simple recalculation is safer and likely fast enough for 150 vars.
            
        return assignment, len(unsat) == 0, len(unsat)

    # Main loop with restarts
    best_assignment = init_assignment()
    best_unsat = float('inf')
    
    # Adaptive step limit: start small, increase if time permits?
    # Or fixed steps per restart.
    # Let's try 2000 steps per restart.
    steps_per_restart = 2000
    
    while not time_up():
        # Generate new assignment
        # Mix of random and best-so-far
        if best_unsat < float('inf') and rng.random() < 0.3:
            assignment = best_assignment[:]
        else:
            assignment = init_assignment()
            
        assignment, solved, cur_unsat = walk_sat(assignment, steps_per_restart)
        
        if solved:
            return assignment
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
    return best_assignment

# EVOLVE-BLOCK-END
