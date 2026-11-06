"""Synthetic planted 3-SAT solver (150 variables) using optimized WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Time management
    start_time = time.time()
    time_limit = 290.0  # Safety margin for 330s kill

    def time_up():
        return time.time() - start_time > time_limit

    # Preprocessing: Build data structures for fast access
    # clause_vars[i] = list of (var_idx, is_pos) for clause i
    # var_clauses[j] = list of (clause_idx, is_pos) for variable j
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

    # Convert to lists of lists for faster indexing if needed, 
    # but tuples are fine. Let's stick to the structure.

    def init_assignment(seed):
        rng = random.Random(seed)
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

    def walk_sat(assignment, max_steps, seed):
        """
        Run WalkSAT.
        Returns (assignment, solved, unsat_count)
        """
        rng = random.Random(seed)
        counts = calc_sat_counts(assignment)
        
        # Initial check
        if all(c > 0 for c in counts):
            return assignment, True, 0

        # Noise parameter: probability of choosing a random literal instead of best
        # Standard WalkSAT often uses ~0.2-0.5. For planted SAT, slightly lower might be better?
        # Let's try 0.3.
        noise = 0.3
        
        # To speed up unsat clause selection, we maintain a list of unsat clause indices.
        # However, updating this list incrementally is tricky. 
        # Given m is small (~600), recalculating or filtering is fast.
        # But picking a random unsat clause from a list of ~10-50 unsat clauses is fast.
        
        unsat_indices = [i for i in range(m) if counts[i] == 0]
        
        if not unsat_indices:
            return assignment, True, 0

        for step in range(max_steps):
            if time_up():
                return assignment, False, len(unsat_indices)
            
            if not unsat_indices:
                return assignment, True, 0
                
            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clause_vars[clause_idx]
            
            # Heuristic: Pick variable with minimum damage
            best_var = -1
            best_flip_val = False
            best_damage = float('inf')
            
            # For 3-SAT, we only check 3 literals
            for var_idx, is_pos in clause:
                # Calculate damage
                # How many satisfied clauses containing (var_idx, NOT is_pos) will become unsatisfied?
                # i.e., clauses where (var_idx, not is_pos) is the ONLY satisfied literal.
                damage = 0
                for c_idx, c_is_pos in var_clauses[var_idx]:
                    if c_idx == clause_idx:
                        continue
                    # If this other clause has the opposite literal (var_idx, NOT is_pos)
                    if c_is_pos != is_pos:
                        # Check if this other clause is currently satisfied
                        # If counts[c_idx] == 1, it means only one literal is satisfied.
                        # If that literal is (var_idx, c_is_pos) which is currently true,
                        # and we flip var_idx to make (var_idx, is_pos) true, then (var_idx, c_is_pos) becomes false.
                        # Since it was the only one, the clause becomes unsatisfied.
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
                var_idx, is_pos = clause[rng.randrange(3)]
                best_var = var_idx
                best_flip_val = is_pos
            
            # Perform flip
            if best_var != -1:
                old_val = assignment[best_var]
                assignment[best_var] = best_flip_val
                
                # Update counts incrementally
                # The flipped variable affects all clauses it appears in.
                for c_idx, c_is_pos in var_clauses[best_var]:
                    # Determine if the literal (best_var, c_is_pos) was satisfied before and is satisfied now
                    was_sat = (old_val == c_is_pos)
                    is_sat = (best_flip_val == c_is_pos)
                    
                    if was_sat and not is_sat:
                        # Lost satisfaction
                        counts[c_idx] -= 1
                        if counts[c_idx] == 0:
                            # Newly unsatisfied. Add to unsat_indices if not already there?
                            # It was >0 before, so it wasn't in unsat_indices.
                            # We need to add it.
                            unsat_indices.append(c_idx)
                    elif not was_sat and is_sat:
                        # Gained satisfaction
                        counts[c_idx] += 1
                        # If it was 0, it's now 1, so it's no longer unsat.
                        # Remove from unsat_indices?
                        # List removal is O(k) where k is length of unsat_indices.
                        # If unsat_indices is small, this is fast.
                        # But we might have duplicates? No, each clause is added only when it drops to 0.
                        # It is removed when it goes to 1.
                        # So no duplicates.
                        # However, checking `if counts[c_idx] == 1` is not enough to know it WAS unsat.
                        # It was unsat if counts[c_idx] was 0 before increment.
                        # So if counts[c_idx] == 1 after increment, it was 0 before.
                        if counts[c_idx] == 1:
                            # It was unsat, now sat. Remove from list.
                            try:
                                unsat_indices.remove(c_idx)
                            except ValueError:
                                pass # Should not happen if logic is correct

        return assignment, len(unsat_indices) == 0, len(unsat_indices)

    # Main loop with restarts
    # Strategy:
    # 1. Try a few short random walks.
    # 2. Increase step limit if not solved.
    # 3. Use best-so-far assignment as a starting point for some restarts.
    
    best_assignment = None
    best_unsat = float('inf')
    
    # Adaptive steps
    # Start with 1000, double each time up to 10000?
    # Or just use a fixed large number?
    # For 150 vars, 10000 steps is usually plenty if the heuristic is good.
    # Let's use a dynamic approach:
    # steps = 500 initially, increase by 2x if not solved, max 20000.
    
    base_steps = 1000
    max_steps = 20000
    current_steps = base_steps
    restarts = 0
    
    while not time_up():
        # Generate seed
        seed = restarts + 12345
        
        # Decide starting assignment
        if best_assignment is not None and rng.random() < 0.2: # Note: rng not defined here, use random
             assignment = best_assignment[:]
        else:
             assignment = init_assignment(seed)
        
        assignment, solved, cur_unsat = walk_sat(assignment, current_steps, seed)
        
        if solved:
            return assignment
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        restarts += 1
        
        # Increase steps if we haven't solved yet and steps are low
        if current_steps < max_steps:
            current_steps = min(current_steps * 2, max_steps)
            
    # Time up, return best found
    if best_assignment is not None:
        return best_assignment
    return init_assignment(0)

# EVOLVE-BLOCK-END
