# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance (50 variables).
    
    Optimized WalkSAT with:
    - Per-clause satisfaction counting for O(degree) flip updates
    - O(1) unsatisfied clause selection via maintained list
    - Correct greedy damage computation
    - Adaptive random-walk probability
    - Multiple restarts with diverse parameters
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock limit, leave 5s safety margin
    deadline = time.time() + 325.0
    
    # Precompute: for each clause, store (var, is_positive) tuples
    # and build variable-to-clause incidence
    clause_vars = []  # list of (var, is_pos) per clause
    var_to_clauses = [[] for _ in range(n)]  # var -> list of (clause_idx, is_pos)
    
    for ci, clause in enumerate(clauses):
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            is_pos = (lit > 0)
            cv.append((var, is_pos))
            var_to_clauses[var].append((ci, is_pos))
        clause_vars.append(cv)
    
    # Precompute clause sizes
    clause_sizes = [len(cv) for cv in clause_vars]
    
    def initial_sat_counts(assignment):
        """Compute number of satisfied literals per clause."""
        sat_counts = [0] * m
        for ci in range(m):
            cnt = 0
            for var, is_pos in clause_vars[ci]:
                if assignment[var] == is_pos:
                    cnt += 1
            sat_counts[ci] = cnt
        return sat_counts
    
    def walk_sat(assignment, max_steps, rng, p_start=0.5, p_end=0.1):
        """
        Optimized WalkSAT with per-clause satisfaction counting.
        Returns True if solution found, False otherwise.
        """
        sat_counts = initial_sat_counts(assignment)
        
        # Build list of unsatisfied clause indices
        unsat_clauses = [ci for ci in range(m) if sat_counts[ci] < clause_sizes[ci]]
        
        if not unsat_clauses:
            return True
        
        for step in range(max_steps):
            # Time check every 500 steps to reduce overhead
            if step % 500 == 0 and time.time() > deadline - 2.0:
                return False
            
            if not unsat_clauses:
                return True
            
            # Adaptive p: start random, become more greedy
            progress = step / max_steps
            p = p_start + (p_end - p_start) * progress
            
            # Pick a random unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Get the variables in this clause that are currently NOT satisfying it
            candidates = []
            for var, is_pos in clause_vars[ci]:
                if assignment[var] != is_pos:
                    candidates.append(var)
            
            if not candidates:
                # Shouldn't happen, but pick random var
                var = rng.randrange(n)
            else:
                # Choose which variable to flip
                if rng.random() < p:
                    # Random choice among candidates
                    var = candidates[rng.randrange(len(candidates))]
                else:
                    # Greedy: pick the flip with minimum damage
                    best_var = candidates[0]
                    best_damage = float('inf')
                    for v in candidates:
                        damage = 0
                        for (ci2, is_pos2) in var_to_clauses[v]:
                            if ci2 == ci:
                                continue
                            # Check if flipping v would break clause ci2
                            # Clause ci2 is currently satisfied (sat_counts[ci2] >= 1)
                            # Flipping v changes whether literal (v, is_pos2) satisfies ci2
                            # If currently satisfying and would become unsatisfying:
                            #   and if it's the ONLY satisfying literal, clause becomes unsat
                            if sat_counts[ci2] > 0:
                                currently_satisfies = (assignment[v] == is_pos2)
                                if currently_satisfies:
                                    # This literal currently satisfies ci2
                                    # After flip, it won't. If it was the only one, damage.
                                    if sat_counts[ci2] == 1:
                                        damage += 1
                        if damage < best_damage:
                            best_damage = damage
                            best_var = v
                            if best_damage == 0:
                                break
                    var = best_var
            
            # Apply the flip
            old_val = assignment[var]
            assignment[var] = not assignment[var]
            new_val = not old_val
            
            # Update satisfaction counts for all clauses containing this variable
            for (ci2, is_pos2) in var_to_clauses[var]:
                was_satisfying = (old_val == is_pos2)
                now_satisfying = (new_val == is_pos2)
                
                if was_satisfying and not now_satisfying:
                    sat_counts[ci2] -= 1
                elif not was_satisfying and now_satisfying:
                    sat_counts[ci2] += 1
            
            # Update unsat_clauses list
            # Rebuild (m is small: ~100-200 for 50-var 3-SAT)
            unsat_clauses = [ci2 for ci2 in range(m) if sat_counts[ci2] < clause_sizes[ci2]]
            
            if not unsat_clauses:
                return True
        
        return False
    
    # Multiple restarts with diverse parameters
    base_steps = max(5000, n * 100)
    
    for restart in range(300):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 7919 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Vary parameters for diversity
        p_start = 0.3 + 0.4 * (restart % 5) / 4.0
        p_end = 0.05 + 0.15 * (restart % 3) / 2.0
        steps = int(base_steps * (1 + rng.random() * 2))
        
        if walk_sat(assignment, steps, rng, p_start=p_start, p_end=p_end):
            return assignment
    
    # Fallback: more aggressive search
    if time.time() < deadline - 10.0:
        for trial in range(100):
            if time.time() > deadline - 5.0:
                break
            rng = random.Random(trial * 9973 + 777)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            if walk_sat(assignment, base_steps * 3, rng, p_start=0.2, p_end=0.0):
                return assignment
    
    return assignment
# EVOLVE-BLOCK-END
