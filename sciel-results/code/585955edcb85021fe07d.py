import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Preprocessing: Map variable to list of (clause_index, is_positive)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            is_pos = l > 0
            var_to_clauses[v].append((ci, is_pos))
    
    m = len(clauses)
    
    def initialize():
        """Initialize assignment and clause counts."""
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_true_count = [0] * m
        for ci, clause in enumerate(clauses):
            count = 0
            for l in clause:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    count += 1
            clause_true_count[ci] = count
        return assignment, clause_true_count

    def update_clause(ci, assignment):
        """Recalculate true count for a specific clause."""
        count = 0
        for l in clauses[ci]:
            v = abs(l) - 1
            if assignment[v] == (l > 0):
                count += 1
        return count

    def is_solved(clause_true_count):
        return all(c >= 1 for c in clause_true_count)

    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s
    
    assignment, clause_true_count = initialize()
    
    restart_threshold = 2000
    steps = 0
    p_walk = 0.2  # Probability to pick a random literal instead of best
    
    while time.time() < deadline:
        if is_solved(clause_true_count):
            return assignment
        
        # Check for restart
        if steps >= restart_threshold:
            assignment, clause_true_count = initialize()
            steps = 0
            continue
        
        # Find unsatisfied clauses
        unsat_indices = [i for i, c in enumerate(clause_true_count) if c == 0]
        if not unsat_indices:
            return assignment
        
        # Select clause: Pick one with minimum true count (all are 0 here, so random)
        # If we had partial satisfaction, we'd weight by 1/(count+1). 
        # Since we only pick unsat (count=0), we just pick randomly from unsat.
        # However, sometimes it's better to look at clauses with 1 true literal? 
        # No, standard WalkSAT picks from unsat. 
        # Let's stick to picking from unsat clauses.
        ci = rng.choice(unsat_indices)
        
        # Select variable to flip
        best_var = -1
        best_gain = -1000
        gains = []
        
        for l in clauses[ci]:
            v = abs(l) - 1
            # Calculate gain of flipping v
            gain = 0
            for (other_ci, is_pos) in var_to_clauses[v]:
                # Check current state of this variable in other_ci
                # If assignment[v] matches is_pos, the literal is currently true.
                # Flipping v makes it false. So we lose 1 satisfied literal in this clause.
                # If assignment[v] does not match is_pos, the literal is currently false.
                # Flipping v makes it true. So we gain 1 satisfied literal in this clause.
                if assignment[v] == is_pos:
                    gain -= 1
                else:
                    gain += 1
            gains.append((gain, v))
            
            if gain > best_gain:
                best_gain = gain
                best_var = v
        
        # Apply WalkSAT heuristic: with prob p_walk, pick random, else pick best
        if rng.random() < p_walk:
            # Pick random variable from the clause
            v = rng.choice([abs(l) - 1 for l in clauses[ci]])
        else:
            v = best_var
        
        # Flip variable v
        assignment[v] = not assignment[v]
        
        # Update clause_true_count for all clauses containing v
        for (affected_ci, _) in var_to_clauses[v]:
            # Recalculate count for affected_ci
            # Optimization: Instead of full recalc, we can adjust by +/- 1
            # But full recalc for 3-SAT is cheap (3 literals)
            clause_true_count[affected_ci] = update_clause(affected_ci, assignment)
            
        steps += 1

    # If deadline reached, return best effort
    return assignment
