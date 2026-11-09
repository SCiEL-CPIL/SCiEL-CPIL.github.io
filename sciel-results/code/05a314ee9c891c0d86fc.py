import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 315 seconds to be safe against 330s kill
    start_time = time.time()
    deadline = start_time + 315.0
    
    # Build index: for each variable, store list of (clause_idx, literal)
    var_occurrences = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append((i, l))
    
    # Helper: compute clause_sat_count from scratch
    def compute_clause_sat_count(assignment):
        counts = [0] * m
        for i, c in enumerate(clauses):
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    counts[i] += 1
        return counts
    
    # Helper: check if all clauses satisfied
    def is_satisfied(clause_sat_count):
        for count in clause_sat_count:
            if count == 0:
                return False
        return True
    
    # Helper: get unsat clause indices
    def get_unsat_indices(clause_sat_count):
        return [i for i, count in enumerate(clause_sat_count) if count == 0]
    
    # Helper: check if a literal is satisfied by current assignment
    def is_literal_satisfied(l, assignment):
        v = abs(l) - 1
        return (l > 0 and assignment[v]) or (l < 0 and not assignment[v])
    
    # Helper: apply a flip to variable v, updating clause_sat_count
    def apply_flip(v, assignment, clause_sat_count):
        assignment[v] = not assignment[v]
        for ci, lit in var_occurrences[v]:
            if is_literal_satisfied(lit, assignment):
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    # WalkSAT step with improved heuristic and unsat set tracking
    def walksat_step(assignment, clause_sat_count, unsat_set, rng, noise):
        if not unsat_set:
            return True
        
        # Pick random unsat clause
        clause_list = list(unsat_set)
        clause_idx = rng.choice(clause_list)
        clause = clauses[clause_idx]
        
        # With probability noise, pick a random literal from the clause
        if rng.random() < noise:
            l = rng.choice(clause)
            v = abs(l) - 1
            old_count = clause_sat_count[clause_idx]
            apply_flip(v, assignment, clause_sat_count)
            new_count = clause_sat_count[clause_idx]
            if old_count == 0 and new_count > 0:
                unsat_set.discard(clause_idx)
            elif old_count > 0 and new_count == 0:
                unsat_set.add(clause_idx)
            # Update other affected clauses
            for ci, lit in var_occurrences[v]:
                if ci == clause_idx:
                    continue
                if clause_sat_count[ci] == 0:
                    unsat_set.add(ci)
                elif ci in unsat_set and clause_sat_count[ci] > 0:
                    unsat_set.discard(ci)
            return False
        
        # Otherwise, pick the literal that minimizes newly unsatisfied clauses
        best_var_idx = None
        best_score = float('inf')
        
        for l in clause:
            v = abs(l) - 1
            
            # Calculate the effect of flipping v
            newly_unsat = 0
            newly_sat = 0
            
            for ci, lit in var_occurrences[v]:
                old_count = clause_sat_count[ci]
                currently_sat = is_literal_satisfied(lit, assignment)
                
                if currently_sat:
                    if old_count == 1:
                        newly_unsat += 1
                else:
                    if old_count == 0:
                        newly_sat += 1
            
            # Score: minimize newly_unsat, maximize newly_sat
            score = newly_unsat * 10 - newly_sat
            if score < best_score:
                best_score = score
                best_var_idx = v
        
        # Apply the best flip
        if best_var_idx is not None:
            old_count = clause_sat_count[clause_idx]
            apply_flip(best_var_idx, assignment, clause_sat_count)
            new_count = clause_sat_count[clause_idx]
            if old_count == 0 and new_count > 0:
                unsat_set.discard(clause_idx)
            elif old_count > 0 and new_count == 0:
                unsat_set.add(clause_idx)
            # Update other affected clauses
            for ci, lit in var_occurrences[best_var_idx]:
                if ci == clause_idx:
                    continue
                if clause_sat_count[ci] == 0:
                    unsat_set.add(ci)
                elif ci in unsat_set and clause_sat_count[ci] > 0:
                    unsat_set.discard(ci)
        
        return False
    
    # Main loop with multiple restarts and adaptive noise
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = float('inf')
    
    while time.time() < deadline:
        # Initialize assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = compute_clause_sat_count(assignment)
        
        # Check if already satisfied
        if is_satisfied(clause_sat_count):
            return assignment
        
        # Build unsat set
        unsat_set = set(get_unsat_indices(clause_sat_count))
        
        if not unsat_set:
            return assignment
        
        # Adaptive noise: start low, increase over steps
        base_noise = 0.2
        max_noise = 0.6
        max_steps_per_run = 10000
        
        steps = 0
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                if best_assignment is not None:
                    return best_assignment
                return assignment
            
            if not unsat_set:
                return assignment
            
            # Adaptive noise: increase with steps to escape local minima
            progress = steps / max_steps_per_run
            noise = base_noise + (max_noise - base_noise) * progress
            
            walksat_step(assignment, clause_sat_count, unsat_set, rng, noise)
            steps += 1
        
        # Track best assignment (fewest unsat clauses)
        unsat_count = len(unsat_set)
        if unsat_count < best_unsat_count:
            best_assignment = assignment[:]
            best_unsat_count = unsat_count
    
    if best_assignment is not None:
        return best_assignment
    return assignment
