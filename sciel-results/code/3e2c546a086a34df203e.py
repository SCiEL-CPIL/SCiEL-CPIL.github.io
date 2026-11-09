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
    
    # Helper: check if a literal is satisfied by current assignment
    def is_literal_satisfied(l, assignment):
        v = abs(l) - 1
        return (l > 0 and assignment[v]) or (l < 0 and not assignment[v])
    
    # Fast WalkSAT with minimal overhead
    def fast_walksat(assignment, rng, max_steps, base_noise):
        # Compute initial clause satisfaction counts
        clause_sat = [0] * m
        for i, c in enumerate(clauses):
            for l in c:
                if is_literal_satisfied(l, assignment):
                    clause_sat[i] += 1
        
        unsat_count = sum(1 for c in clause_sat if c == 0)
        
        for step in range(max_steps):
            if unsat_count == 0:
                return True, assignment
            
            # Pick a random unsatisfied clause
            while True:
                ci = rng.randrange(m)
                if clause_sat[ci] == 0:
                    break
            
            clause = clauses[ci]
            
            # Decide: random flip or weighted flip
            if rng.random() < base_noise:
                l = clause[rng.randrange(3)]
                v = abs(l) - 1
            else:
                # Weighted: pick literal that minimizes newly unsat clauses
                best_v = -1
                best_score = float('inf')
                for l in clause:
                    v = abs(l) - 1
                    newly_unsat = 0
                    for cj, lit in var_occurrences[v]:
                        if clause_sat[cj] == 1 and is_literal_satisfied(lit, assignment):
                            newly_unsat += 1
                    score = newly_unsat
                    if score < best_score:
                        best_score = score
                        best_v = v
                v = best_v
            
            # Apply flip
            assignment[v] = not assignment[v]
            for cj, lit in var_occurrences[v]:
                if is_literal_satisfied(lit, assignment):
                    old = clause_sat[cj]
                    clause_sat[cj] = old + 1
                    if old == 0:
                        unsat_count -= 1
                else:
                    old = clause_sat[cj]
                    clause_sat[cj] = old - 1
                    if old == 1:
                        unsat_count += 1
        
        return False, assignment
    
    # Main loop with restarts and adaptive parameters
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')
    
    restart_count = 0
    while time.time() < deadline:
        # Initialize assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count
        initial_unsat = 0
        for c in clauses:
            sat = False
            for l in c:
                if is_literal_satisfied(l, assignment):
                    sat = True
                    break
            if not sat:
                initial_unsat += 1
        
        if initial_unsat == 0:
            return assignment
        
        # Adaptive parameters: increase noise and steps with restarts
        noise = min(0.1 + restart_count * 0.05, 0.5)
        max_steps = min(5000 + restart_count * 2000, 30000)
        
        solved, final_assignment = fast_walksat(assignment, rng, max_steps, noise)
        
        if solved:
            return final_assignment
        
        # Track best
        final_unsat = 0
        for c in clauses:
            sat = False
            for l in c:
                if is_literal_satisfied(l, final_assignment):
                    sat = True
                    break
            if not sat:
                final_unsat += 1
        
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = final_assignment[:]
        
        restart_count += 1
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
