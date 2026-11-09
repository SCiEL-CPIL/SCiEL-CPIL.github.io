import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Global deadline: 300s limit, use 290s
    deadline = time.time() + 290.0
    
    # Pre-compute counts for phase saving
    pos_counts = [0] * n
    neg_counts = [0] * n
    for clause in clauses:
        for l in clause:
            if l > 0:
                pos_counts[l-1] += 1
            else:
                neg_counts[-l-1] += 1
                
    rng = random.Random(0)
    
    def init_assignment():
        return [pos_counts[i] > neg_counts[i] or (pos_counts[i] == neg_counts[i] and rng.random() < 0.5) for i in range(n)]

    assignment = init_assignment()
    
    def is_sat(clause):
        for l in clause:
            if assignment[l-1] == (l > 0):
                return True
        return False

    # Initial check
    unsat_count = sum(0 if is_sat(c) else 1 for c in clauses)
    if unsat_count == 0:
        return assignment

    best_assignment = assignment.copy()
    best_unsat_count = unsat_count

    # Parameters
    max_steps_per_restart = 5000
    max_restarts = 1000
    p = 0.5 # Randomness parameter

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart > 0:
            assignment = init_assignment()
            unsat_count = sum(0 if is_sat(c) else 1 for c in clauses)
            if unsat_count == 0:
                return assignment
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                return best_assignment

            if unsat_count == 0:
                return assignment

            # Find unsat clauses
            unsat_clauses = []
            for c in clauses:
                if not is_sat(c):
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment

            clause = rng.choice(unsat_clauses)
            
            # Pick literal
            if rng.random() < p:
                literal = rng.choice(clause)
            else:
                # Simple greedy: pick literal appearing most in unsat_clauses
                # This is O(3 * M) which is okay for small M
                scores = [0, 0, 0]
                for i, lit in enumerate(clause):
                    for uc in unsat_clauses:
                        if lit in uc:
                            scores[i] += 1
                # Pick max score, random tie break
                max_score = max(scores)
                candidates = [clause[i] for i in range(3) if scores[i] == max_score]
                literal = rng.choice(candidates)

            var = abs(literal) - 1
            assignment[var] = not assignment[var]
            
            # Recount unsat
            unsat_count = sum(0 if is_sat(c) else 1 for c in clauses)
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

    return best_assignment
