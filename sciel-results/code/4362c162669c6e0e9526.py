"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin before 330s limit
    m = len(clauses)
    
    def is_sat(assignment, clause):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for clause in clauses:
            if not is_sat(assignment, clause):
                cnt += 1
        return cnt
    
    # Try multiple restarts with different seeds
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 104729 + 31)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if count_unsat(assignment) == 0:
            return assignment
        
        # WalkSAT with simple random variable selection
        max_steps = 3000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            unsat_clause = None
            for clause in clauses:
                if not is_sat(assignment, clause):
                    unsat_clause = clause
                    break
            
            if unsat_clause is None:
                return assignment
            
            # Pick a random literal from the unsat clause and flip
            lit = unsat_clause[rng.randrange(3)]
            var = abs(lit) - 1
            assignment[var] = not assignment[var]
    
    return assignment
# EVOLVE-BLOCK-END
