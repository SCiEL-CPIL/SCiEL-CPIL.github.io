# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 285s to be safe
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Preprocess: convert to 0-based indices with signs
    # clause_vars[c] = list of (var_idx, sign) where sign is +1 or -1
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for vi, _ in clause_vars[ci]:
            var_to_clauses[vi].append(ci)
    
    def evaluate(assignment):
        """Return number of unsatisfied clauses."""
        unsat = 0
        for ci in range(m):
            sat = False
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat
    
    def walksat_step(assignment, rng, p_flip=0.55):
        """One WalkSAT step. Returns new assignment (mutated in place)."""
        # Find all unsatisfied clauses
        unsat_clauses = []
        for ci in range(m):
            sat = False
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        
        if not unsat_clauses:
            return assignment
        
        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_clauses)
        clause = clause_vars[ci]
        
        # With probability p_flip, flip a literal in the clause (WalkSAT)
        # Otherwise, flip the variable that minimizes the increase in unsat count
        if rng.random() < p_flip:
            vi, sign = rng.choice(clause)
            assignment[vi] = not assignment[vi]
        else:
            # Find best flip (minimizes new unsat count)
            best_var = None
            best_cost = float('inf')
            for vi, sign in clause:
                # Simulate flip
                old_val = assignment[vi]
                assignment[vi] = not old_val
                # Count how many clauses become unsatisfied
                cost = 0
                for cj in var_to_clauses[vi]:
                    sat = False
                    for vj, sj in clause_vars[cj]:
                        if assignment[vj] == (sj == 1):
                            sat = True
                            break
                    if not sat:
                        cost += 1
                assignment[vi] = old_val
                if cost < best_cost:
                    best_cost = cost
                    best_var = vi
            
            if best_var is not None:
                assignment[best_var] = not assignment[best_var]
        
        return assignment
    
    def local_search(assignment, rng, max_iters=5000):
        """Greedy local search to polish."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci in range(m):
                sat = False
                for vi, sign in clause_vars[ci]:
                    if assignment[vi] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                break
            
            # Try flipping each variable in an unsatisfied clause, pick best
            ci = rng.choice(unsat_clauses)
            clause = clause_vars[ci]
            
            best_var = None
            best_cost = float('inf')
            for vi, sign in clause:
                old_val = assignment[vi]
                assignment[vi] = not old_val
                cost = 0
                for cj in var_to_clauses[vi]:
                    sat = False
                    for vj, sj in clause_vars[cj]:
                        if assignment[vj] == (sj == 1):
                            sat = True
                            break
                    if not sat:
                        cost += 1
                assignment[vi] = old_val
                if cost < best_cost:
                    best_cost = cost
                    best_var = vi
            
            if best_var is not None and best_cost < float('inf'):
                assignment[best_var] = not assignment[best_var]
            else:
                break
        
        return assignment
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random(42)
    
    # Multiple restarts with different strategies
    restart = 0
    while time.time() < deadline:
        restart += 1
        
        # Initialize randomly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Phase 1: WalkSAT with random p_flip
        p_flip = 0.5 + 0.2 * rng.random()  # 0.5 to 0.7
        steps = 5000 + 5000 * rng.random()
        
        for _ in range(int(steps)):
            if time.time() > deadline:
                break
            walksat_step(assignment, rng, p_flip)
        
        # Phase 2: Local search polish
        assignment = local_search(assignment, rng, max_iters=2000)
        
        # Check if solved
        unsat = evaluate(assignment)
        if unsat == 0:
            return assignment
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        # If we've done many restarts, increase step budget slightly
        if restart % 5 == 0:
            steps = int(steps * 1.1)
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
# EVOLVE-BLOCK-END
