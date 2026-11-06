"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 290s safety margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-compute clause structure: for each clause, list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, list of (clause_idx, is_positive) it appears in
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    def solve_one_restart(rng):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # clause_sat_count[ci] = number of literals in clause ci that are true
        # clause_satisfied[ci] = whether clause ci is satisfied
        clause_sat_count = [0] * m
        clause_satisfied = [False] * m
        unsat_clauses = []
        unsat_index = {}
        
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if assignment[vi] == pos:
                    count += 1
            clause_sat_count[ci] = count
            if count > 0:
                clause_satisfied[ci] = True
            else:
                clause_satisfied[ci] = False
                unsat_clauses.append(ci)
                unsat_index[ci] = len(unsat_clauses) - 1
        
        unsat_count = len(unsat_clauses)
        
        if unsat_count == 0:
            return assignment
        
        # Parameters for weighted WalkSAT
        max_steps = 10000
        p = 0.5  # probability of picking a random literal from the clause
        
        for step in range(max_steps):
            # Check deadline
            if time.time() > deadline:
                return None
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            idx_in_list = rng.randrange(len(unsat_clauses))
            ci = unsat_clauses[idx_in_list]
            
            cv = clause_vars[ci]
            
            # Score each literal in the clause
            scores = []
            
            for vi, pos in cv:
                gain = 0  # clauses that go from unsat to sat
                loss = 0  # clauses that go from sat to unsat
                
                for cj, pos_j in var_clauses[vi]:
                    if cj == ci:
                        continue
                    literal_true = (assignment[vi] == pos_j)
                    if literal_true:
                        if clause_satisfied[cj] and clause_sat_count[cj] == 1:
                            loss += 1
                    else:
                        if not clause_satisfied[cj]:
                            gain += 1
                
                gain += 1 # For clause ci itself
                score = gain - loss
                scores.append((score, vi))
            
            # Choose a literal to flip
            if rng.random() < p:
                vi = rng.choice(cv)[0]
            else:
                best_score = max(s for s, _ in scores)
                candidates = [vi for s, vi in scores if s == best_score]
                vi = rng.choice(candidates)
            
            # Flip variable vi
            assignment[vi] = not assignment[vi]
            
            # Update clause statuses for all clauses containing vi
            for cj, pos_j in var_clauses[vi]:
                literal_now_true = (assignment[vi] == pos_j)
                literal_was_true = not literal_now_true
                
                if literal_now_true and not literal_was_true:
                    # Literal went from false to true
                    clause_sat_count[cj] += 1
                    if not clause_satisfied[cj]:
                        clause_satisfied[cj] = True
                        # Remove cj from unsat_clauses efficiently
                        idx = unsat_index[cj]
                        last_idx = len(unsat_clauses) - 1
                        if idx != last_idx:
                            last_ci = unsat_clauses[last_idx]
                            unsat_clauses[idx] = last_ci
                            unsat_index[last_ci] = idx
                        unsat_clauses.pop()
                        del unsat_index[cj]
                        unsat_count -= 1
                elif not literal_now_true and literal_was_true:
                    # Literal went from true to false
                    clause_sat_count[cj] -= 1
                    if clause_satisfied[cj] and clause_sat_count[cj] == 0:
                        clause_satisfied[cj] = False
                        unsat_clauses.append(cj)
                        unsat_index[cj] = len(unsat_clauses) - 1
                        unsat_count += 1
            
            if unsat_count == 0:
                return assignment
        
        return None
    
    # Multiple restarts
    rng = random.Random(0)
    
    while time.time() < deadline:
        result = solve_one_restart(rng)
        if result is not None:
            # Verify the solution
            valid = True
            for c in clauses:
                sat = False
                for l in c:
                    if result[abs(l) - 1] == (l > 0):
                        sat = True
                        break
                if not sat:
                    valid = False
                    break
            if valid:
                return result
    
    # Return a random assignment as fallback
    return [False] * n
# EVOLVE-BLOCK-END
