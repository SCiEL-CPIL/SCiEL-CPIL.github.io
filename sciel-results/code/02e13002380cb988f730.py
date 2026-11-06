"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock limit, use 325s to be safe
    deadline = time.time() + 325.0
    
    # Preprocess: build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    clause_vars = []
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            v = abs(l) - 1
            pos = l > 0
            cv.append((v, pos))
            var_clauses[v].append((i, pos))
        clause_vars.append(cv)
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            sat = False
            for v, pos in clause_vars[i]:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def solve_once(seed):
        rng = random.Random(seed)
        
        # Initialize assignment randomly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for i in range(m):
            for v, pos in clause_vars[i]:
                if assignment[v] == pos:
                    sat_count[i] += 1
        
        # unsat_clauses: set of clause indices that are not satisfied
        unsat_clauses = set(i for i in range(m) if sat_count[i] == 0)
        
        if not unsat_clauses:
            return assignment
        
        # WalkSAT with damage-minimizing heuristic
        gamma = 0.6
        max_steps = 80000
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            clause_idx = next(iter(unsat_clauses))
            clause = clause_vars[clause_idx]
            
            # Score each literal
            best_var = -1
            best_score = float('inf')
            best_pos = False
            
            if rng.random() < gamma:
                # Pick from literals that satisfy this clause
                candidates = [(v, pos) for v, pos in clause if assignment[v] != pos]
                if not candidates:
                    candidates = clause
            else:
                candidates = clause
            
            for v, pos in candidates:
                damage = 0
                for ci, cpos in var_clauses[v]:
                    if ci == clause_idx:
                        continue
                    if assignment[v] == cpos:
                        if sat_count[ci] == 1:
                            damage += 1
                score = damage
                if score < best_score:
                    best_score = score
                    best_var = v
                    best_pos = pos
            
            if best_var < 0:
                v = rng.randrange(n)
            else:
                v = best_var
            
            new_val = not assignment[v]
            
            for ci, cpos in var_clauses[v]:
                if assignment[v] == cpos:
                    sat_count[ci] -= 1
                    if sat_count[ci] == 0:
                        unsat_clauses.add(ci)
                else:
                    sat_count[ci] += 1
                    if sat_count[ci] > 0 and ci in unsat_clauses:
                        unsat_clauses.discard(ci)
            
            assignment[v] = new_val
        
        return assignment
    
    # Try multiple restarts
    best_assignment = None
    best_unsat = float('inf')
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        
        assignment = solve_once(seed)
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
    
    if best_assignment is not None:
        return best_assignment
    
    return [False] * n
# EVOLVE-BLOCK-END
