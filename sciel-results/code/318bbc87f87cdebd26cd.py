import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Preprocessing: build efficient data structures
    # clause_vars[i] = list of (var_idx, sign) for clause i
    # var_to_clauses[v] = list of (clause_idx, pos) for variable v
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cv = []
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
            var_to_clauses[v].append((i, j))
        clause_vars.append(cv)
    
    # Precompute for each variable and each clause it appears in,
    # the other two literals (var, sign) to speed up delta computation
    # var_other_lits[v] = dict: clause_idx -> (other_var1, other_sign1, other_var2, other_sign2)
    var_other_lits = [{} for _ in range(n)]
    for v in range(n):
        for (ci, pos) in var_to_clauses[v]:
            others = []
            for p in range(3):
                if p != pos:
                    others.append(clause_vars[ci][p])
            var_other_lits[v][ci] = (others[0][0], others[0][1], others[1][0], others[1][1])
    
    # Fast evaluation: returns (num_unsat, unsat_clause_set)
    # assignment is list of bools (length n)
    # lit_sat[c] = whether clause c is satisfied
    def fast_evaluate(assignment):
        num_unsat = 0
        unsat_clauses = []
        for c in range(C):
            cv = clause_vars[c]
            sat = False
            for (v, s) in cv:
                val = assignment[v]
                if (s == 1 and val) or (s == -1 and not val):
                    sat = True
                    break
            if sat:
                num_unsat -= 1  # was unsat, now sat
            else:
                num_unsat += 1  # now unsat
                unsat_clauses.append(c)
        return num_unsat, unsat_clauses
    
    # Compute delta for flipping variable v given current assignment
    # Returns (delta, new_unsat_count)
    def compute_delta(v, assignment, num_unsat, unsat_set):
        delta = 0
        for (ci, pos) in var_to_clauses[v]:
            # Get the other two literals
            o1v, o1s, o2v, o2s = var_other_lits[v][ci]
            
            # Check if the other two literals are satisfied
            o1_sat = (o1s == 1 and assignment[o1v]) or (o1s == -1 and not assignment[o1v])
            o2_sat = (o2s == 1 and assignment[o2v]) or (o2s == -1 and not assignment[o2v])
            
            if o1_sat and o2_sat:
                # If clause is currently unsat, flipping v satisfies it
                if ci in unsat_set:
                    delta += 1
            else:
                # If clause is currently sat, check if flipping v would unsatisfy it
                # The clause is sat iff at least one literal is sat
                # Currently sat means: (v's lit sat) OR (o1 sat) OR (o2 sat)
                # Since we're flipping v, after flip v's lit will be unsat
                # So clause remains sat iff o1_sat OR o2_sat
                # If NOT (o1_sat OR o2_sat), i.e., both unsat, then clause becomes unsat
                if not o1_sat and not o2_sat:
                    # This clause is currently sat (because v's lit is sat)
                    # After flip, it becomes unsat
                    if ci not in unsat_set:
                        delta -= 1
        return delta
    
    # WalkSAT with min-conflicts style heuristic
    def walk_sat(assignment, num_unsat, unsat_clauses, rng, max_steps, deadline):
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True
            
            # Pick a random unsatisfied clause
            c = rng.choice(unsat_clauses)
            cv = clause_vars[c]
            
            # For each variable in this clause, compute delta
            best_v = -1
            best_score = -1e9
            for (v, s) in cv:
                delta = compute_delta(v, assignment, num_unsat, set(unsat_clauses))
                # Add random noise for diversity
                noise = rng.random() * 0.5
                score = delta + noise
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            # Flip best_v
            assignment[best_v] = not assignment[best_v]
            
            # Update num_unsat and unsat_clauses incrementally
            # Recompute affected clauses
            affected = set()
            for (ci, pos) in var_to_clauses[best_v]:
                affected.add(ci)
            
            # Remove affected clauses from unsat_clauses if they become sat
            new_unsat = []
            for ci in unsat_clauses:
                if ci in affected:
                    # Check if clause is now satisfied
                    cv = clause_vars[ci]
                    sat = False
                    for (v, s) in cv:
                        val = assignment[v]
                        if (s == 1 and val) or (s == -1 and not val):
                            sat = True
                            break
                    if not sat:
                        new_unsat.append(ci)
                else:
                    new_unsat.append(ci)
            
            # Add affected clauses that became unsat
            for ci in affected:
                if ci not in set(new_unsat):
                    cv = clause_vars[ci]
                    sat = False
                    for (v, s) in cv:
                        val = assignment[v]
                        if (s == 1 and val) or (s == -1 and not val):
                            sat = True
                            break
                    if not sat:
                        new_unsat.append(ci)
            
            unsat_clauses = new_unsat
            num_unsat = len(unsat_clauses)
            
            if num_unsat == 0:
                return True
        
        return False
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    
    best_num_unsat = C + 1
    best_assignment = None
    
    # Multiple restarts with increasing step limits
    restart = 0
    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Evaluate
        num_unsat, unsat_clauses = fast_evaluate(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive step limit
        base_steps = 5000 + restart * 1000
        max_steps = min(base_steps, 100000)
        
        solved = walk_sat(assignment, num_unsat, unsat_clauses, rng, max_steps, deadline)
        
        if solved:
            return assignment
        
        restart += 1
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
