import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based
    clauses_0based = []
    for clause in clauses:
        c = []
        for l in clause:
            if l > 0:
                c.append(l - 1)
            else:
                c.append(-l - 1)
        clauses_0based.append(c)
        
    m = len(clauses_0based)
    
    # Build var to clauses map
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for lit in clauses_0based[i]:
            var = abs(lit)
            var_to_clauses[var].append(i)
            
    def is_clause_satisfied(clause_idx, assignment):
        clause = clauses_0based[clause_idx]
        for lit in clause:
            var = abs(lit)
            val = assignment[var]
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False

    deadline = time.time() + 290.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(42)
    
    p_greedy = 0.9
    restart_interval = 10000
    max_restarts = 100000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            assignment = best_assignment[:]
            # Perturb
            for _ in range(n // 5):
                idx = rng.randint(0, n - 1)
                assignment[idx] = not assignment[idx]
        
        # Initialize clause states
        clause_sat = [False] * m
        unsat_clauses = []
        unsat_count = 0
        clause_weights = [1] * m # Initialize weights
        
        for i in range(m):
            if is_clause_satisfied(i, assignment):
                clause_sat[i] = True
            else:
                clause_sat[i] = False
                unsat_clauses.append(i)
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        steps = 0
        while steps < restart_interval:
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
            
            # Weighted selection of unsat clause
            # Sum of weights
            # To avoid O(M) sum every step, we can just pick random from list 
            # and rely on the list composition. 
            # But weighted is better.
            # Simple weighted pick:
            # Create cumulative sum? O(M).
            # For M=300, O(M) is fine.
            
            # Let's just pick random from unsat_clauses for speed, 
            # as maintaining exact weights with removals is complex.
            # The "hardness" is implicitly handled by the fact that 
            # hard clauses stay unsat longer, but they are all in the list.
            # Actually, random pick from unsat list is standard WalkSAT.
            
            idx = rng.choice(unsat_clauses)
            clause = clauses_0based[idx]
            
            # Find satisfying vars
            satisfying_vars = []
            for lit in clause:
                var = abs(lit)
                cur_val = assignment[var]
                if (lit > 0 and not cur_val) or (lit < 0 and cur_val):
                    satisfying_vars.append(var)
            
            if satisfying_vars:
                if rng.random() < p_greedy:
                    var_to_flip = rng.choice(satisfying_vars)
                else:
                    var_to_flip = rng.choice([abs(lit) for lit in clause])
            else:
                var_to_flip = rng.choice([abs(lit) for lit in clause])
            
            # Flip
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Update affected clauses
            for ci in var_to_clauses[var_to_flip]:
                new_sat = is_clause_satisfied(ci, assignment)
                old_sat = clause_sat[ci]
                
                if old_sat == new_sat:
                    continue
                    
                clause_sat[ci] = new_sat
                
                if new_sat:
                    # Was unsat, now sat
                    unsat_count -= 1
                    try:
                        unsat_clauses.remove(ci)
                    except ValueError:
                        pass
                else:
                    # Was sat, now unsat
                    unsat_count += 1
                    unsat_clauses.append(ci)
                    
            steps += 1
            
        # End of restart interval
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
    return best_assignment if best_assignment else [False] * n
