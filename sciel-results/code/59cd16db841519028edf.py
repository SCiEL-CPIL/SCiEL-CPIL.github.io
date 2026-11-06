import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Pre-process: store (var_idx, sign) where sign is 1 for pos, -1 for neg
    proc_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            pc.append((v, s))
        proc_clauses.append(pc)
        
    m = len(proc_clauses)
    
    # Build reverse index: var -> list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(proc_clauses):
        for v, s in c:
            var_clauses[v].append((ci, s))
    
    best_assignment = [False] * n
    best_unsat = m
    
    rng = random.Random(42)
    
    def is_clause_sat(ci, assignment):
        for v, s in proc_clauses[ci]:
            if assignment[v] == (s == 1):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for ci in range(m):
            if not is_clause_sat(ci, assignment):
                cnt += 1
        return cnt
    
    # WalkSAT with improved heuristics
    steps_per_restart = 5000
    noise_prob = 0.15
    
    while time.time() < deadline:
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track which clauses are satisfied
        clause_sat = [is_clause_sat(ci, assignment) for ci in range(m)]
        unsat_count = sum(1 for s in clause_sat if not s)
        
        if unsat_count == 0:
            return assignment
        
        # Update best
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT loop
        no_progress_steps = 0
        prev_unsat = unsat_count
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
            
            # Collect unsatisfied clause indices
            unsat_indices = [ci for ci in range(m) if not clause_sat[ci]]
            
            # Pick random unsat clause
            ci = rng.choice(unsat_indices)
            clause = proc_clauses[ci]
            
            # Heuristic: for each literal in clause, compute cost of flipping
            # Cost = (num sat clauses that would become unsat) - (num unsat clauses that would become sat, excluding ci)
            best_var = -1
            best_sign = 1
            best_cost = float('inf')
            
            for v, s in clause:
                # Flipping v to value (s==1)
                new_val = (s == 1)
                current_val = assignment[v]
                
                # Clauses affected are those in var_clauses[v]
                cost = 0
                for (cj, sj) in var_clauses[v]:
                    if cj == ci:
                        continue
                    # Check if clause cj would change state
                    # Currently: clause_sat[cj]
                    # After flip: check if cj is satisfied with new_val
                    # A clause is sat if any literal is true
                    # Literal (v, sj) is true if assignment[v] == (sj==1)
                    
                    # Check if cj has another literal that's currently true
                    # If yes, cj stays sat regardless of v's flip
                    # If no, cj's state depends on v's literal
                    
                    # Efficient check: does cj have any true literal other than v's?
                    has_other_true = False
                    for (vv, ss) in proc_clauses[cj]:
                        if vv != v and assignment[vv] == (ss == 1):
                            has_other_true = True
                            break
                    
                    if not has_other_true:
                        # cj's state depends on v's literal
                        currently_true = (assignment[v] == (sj == 1))
                        after_true = (new_val == (sj == 1))
                        
                        if clause_sat[cj] and not after_true:
                            # cj was sat, becomes unsat -> bad
                            cost += 1
                        elif not clause_sat[cj] and after_true:
                            # cj was unsat, becomes sat -> good
                            cost -= 1
                
                if cost < best_cost:
                    best_cost = cost
                    best_var = v
                    best_sign = s
            
            # With prob noise, pick random literal from clause
            if rng.random() < noise_prob:
                v, s = rng.choice(clause)
            else:
                v, s = best_var, best_sign
            
            # Flip
            new_val = (s == 1)
            old_val = assignment[v]
            if new_val == old_val:
                continue
                
            assignment[v] = new_val
            
            # Update clause satisfaction for all clauses containing v
            for (cj, sj) in var_clauses[v]:
                was_sat = clause_sat[cj]
                clause_sat[cj] = is_clause_sat(cj, assignment)
                if was_sat and not clause_sat[cj]:
                    unsat_count += 1
                elif not was_sat and clause_sat[cj]:
                    unsat_count -= 1
            
            # Check progress
            if unsat_count < prev_unsat:
                no_progress_steps = 0
                prev_unsat = unsat_count
            else:
                no_progress_steps += 1
            
            # Restart if stuck
            if no_progress_steps > 500:
                break
            
            # Update best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
    
    return best_assignment
