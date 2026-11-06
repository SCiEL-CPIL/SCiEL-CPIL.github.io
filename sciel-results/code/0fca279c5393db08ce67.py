import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 330s limit, leave 40s margin
    
    m = len(clauses)
    
    # Pre-process clauses
    # Store as list of lists of (var_idx, is_pos)
    proc_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            pc.append((v, p))
        proc_clauses.append(pc)
        
    # Build incidence lists: for each variable, which clauses contain it and with what sign?
    # var_clauses[v] = list of (clause_idx, is_pos)
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(proc_clauses):
        for v, p in c:
            var_clauses[v].append((i, p))
            
    # State:
    # assignment: list of bools
    # clause_sat: list of bools indicating if clause is satisfied
    # unsat_count: int
    
    def check_clause(clause, assignment):
        for v, p in clause:
            if assignment[v] == p:
                return True
        return False

    def update_clause_status(ci, assignment):
        return check_clause(proc_clauses[ci], assignment)

    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random()
    
    while time.time() < deadline:
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial status check
        clause_sat = [check_clause(c, assignment) for c in proc_clauses]
        unsat_indices = [i for i, s in enumerate(clause_sat) if not s]
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
        # WalkSAT loop
        # Heuristic parameters
        # p_noise: probability of random flip
        # We can adapt this, but fixed 0.1-0.3 is usually good.
        p_noise = 0.2
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return assignment
                
            # Pick a random unsat clause
            ci = rng.choice(unsat_indices)
            clause = proc_clauses[ci]
            
            # Evaluate each variable in the clause
            best_var = -1
            best_score = float('-inf')
            best_p = False
            scores = []
            
            for v, p in clause:
                # Simulate flipping v to p
                # Current state: assignment[v]
                # If we set assignment[v] to p:
                # 1. Clause ci becomes satisfied (since it was unsat, and p is a literal in it).
                #    Actually, we need to check if it was the *only* way? No, if we set v to p, 
                #    literal (v,p) becomes true, so clause is sat.
                
                # We need to count:
                # gain: number of *other* unsat clauses that become satisfied.
                # loss: number of *satisfied* clauses that become unsatisfied.
                
                gain = 0
                loss = 0
                
                # Check all clauses containing v
                for cj_idx, cj_p in var_clauses[v]:
                    if cj_idx == ci:
                        continue
                    
                    # Determine if flipping v to p affects this clause's status
                    # Current status of cj_idx is clause_sat[cj_idx]
                    
                    # If clause is currently UNSAT:
                    if not clause_sat[cj_idx]:
                        # It becomes SAT if it contains the literal (v, p) that we are setting.
                        # Because currently all literals are false.
                        # If we set v to p, and the clause has (v, p), that literal becomes true -> SAT.
                        # If the clause has (v, not p), that literal remains false (was false, stays false).
                        if cj_p == p:
                            gain += 1
                    # If clause is currently SAT:
                    else:
                        # It becomes UNSAT if:
                        # 1. It contains the literal (v, not p).
                        # 2. That literal was the *only* one making it true.
                        #    i.e., all other literals in the clause are false.
                        
                        if cj_p != p:
                            # The literal (v, not p) is in the clause.
                            # Currently, since clause is SAT, at least one literal is true.
                            # If (v, not p) is true, it means assignment[v] == not p.
                            # We are changing assignment[v] to p. So (v, not p) becomes false.
                            # We need to check if any OTHER literal in the clause is true.
                            other_sat = False
                            for vv, pp in proc_clauses[cj_idx]:
                                if vv == v:
                                    continue
                                if assignment[vv] == pp:
                                    other_sat = True
                                    break
                            if not other_sat:
                                loss += 1
                
                score = gain - loss
                scores.append((score, v, p))
                
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_p = p
            
            # Choose variable to flip
            if rng.random() < p_noise:
                # Random choice from clause
                v, p = rng.choice(clause)
            else:
                # Greedy choice (max score). Tie-break random if needed?
                # For simplicity, pick the first best found.
                v, p = best_var, best_p
                
            # Perform flip
            old_val = assignment[v]
            assignment[v] = p
            
            # Update clause statuses
            # Only clauses containing v can change status
            for cj_idx, _ in var_clauses[v]:
                old_sat = clause_sat[cj_idx]
                new_sat = check_clause(proc_clauses[cj_idx], assignment)
                
                if old_sat != new_sat:
                    clause_sat[cj_idx] = new_sat
                    if old_sat:
                        # Was sat, now unsat
                        unsat_indices.append(cj_idx)
                        unsat_count += 1
                    else:
                        # Was unsat, now sat
                        # Remove from unsat_indices (O(N) operation, but N is small)
                        # Optimization: Mark as removed and clean up occasionally, or just rebuild if list gets too big?
                        # For n=50, m~100-1000, linear search in unsat_indices is okay.
                        try:
                            unsat_indices.remove(cj_idx)
                        except ValueError:
                            pass
                        unsat_count -= 1
            
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
        # End of restart
        # If we didn't solve, continue to next restart
        # We could also try to restart from best_assignment if it's significantly better
        if best_unsat < unsat_count and rng.random() < 0.1:
            # 10% chance to restart from best known state
            assignment = best_assignment[:]
            clause_sat = [check_clause(c, assignment) for c in proc_clauses]
            unsat_indices = [i for i, s in enumerate(clause_sat) if not s]
            unsat_count = len(unsat_indices)

    if best_assignment is not None:
        return best_assignment
    return assignment
