"""Improved planted 3-SAT solver with efficient WalkSAT + restarts."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    m = len(clauses)
    
    # Preprocess: convert to 0-based, store clause data
    # clause_lits[ci] = list of (var_idx, sign) where sign=1 means +var, -1 means -var
    # var_clauses[v] = list of clause indices containing variable v
    # For fast delta computation, also store for each variable v and each clause ci containing v,
    # the indices of the other two literals in ci (for quick "satisfied by others" check)
    
    clause_lits = []
    var_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        cl = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cl.append((v, s))
            var_clauses[v].append(ci)
        clause_lits.append(cl)
    
    # For each clause, precompute for each position the other two positions
    # clause_other[ci][k] = (j, l) where j is the other position, l is the literal at that position
    # This lets us check "is clause satisfied by other literals" in O(1)
    clause_other = []
    for ci in range(m):
        cl = clause_lits[ci]
        others = []
        for k in range(len(cl)):
            # The other two literals
            o = []
            for j in range(len(cl)):
                if j != k:
                    o.append((j, cl[j]))
            others.append(o)
        clause_other.append(others)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # WalkSAT with efficient tracking
    # assignment: list of booleans (True = var is True)
    # clause_sat[ci]: whether clause ci is satisfied
    # num_unsat: count of unsatisfied clauses
    
    def init_state(assignment):
        clause_sat = [False] * m
        num_unsat = 0
        for ci in range(m):
            sat = False
            for v, s in clause_lits[ci]:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            clause_sat[ci] = sat
            if not sat:
                num_unsat += 1
        return clause_sat, num_unsat
    
    def count_unsat(assignment):
        cnt = 0
        for ci in range(m):
            sat = False
            for v, s in clause_lits[ci]:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    # Main search: multiple restarts with WalkSAT
    # Use adaptive step budget and noise schedule
    
    restart = 0
    max_restarts = 5000
    base_steps = 3000
    
    while time.time() < deadline:
        if restart >= max_restarts:
            break
            
        rng = random.Random(restart * 7919 + 31)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, num_unsat = init_state(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
        
        # Adaptive steps: increase slightly with restart count, but cap by time
        steps = base_steps + restart * 10
        if steps > 10000:
            steps = 10000
        
        # Noise parameter: start lower, increase with steps within a restart
        # This helps escape local minima
        initial_noise = 0.0
        noise_increment = 0.1 / max(1, steps // 10)
        
        for step in range(steps):
            if time.time() >= deadline:
                break
                
            if num_unsat == 0:
                return assignment
            
            # Find an unsatisfied clause
            # To avoid O(m) scan every step, we can keep a list of unsat clauses
            # But for simplicity and correctness, let's do a quick scan
            # Optimization: pick a random clause, if unsat use it, else scan
            ci = rng.randrange(m)
            if clause_sat[ci]:
                # Need to find an unsat clause
                found = False
                for trial in range(20):
                    ci2 = rng.randrange(m)
                    if not clause_sat[ci2]:
                        ci = ci2
                        found = True
                        break
                if not found:
                    # Fallback: full scan
                    for ci2 in range(m):
                        if not clause_sat[ci2]:
                            ci = ci2
                            found = True
                            break
                    if not found:
                        return assignment
            
            # For each literal in this unsat clause, compute the delta
            # delta = (# clauses that become unsat) - (# clauses that become sat)
            # We want to minimize delta (i.e., maximize gain = -delta)
            
            cl = clause_lits[ci]
            best_var = -1
            best_score = float('inf')
            
            noise = initial_noise + step * noise_increment
            noise = min(noise, 0.8)
            
            for k, (v, s) in enumerate(cl):
                # Compute gain if we flip v
                gain = 0
                for c_idx in var_clauses[v]:
                    # Check if c_idx is satisfied by literals other than v
                    others = clause_other[c_idx][k] if c_idx == ci else None
                    if c_idx == ci:
                        # We already know the other literals' positions
                        sat_by_others = False
                        for j, (v2, s2) in others:
                            if assignment[v2] == (s2 == 1):
                                sat_by_others = True
                                break
                    else:
                        # Need to find v's position in c_idx and check others
                        sat_by_others = False
                        for j, (v2, s2) in enumerate(clause_lits[c_idx]):
                            if v2 == v:
                                # Found v, check other literals
                                for j2, (v3, s3) in enumerate(clause_lits[c_idx]):
                                    if j2 != j:
                                        if assignment[v3] == (s3 == 1):
                                            sat_by_others = True
                                            break
                                break
                            elif assignment[v2] == (s2 == 1):
                                sat_by_others = True
                                break
                    
                    if sat_by_others:
                        continue
                    else:
                        # Clause is satisfied iff v is set to satisfy it
                        if clause_sat[c_idx]:
                            # Currently sat, so v is satisfying it. Flipping makes it unsat.
                            gain -= 1
                        else:
                            # Currently unsat, so v is NOT satisfying it. Flipping makes it sat.
                            gain += 1
                
                score = -gain + rng.random() * noise
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var == -1:
                v = rng.randrange(n)
            else:
                v = best_var
            
            # Flip v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat and num_unsat for all clauses containing v
            for c_idx in var_clauses[v]:
                # Determine new sat status
                new_sat = False
                for v2, s2 in clause_lits[c_idx]:
                    if assignment[v2] == (s2 == 1):
                        new_sat = True
                        break
                
                old_sat = clause_sat[c_idx]
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                clause_sat[c_idx] = new_sat
            
            if num_unsat == 0:
                return assignment
            
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
        
        restart += 1
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
