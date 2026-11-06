"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, use 295s to be safe
    deadline = time.time() + 295.0

    # Convert 1-based signed literals to 0-based var indices and signs
    cvars = []
    csigns = []
    for cl in clauses:
        vs = []
        ss = []
        for l in cl:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            vs.append(v)
            ss.append(s)
        cvars.append(vs)
        csigns.append(ss)
        
    # Build variable-clause incidence
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v in cvars[ci]:
            var_clauses[v].append(ci)
            
    # Precompute for each clause and each variable in it, the index of that variable
    # and the sign, to speed up lookups
    # clause_var_index[ci] = dict mapping var -> (index_in_clause, sign)
    clause_var_info = []
    for ci in range(m):
        info = {}
        for k, v in enumerate(cvars[ci]):
            info[v] = (k, csigns[ci][k])
        clause_var_info.append(info)
        
    # Also precompute: for each variable v, for each clause cj containing v, 
    # the sign of v in cj
    # var_clause_sign[v] = list of (cj, sign) for each clause cj containing v
    var_clause_sign = [[] for _ in range(n)]
    for v in range(n):
        for cj in var_clauses[v]:
            k, s = clause_var_info[cj][v]
            var_clause_sign[v].append((cj, s))
            
    def check_all(assignment):
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]:
                        sat = True
                        break
                else:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                return False
        return True

    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    
    def run_walksat():
        nonlocal best_assignment, best_unsat, deadline
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Count unsat clauses
        unsat_count = [0] * m
        total_unsat = 0
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]:
                        sat = True
                        break
                else:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                unsat_count[ci] = 1
                total_unsat += 1
                
        if total_unsat == 0:
            return assignment
            
        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]
            
        # Maintain list of unsat clause indices
        unsat_clauses = [ci for ci in range(m) if unsat_count[ci] > 0]
        
        # Adaptive noise: start with high randomness, decrease over time
        # Use a schedule based on steps
        max_steps = 200000
        noise_start = 0.3
        noise_end = 0.05
        noise_decay = 0.999995
        
        noise = noise_start
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_clauses:
                return assignment
                
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
                
            # Pick a random unsat clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            vs = cvars[ci]
            ss = csigns[ci]
            
            # Calculate scores for each variable in the clause
            # Score: number of OTHER unsat clauses that would be satisfied if we flip this var
            scores = []
            for v in vs:
                score = 0
                for cj, s_cj in var_clause_sign[v]:
                    if cj == ci:
                        continue
                    if unsat_count[cj] == 0:
                        continue
                    # Check if flipping v satisfies cj
                    # If s_cj == 1, we want True. If assignment[v] is False, flipping to True satisfies.
                    # If s_cj == -1, we want False. If assignment[v] is True, flipping to False satisfies.
                    if s_cj == 1:
                        if not assignment[v]:
                            score += 1
                    else:
                        if assignment[v]:
                            score += 1
                scores.append(score)
                
            # Pick variable to flip
            # With probability (1-noise), pick max score. Else random.
            if rng.random() < (1 - noise):
                max_score = max(scores)
                # Pick random among those with max score
                candidates = [i for i, s in enumerate(scores) if s == max_score]
                idx = candidates[rng.randrange(len(candidates))]
            else:
                idx = rng.randrange(len(vs))
                
            v_to_flip = vs[idx]
            
            # Flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update unsat counts for clauses containing v_to_flip
            for cj, s_v in var_clause_sign[v_to_flip]:
                was_unsat = unsat_count[cj] > 0
                
                # Determine if the literal for v_to_flip in cj was true before and is true after
                lit_was_true = (old_val and s_v == 1) or (not old_val and s_v == -1)
                lit_now_true = (new_val and s_v == 1) or (not new_val and s_v == -1)
                
                if was_unsat:
                    if lit_now_true:
                        # Now satisfied
                        unsat_count[cj] = 0
                        total_unsat -= 1
                        # Remove from unsat_clauses list
                        # Find and remove cj
                        # For efficiency, we could use a set, but list is fine for moderate m
                        # Actually, let's rebuild the list periodically or use a different structure
                        # For now, we'll mark it and rebuild when needed
                        pass
                    else:
                        # Still unsat
                        pass
                else:
                    # Was sat
                    if lit_was_true and not lit_now_true:
                        # We flipped the literal that was true. Check if any other literal is true
                        any_other_true = False
                        vs_cj = cvars[cj]
                        ss_cj = csigns[cj]
                        for k in range(len(vs_cj)):
                            if vs_cj[k] == v_to_flip:
                                continue
                            v_k = vs_cj[k]
                            s_k = ss_cj[k]
                            if (assignment[v_k] and s_k == 1) or (not assignment[v_k] and s_k == -1):
                                any_other_true = True
                                break
                        if not any_other_true:
                            # Now unsat
                            unsat_count[cj] = 1
                            total_unsat += 1
                    # Else still sat
            
            # Periodically rebuild unsat_clauses list to keep it accurate
            if step % 100 == 0:
                unsat_clauses = [ci2 for ci2 in range(m) if unsat_count[ci2] > 0]
                
            if total_unsat == 0:
                return assignment
                
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]
                
            # Decay noise
            noise = max(noise_end, noise * noise_decay)
            
        return assignment

    # Main loop: Restarts
    # We use adaptive restarts: if we don't improve, we restart sooner.
    restarts = 0
    last_best_unsat = best_unsat
    restart_count = 0
    
    while time.time() < deadline:
        if best_unsat == 0:
            break
            
        result = run_walksat()
        if result is not None:
            # Verify
            if check_all(result):
                return result
        
        restarts += 1
        restart_count += 1
        
        # Adaptive restart: if no improvement in last restart, restart sooner
        # (Already handled by run_walksat's internal step limit and time check)
        
    if best_assignment is not None:
        # Final check
        if check_all(best_assignment):
            return best_assignment
        # If not satisfied, return it anyway (will score 0 for this instance)
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
