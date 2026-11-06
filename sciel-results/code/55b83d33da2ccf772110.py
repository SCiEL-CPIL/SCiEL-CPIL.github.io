"""Synthetic planted 3-SAT solver with clause-weighted WalkSAT and preprocessing."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, use 290s to be safe
    deadline = time.time() + 290.0
    
    # Preprocessing: pure literal elimination and unit propagation
    # First, convert to 0-based
    def preprocess(clauses, n):
        # Pure literal elimination
        # For each variable, check if it appears only with one sign
        var_pos = [set() for _ in range(n)]  # clause indices where var appears positive
        var_neg = [set() for _ in range(n)]  # clause indices where var appears negative
        
        for ci, cl in enumerate(clauses):
            for lit in cl:
                v = abs(lit) - 1
                if lit > 0:
                    var_pos[v].add(ci)
                else:
                    var_neg[v].add(ci)
        
        # Remove pure literals from clauses
        new_clauses = []
        for ci, cl in enumerate(clauses):
            new_cl = []
            for lit in cl:
                v = abs(lit) - 1
                if lit > 0:
                    # Check if v appears negatively anywhere
                    if var_neg[v]:
                        new_cl.append(lit)
                else:
                    # Check if v appears positively anywhere
                    if var_pos[v]:
                        new_cl.append(lit)
            if len(new_cl) > 0:
                new_clauses.append(new_cl)
            # If clause becomes empty, instance is unsat (but we assume satisfiable)
        
        # Unit propagation
        unit_vars = {}  # var -> True/False
        changed = True
        while changed:
            changed = False
            new_clauses2 = []
            for cl in new_clauses:
                # Check if clause is satisfied
                sat = False
                remaining = []
                for lit in cl:
                    v = abs(lit) - 1
                    if v in unit_vars:
                        val = unit_vars[v]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            sat = True
                            break
                    else:
                        remaining.append(lit)
                
                if sat:
                    continue  # Clause is satisfied, remove it
                
                if len(remaining) == 1:
                    # Unit clause
                    lit = remaining[0]
                    v = abs(lit) - 1
                    val = lit > 0
                    if v in unit_vars and unit_vars[v] != val:
                        return None  # Conflict, unsat
                    unit_vars[v] = val
                    changed = True
                    continue
                
                if len(remaining) > 0:
                    new_clauses2.append(remaining)
            
            new_clauses = new_clauses2
        
        return new_clauses, unit_vars
    
    clauses_pp, unit_vars = preprocess(clauses, n)
    m_pp = len(clauses_pp)
    
    # If no clauses left, any assignment satisfying unit_vars works
    if m_pp == 0:
        assignment = [False] * n
        for v, val in unit_vars.items():
            assignment[v] = val
        return assignment
    
    # Build data structures for remaining clauses
    cvars = []
    csigns = []
    for cl in clauses_pp:
        vs = []
        ss = []
        for l in cl:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            vs.append(v)
            ss.append(s)
        cvars.append(vs)
        csigns.append(ss)
    
    # Variable-clause incidence
    var_clauses = [[] for _ in range(n)]
    for ci in range(m_pp):
        for v in cvars[ci]:
            var_clauses[v].append(ci)
    
    # Precompute literal positions for O(1) lookup
    # lit_pos[ci][v] = index of v in clause ci, or -1
    lit_pos = [{} for _ in range(m_pp)]
    for ci in range(m_pp):
        for k, v in enumerate(cvars[ci]):
            lit_pos[ci][v] = k
    
    # Clause weights: frequency of variables in unsat clauses
    # Initialize with clause lengths (shorter clauses are "harder")
    clause_weights = [1.0 / len(cvars[ci]) for ci in range(m_pp)]
    
    def is_satisfied(cl_idx, assignment):
        vs = cvars[cl_idx]
        ss = csigns[cl_idx]
        for k in range(len(vs)):
            v = vs[k]
            s = ss[k]
            if s == 1:
                if assignment[v]:
                    return True
            else:
                if not assignment[v]:
                    return True
        return False
    
    def check_all(assignment):
        for ci in range(m_pp):
            if not is_satisfied(ci, assignment):
                return False
        return True
    
    def check_all_with_units(assignment):
        # Check unit constraints
        for v, val in unit_vars.items():
            if assignment[v] != val:
                return False
        # Check remaining clauses
        return check_all(assignment)
    
    rng = random.Random(42)
    
    def run_walksat():
        nonlocal deadline
        
        # Initial random assignment respecting unit vars
        assignment = [False] * n
        for v, val in unit_vars.items():
            assignment[v] = val
        for v in range(n):
            if v not in unit_vars:
                assignment[v] = rng.random() < 0.5
        
        # Count unsat
        unsat_count = [0] * m_pp
        total_unsat = 0
        unsat_list = []
        for ci in range(m_pp):
            if not is_satisfied(ci, assignment):
                unsat_count[ci] = 1
                total_unsat += 1
                unsat_list.append(ci)
        
        if total_unsat == 0:
            return assignment
        
        # Dynamic noise: starts high, decreases over time
        noise = 0.3
        noise_decay = 0.9995
        min_noise = 0.05
        
        no_improve_steps = 0
        local_best = total_unsat
        
        max_steps = 80000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Update noise
            noise = max(min_noise, noise * noise_decay)
            
            # Pick a random unsat clause (weighted by clause weight)
            if not unsat_list:
                return assignment
            
            # Weighted selection: pick from unsat_list with probability proportional to clause_weights
            ci = rng.choice(unsat_list)
            vs = cvars[ci]
            ss = csigns[ci]
            
            # Calculate scores for each variable in the clause
            # Score: weighted sum of other unsat clauses that would be satisfied
            scores = []
            for v in vs:
                score = 0.0
                for cj in var_clauses[v]:
                    if cj == ci:
                        continue
                    if unsat_count[cj] == 0:
                        continue
                    # Check if flipping v satisfies cj
                    idx_v = lit_pos[cj].get(v, -1)
                    if idx_v == -1:
                        continue
                    s_cj = csigns[cj][idx_v]
                    if s_cj == 1:
                        if not assignment[v]:
                            score += clause_weights[cj]
                    else:
                        if assignment[v]:
                            score += clause_weights[cj]
                scores.append(score)
            
            # Pick variable to flip
            # With probability (1 - noise), pick max score. Else random.
            if rng.random() < (1 - noise):
                max_score = max(scores)
                candidates = [i for i, s in enumerate(scores) if s == max_score]
                idx = rng.choice(candidates)
            else:
                idx = rng.randint(0, len(vs) - 1)
            
            v_to_flip = vs[idx]
            
            # Flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update unsat counts for clauses containing v_to_flip
            for cj in var_clauses[v_to_flip]:
                idx_v = lit_pos[cj].get(v_to_flip, -1)
                if idx_v == -1:
                    continue
                
                s_v = csigns[cj][idx_v]
                lit_was_true = (old_val and s_v == 1) or (not old_val and s_v == -1)
                lit_now_true = (new_val and s_v == 1) or (not new_val and s_v == -1)
                
                was_unsat = unsat_count[cj] > 0
                
                if was_unsat:
                    if lit_now_true:
                        unsat_count[cj] = 0
                        total_unsat -= 1
                        # Remove from unsat_list lazily (mark as 0)
                    else:
                        # Still unsat
                        pass
                else:
                    # Was sat
                    if lit_was_true and not lit_now_true:
                        # Check if any other literal is true
                        any_other_true = False
                        vs_cj = cvars[cj]
                        ss_cj = csigns[cj]
                        for k in range(len(vs_cj)):
                            if k == idx_v:
                                continue
                            v_k = vs_cj[k]
                            s_k = ss_cj[k]
                            if (assignment[v_k] and s_k == 1) or (not assignment[v_k] and s_k == -1):
                                any_other_true = True
                                break
                        if not any_other_true:
                            unsat_count[cj] = 1
                            total_unsat += 1
                            unsat_list.append(cj)
            
            if total_unsat == 0:
                return assignment
            
            # Track best
            if total_unsat < local_best:
                local_best = total_unsat
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Adaptive restart: if stuck for too long, break
            if no_improve_steps > 300:
                break
            
            # Periodic cleanup of unsat_list
            if step % 500 == 0 and len(unsat_list) > total_unsat * 2 + 10:
                unsat_list = [ci for ci in unsat_list if unsat_count[ci] > 0]
        
        return assignment
    
    best_assignment = None
    best_unsat = m_pp + 1
    
    # Main loop: Restarts
    restarts = 0
    while time.time() < deadline:
        if best_unsat == 0:
            break
        
        result = run_walksat()
        if result is not None:
            if check_all_with_units(result):
                return result
            # Update best
            # Count unsat for this result
            cu = 0
            for ci in range(m_pp):
                if not is_satisfied(ci, result):
                    cu += 1
            if cu < best_unsat:
                best_unsat = cu
                best_assignment = result[:]
        
        restarts += 1
    
    if best_assignment is not None:
        if check_all_with_units(best_assignment):
            return best_assignment
        return best_assignment
    
    # Fallback: return assignment satisfying unit vars
    assignment = [False] * n
    for v, val in unit_vars.items():
        assignment[v] = val
    return assignment
