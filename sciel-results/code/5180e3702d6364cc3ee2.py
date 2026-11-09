"""Planted 3-SAT solver: Optimized WalkSAT with incremental unsat tracking."""
import time
import random
import numpy as np


def solve(instance):
    n_orig = instance["n"]
    clauses_orig = instance["clauses"]
    m_orig = len(clauses_orig)

    start_time = time.time()
    deadline = start_time + 285.0

    # Convert to 0-based
    clause_vars = []
    for clause in clauses_orig:
        cv = [(abs(l) - 1, 1 if l > 0 else -1) for l in clause]
        clause_vars.append(cv)

    n = n_orig
    m = m_orig

    # Build var_to_clauses
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v, s in clause_vars[ci]:
            var_to_clauses[v].append((ci, s))

    # Simple preprocessing: remove satisfied clauses, unit propagation
    # We'll do a light version to avoid overhead
    assignment_fixed = {}  # var -> bool from preprocessing
    active_clauses = []
    
    # Unit propagation (one pass)
    changed = True
    while changed and time.time() < deadline - 270:
        changed = False
        new_active = []
        for ci in range(len(active_clauses) if active_clauses else m):
            if active_clauses:
                cv = active_clauses[ci]
            else:
                cv = clause_vars[ci]
            
            # Check if satisfied
            sat = False
            for v, s in cv:
                if v in assignment_fixed:
                    val = assignment_fixed[v]
                    if (val and s == 1) or (not val and s == -1):
                        sat = True
                        break
            
            if sat:
                changed = True
                continue
            
            # Count unassigned literals
            unassigned = [(v, s) for v, s in cv if v not in assignment_fixed]
            
            if len(unassigned) == 0:
                # UNSAT - shouldn't happen for planted
                return [False] * n_orig
            
            if len(unassigned) == 1:
                v, s = unassigned[0]
                assignment_fixed[v] = (s == 1)
                changed = True
                continue
            
            new_active.append(cv)
        
        if not active_clauses:
            active_clauses = new_active
        else:
            active_clauses = new_active
    
    # If no preprocessing was done, use original
    if not active_clauses:
        active_clauses = clause_vars
    
    m = len(active_clauses)
    
    # Map remaining vars to compact indices
    remaining_vars = set()
    for cv in active_clauses:
        for v, s in cv:
            if v not in assignment_fixed:
                remaining_vars.add(v)
    
    var_list = sorted(remaining_vars)
    var_map = {v: i for i, v in enumerate(var_list)}
    n_comp = len(var_list)
    
    if n_comp == 0:
        full = [False] * n_orig
        for v, val in assignment_fixed.items():
            full[v] = val
        return full

    # Build compact clause structure
    compact_clauses = []
    compact_var_to_clauses = [[] for _ in range(n_comp)]
    for cv in active_clauses:
        new_cv = []
        for v, s in cv:
            if v in var_map:
                nv = var_map[v]
                new_cv.append((nv, s))
                compact_var_to_clauses[nv].append((len(compact_clauses), s))
        if new_cv:
            compact_clauses.append(new_cv)
    
    m_comp = len(compact_clauses)
    
    if m_comp == 0:
        full = [False] * n_orig
        for v, val in assignment_fixed.items():
            full[v] = val
        return full

    # WalkSAT with incremental unsat tracking
    def walksat_incremental(clauses_local, var_to_clauses_local, n_local, time_budget, rng):
        m_local = len(clauses_local)
        if m_local == 0:
            return np.zeros(n_local, dtype=bool)
        
        # Initial random assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n_local)], dtype=bool)
        
        # Compute initial sat counts per clause
        clause_sat_count = np.zeros(m_local, dtype=np.int32)
        for ci in range(m_local):
            cnt = 0
            for v, s in clauses_local[ci]:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            clause_sat_count[ci] = cnt
        
        unsat = int(np.sum(clause_sat_count == 0))
        if unsat == 0:
            return assignment
        
        best_assign = assignment.copy()
        best_unsat = unsat
        
        # Luby restart sequence
        luby_seq = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 16, 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 16, 32]
        luby_idx = 0
        restart_num = 0
        p_good = 0.55
        
        while time.time() < time_budget:
            if restart_num > 0:
                assignment = np.array([rng.random() < 0.5 for _ in range(n_local)], dtype=bool)
                for ci in range(m_local):
                    cnt = 0
                    for v, s in clauses_local[ci]:
                        if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                            cnt += 1
                    clause_sat_count[ci] = cnt
                unsat = int(np.sum(clause_sat_count == 0))
                if unsat == 0:
                    return assignment
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assign = assignment.copy()
            
            # Get steps from Luby sequence
            if luby_idx < len(luby_seq):
                luby_mult = luby_seq[luby_idx]
                luby_idx += 1
            else:
                luby_mult = 16
                luby_idx = 31  # reset to high value
            
            base_steps = 200
            if n_local > 80:
                base_steps = 400
            steps = base_steps * luby_mult
            
            stagnant = 0
            max_stagnant = 80 + n_local // 2
            
            for step in range(steps):
                if time.time() > time_budget:
                    break
                if unsat == 0:
                    return assignment
                
                # Pick random unsat clause
                unsat_mask = clause_sat_count == 0
                unsat_indices = np.where(unsat_mask)[0]
                if len(unsat_indices) == 0:
                    return assignment
                ci = int(unsat_indices[rng.randint(0, len(unsat_indices) - 1)])
                lits = clauses_local[ci]
                
                if rng.random() < p_good:
                    # Gain heuristic: find var in this clause that when flipped
                    # satisfies it and also helps other unsat clauses
                    best_var = lits[0][0]
                    best_gain = -1
                    for v, s in lits:
                        # Would flipping v satisfy this clause?
                        current_satisfies = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
                        if current_satisfies:
                            # Flipping would unsatisfy, skip
                            continue
                        # Count other unsat clauses containing v
                        gain = 0
                        for (aci, asgn) in var_to_clauses_local[v]:
                            if aci == ci:
                                continue
                            if clause_sat_count[aci] == 0:
                                # Check if flipping v would satisfy aci
                                av = assignment[v]
                                would_sat = (not av and asgn == 1) or (av and asgn == -1)
                                if would_sat:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_var = v
                    v = best_var
                else:
                    v = lits[rng.randint(0, len(lits) - 1)][0]
                
                # Flip v and update clause_sat_count incrementally
                old_val = assignment[v]
                new_val = not old_val
                assignment[v] = new_val
                
                # Update sat counts for all clauses containing v
                for (aci, asgn) in var_to_clauses_local[v]:
                    was_sat = (old_val and asgn == 1) or (not old_val and asgn == -1)
                    is_sat = (new_val and asgn == 1) or (not new_val and asgn == -1)
                    if was_sat and not is_sat:
                        clause_sat_count[aci] -= 1
                        if clause_sat_count[aci] == 0:
                            unsat += 1
                    elif not was_sat and is_sat:
                        clause_sat_count[aci] += 1
                        if clause_sat_count[aci] == 1:
                            unsat -= 1
                
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assign = assignment.copy()
                
                # Track stagnation
                if unsat >= best_unsat:
                    stagnant += 1
                else:
                    stagnant = 0
                
                if stagnant > max_stagnant:
                    break
            
            restart_num += 1
        
        return best_assign

    # Try multiple seeds
    best_overall = None
    best_unsat_overall = m_comp + 1
    
    seeds_to_try = [42, 123, 777, 2024, 999, 555, 111]
    remaining_time = deadline - time.time()
    time_per_seed = remaining_time * 0.8 / len(seeds_to_try)
    
    for seed in seeds_to_try:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        budget = time.time() + time_per_seed
        result = walksat_incremental(compact_clauses, compact_var_to_clauses, n_comp, budget, rng)
        if result is not None:
            # Check if satisfied
            all_sat = True
            for ci in range(m_comp):
                sat = False
                for v, s in compact_clauses[ci]:
                    if (result[v] and s == 1) or (not result[v] and s == -1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                # Build full assignment
                full = [False] * n_orig
                for v, val in assignment_fixed.items():
                    full[v] = val
                for i, v in enumerate(var_list):
                    full[v] = bool(result[i])
                return full
            else:
                # Count unsat
                unsat_cnt = 0
                for ci in range(m_comp):
                    sat = False
                    for v, s in compact_clauses[ci]:
                        if (result[v] and s == 1) or (not result[v] and s == -1):
                            sat = True
                            break
                    if not sat:
                        unsat_cnt += 1
                if unsat_cnt < best_unsat_overall:
                    best_unsat_overall = unsat_cnt
                    best_overall = result.copy()
    
    # Final long run with best seed
    if time.time() < deadline - 10.0:
        rng = random.Random(31337)
        budget = deadline - 2.0
        result = walksat_incremental(compact_clauses, compact_var_to_clauses, n_comp, budget, rng)
        if result is not None:
            all_sat = True
            for ci in range(m_comp):
                sat = False
                for v, s in compact_clauses[ci]:
                    if (result[v] and s == 1) or (not result[v] and s == -1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                full = [False] * n_orig
                for v, val in assignment_fixed.items():
                    full[v] = val
                for i, v in enumerate(var_list):
                    full[v] = bool(result[i])
                return full
            else:
                unsat_cnt = 0
                for ci in range(m_comp):
                    sat = False
                    for v, s in compact_clauses[ci]:
                        if (result[v] and s == 1) or (not result[v] and s == -1):
                            sat = True
                            break
                    if not sat:
                        unsat_cnt += 1
                if unsat_cnt < best_unsat_overall:
                    best_unsat_overall = unsat_cnt
                    best_overall = result.copy()
    
    # Build final answer from best found
    if best_overall is not None:
        full = [False] * n_orig
        for v, val in assignment_fixed.items():
            full[v] = val
        for i, v in enumerate(var_list):
            full[v] = bool(best_overall[i])
        return full
    
    # Fallback
    full = [False] * n_orig
    for v, val in assignment_fixed.items():
        full[v] = val
    rng = random.Random(0)
    for i, v in enumerate(var_list):
        full[v] = rng.random() < 0.5
    return full
