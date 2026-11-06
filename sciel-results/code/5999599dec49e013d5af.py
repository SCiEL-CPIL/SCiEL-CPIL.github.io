import random
import time
import math

def solve(instance):
    start_time = time.time()
    deadline = 325.0  # Safety margin for 330s limit

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # --- Preprocessing: Pure Literal + Unit Propagation ---
    # This is crucial to reduce the problem size.
    work_clauses = [list(c) for c in clauses]
    assigned = {}
    active = [True] * m
    
    # Helper to update active clauses and check for unit/pure
    def preprocess():
        changed = True
        while changed:
            changed = False
            # Pure Literal
            pos_vars = set()
            neg_vars = set()
            for ci in range(m):
                if not active[ci]: continue
                for lit in work_clauses[ci]:
                    v = abs(lit) - 1
                    if v in assigned: continue
                    if lit > 0: pos_vars.add(v)
                    else: neg_vars.add(v)
            
            for v in pos_vars - neg_vars:
                assigned[v] = True
                changed = True
            for v in neg_vars - pos_vars:
                assigned[v] = False
                changed = True
            
            if changed:
                # Update active clauses
                for ci in range(m):
                    if not active[ci]: continue
                    for lit in work_clauses[ci]:
                        v = abs(lit) - 1
                        if v in assigned:
                            if (lit > 0 and assigned[v]) or (lit < 0 and not assigned[v]):
                                active[ci] = False
                                break
                continue

            # Unit Propagation
            unit_found = False
            for ci in range(m):
                if not active[ci]: continue
                unassigned_lits = [lit for lit in work_clauses[ci] if (abs(lit) - 1) not in assigned]
                if len(unassigned_lits) == 0:
                    return False # Contradiction
                elif len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    v = abs(lit) - 1
                    assigned[v] = (lit > 0)
                    unit_found = True
                    changed = True
            
            if unit_found:
                # Update active clauses
                for ci in range(m):
                    if not active[ci]: continue
                    for lit in work_clauses[ci]:
                        v = abs(lit) - 1
                        if v in assigned:
                            if (lit > 0 and assigned[v]) or (lit < 0 and not assigned[v]):
                                active[ci] = False
                                break
        return True

    if not preprocess():
        return [False] * n # Should not happen for SAT instances, but safe

    # Build Reduced Problem
    unassigned_vars = [v for v in range(n) if v not in assigned]
    var_map = {v: i for i, v in enumerate(unassigned_vars)}
    n_red = len(unassigned_vars)

    if n_red == 0:
        return [assigned.get(v, False) for v in range(n)]

    red_clauses = []
    for ci in range(m):
        if not active[ci]: continue
        lits = []
        for lit in work_clauses[ci]:
            v = abs(lit) - 1
            if v not in assigned:
                lits.append(lit)
        if len(lits) <= 1:
            if len(lits) == 1:
                assigned[abs(lits[0]) - 1] = (lits[0] > 0)
            continue
        red_clauses.append(lits)
        
    m_red = len(red_clauses)
    if m_red == 0:
        return [assigned.get(v, False) for v in range(n)]

    # Prepare structures for WalkSAT
    # var_occ[v] = list of (clause_idx, polarity)
    var_occ = [[] for _ in range(n_red)]
    clause_lits = [] # List of lists of (var_idx, polarity)
    pos_counts = [0] * n_red
    neg_counts = [0] * n_red

    for ci, c in enumerate(red_clauses):
        lits = []
        for lit in c:
            v_old = abs(lit) - 1
            v_new = var_map[v_old]
            p = 1 if lit > 0 else 0
            lits.append((v_new, p))
            var_occ[v_new].append((ci, p))
            if p == 1: pos_counts[v_new] += 1
            else: neg_counts[v_new] += 1
        clause_lits.append(lits)

    # WalkSAT Implementation
    def walksat(max_restarts, base_steps, noise_init, noise_decay, weight_factor):
        best_unsat = float('inf')
        best_assign = None
        phase = [False] * n_red # Phase memory

        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Initialize assignment
            if restart == 0:
                # Polarity heuristic
                assign = [False] * n_red
                for v in range(n_red):
                    assign[v] = pos_counts[v] >= neg_counts[v]
            else:
                # Use phase memory with noise
                assign = phase[:]
                noise_frac = 0.1
                for v in range(n_red):
                    if random.random() < noise_frac:
                        assign[v] = not assign[v]
            
            # Compute initial state
            clause_sat_count = [0] * m_red
            unsat_clauses = []
            unsat_pos = {}
            for ci in range(m_red):
                cnt = 0
                for v, p in clause_lits[ci]:
                    if assign[v] == (p == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
                if cnt == 0:
                    unsat_pos[ci] = len(unsat_clauses)
                    unsat_clauses.append(ci)
            
            unsat_count = len(unsat_clauses)
            
            if unsat_count == 0:
                return assign
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign[:]
                phase = assign[:]

            # Clause weights for selection
            weights = [1.0] * m_red
            total_weight = 1.0
            
            steps = 0
            max_steps = base_steps + restart * 50
            noise = noise_init * (noise_decay ** restart)
            
            while steps < max_steps and time.time() - start_time < deadline:
                if unsat_count == 0:
                    return assign

                # Pick unsat clause
                if len(unsat_clauses) == 1:
                    ci = unsat_clauses[0]
                else:
                    # Weighted random selection
                    r = random.random() * total_weight
                    cum = 0.0
                    ci = unsat_clauses[0]
                    for uc in unsat_clauses:
                        cum += weights[uc]
                        if r <= cum:
                            ci = uc
                            break
                
                clause = clause_lits[ci]
                
                # Choose variable to flip
                if random.random() < noise:
                    v, p = random.choice(clause)
                else:
                    # Greedy: minimize damage
                    # Damage = number of other unsatisfied clauses that will become satisfied
                    # Actually, we want to maximize the number of satisfied clauses, or minimize the number of newly unsatisfied clauses.
                    # Standard WalkSAT: pick var that satisfies the most other unsatisfied clauses.
                    best_v = -1
                    best_gain = -float('inf')
                    for v, p in clause:
                        old_val = assign[v]
                        gain = 0
                        for (cci, cp) in var_occ[v]:
                            if cci == ci:
                                continue
                            # If cci is unsatisfied (clause_sat_count[cci] == 0)
                            # and flipping v would satisfy it (new_val == (cp == 1))
                            # Note: new_val is not old_val
                            if clause_sat_count[cci] == 0:
                                if (not old_val) == (cp == 1): # Wait, new_val = not old_val
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    v = best_v

                # Flip
                old_val = assign[v]
                assign[v] = not old_val
                new_val = assign[v]

                # Update counts
                for (cci, cp) in var_occ[v]:
                    old_sat = old_val == (cp == 1)
                    new_sat = new_val == (cp == 1)
                    if old_sat and not new_sat:
                        clause_sat_count[cci] -= 1
                        if clause_sat_count[cci] == 0:
                            unsat_count += 1
                            unsat_pos[cci] = len(unsat_clauses)
                            unsat_clauses.append(cci)
                            weights[cci] *= weight_factor
                            total_weight += weights[cci] - 1.0
                    elif not old_sat and new_sat:
                        clause_sat_count[cci] += 1
                        if clause_sat_count[cci] == 1:
                            unsat_count -= 1
                            # Remove from unsat_clauses
                            pos = unsat_pos[cci]
                            last = unsat_clauses[-1]
                            unsat_clauses[pos] = last
                            unsat_pos[last] = pos
                            unsat_clauses.pop()
                            del unsat_pos[cci]
                            old_w = weights[cci]
                            weights[cci] /= weight_factor
                            total_weight += weights[cci] - old_w

                steps += 1

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assign[:]
                    phase = assign[:]
                    if unsat_count == 0:
                        return assign

        return best_assign

    # Run WalkSAT
    result = walksat(
        max_restarts=1000, 
        base_steps=100, 
        noise_init=0.4, 
        noise_decay=0.99, 
        weight_factor=1.1
    )
    
    if result is not None:
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        final_assign = [full_assign.get(v, False) for v in range(n)]
        
        # Verify
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and final_assign[v]) or (lit < 0 and not final_assign[v]):
                    sat = True
                    break
            if not sat:
                break
        else:
            return final_assign

    # Fallback: Return best effort or random
    if result is not None:
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        return [full_assign.get(v, False) for v in range(n)]
    
    return [random.random() < 0.5 for _ in range(n)]
