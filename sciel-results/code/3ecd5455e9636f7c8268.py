import random
import time
import math
import numpy as np

def solve(instance):
    start_time = time.time()
    deadline = 325.0  # Safety margin

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # --- Preprocessing: Pure Literal + Unit Propagation ---
    # We use a slightly optimized version of the original preprocessing
    work_clauses = [list(c) for c in clauses]
    assigned = {}
    active = [True] * m
    
    # Helper to update active clauses
    def update_active():
        for ci in range(m):
            if not active[ci]:
                continue
            for lit in work_clauses[ci]:
                v = abs(lit) - 1
                if v in assigned:
                    if (lit > 0 and assigned[v]) or (lit < 0 and not assigned[v]):
                        active[ci] = False
                        break

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
            update_active()
            continue
            
        # Unit Propagation
        unit_found = False
        for ci in range(m):
            if not active[ci]: continue
            unassigned_lits = [lit for lit in work_clauses[ci] if (abs(lit) - 1) not in assigned]
            if len(unassigned_lits) == 0:
                return [False] * n # Contradiction, shouldn't happen for SAT
            elif len(unassigned_lits) == 1:
                lit = unassigned_lits[0]
                v = abs(lit) - 1
                assigned[v] = (lit > 0)
                unit_found = True
                changed = True
        
        if unit_found:
            update_active()

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
    clause_lits = []
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

    def check_satisfied(assign):
        unsat = 0
        for ci in range(m_red):
            sat = False
            for v, p in clause_lits[ci]:
                if assign[v] == (p == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    def get_initial_assign(strategy):
        assign = [False] * n_red
        if strategy == 'polarity':
            for v in range(n_red):
                assign[v] = pos_counts[v] >= neg_counts[v]
        elif strategy == 'random':
            for v in range(n_red):
                assign[v] = random.random() < 0.5
        else: # phase
            pass
        return assign

    best_unsat_global = float('inf')
    best_assign_global = None
    phase = [False] * n_red # Global phase memory

    def weighted_walk_sat(max_restarts, base_steps, noise_init, noise_decay, weight_factor, spectral_init=None):
        nonlocal best_unsat_global, best_assign_global, phase
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Initialize
            if restart == 0 and spectral_init is not None:
                assign = spectral_init[:]
            elif restart == 0:
                assign = get_initial_assign('polarity')
            elif restart == 1:
                assign = get_initial_assign('random')
            else:
                # Phase saving: use phase from best previous run
                assign = phase[:]
                # Add some random noise to phase
                for v in range(n_red):
                    if random.random() < 0.1:
                        assign[v] = not assign[v]

            # Compute initial state
            clause_sat_count = [0] * m_red
            unsat_count = 0
            for ci in range(m_red):
                cnt = 0
                for v, p in clause_lits[ci]:
                    if assign[v] == (p == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
                if cnt == 0:
                    unsat_count += 1
            
            if unsat_count == 0:
                return assign
                
            if unsat_count < best_unsat_global:
                best_unsat_global = unsat_count
                best_assign_global = assign[:]
                phase = assign[:]

            # Clause weights
            weights = [1.0] * m_red
            
            steps = 0
            max_steps = base_steps + restart * 50
            noise = noise_init * (noise_decay ** restart)
            
            while steps < max_steps and time.time() - start_time < deadline:
                if unsat_count == 0:
                    return assign

                # Pick unsat clause weighted by 'weights'
                # Simple random pick is faster, but weighted is smarter.
                # For speed, let's stick to random pick of unsat clause, 
                # but we can bias the variable flip.
                
                ci = random.randrange(m_red)
                while clause_sat_count[ci] != 0:
                    ci = random.randrange(m_red)
                
                clause = clause_lits[ci]
                
                # Choose variable to flip
                if random.random() < noise:
                    v, p = random.choice(clause)
                else:
                    # Greedy: minimize damage
                    best_v = -1
                    best_damage = float('inf')
                    for v, p in clause:
                        old_val = assign[v]
                        damage = 0
                        for (cci, cp) in var_occ[v]:
                            if cci == ci:
                                continue
                            if clause_sat_count[cci] == 1:
                                if old_val == (cp == 1):
                                    damage += 1
                        if damage < best_damage:
                            best_damage = damage
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
                            weights[cci] *= weight_factor
                    elif not old_sat and new_sat:
                        clause_sat_count[cci] += 1
                        if clause_sat_count[cci] == 1:
                            unsat_count -= 1
                            weights[cci] /= weight_factor

                steps += 1

                if unsat_count < best_unsat_global:
                    best_unsat_global = unsat_count
                    best_assign_global = assign[:]
                    phase = assign[:]
                    if unsat_count == 0:
                        return assign

        return best_assign_global

    # Spectral phase-recovery: build interaction matrix and use power iteration
    # to estimate the planted global phase vector
    spectral_assign = None
    if n_red <= 200 and time.time() - start_time < deadline - 30:
        try:
            # Build clause interaction matrix: A[v,w] counts how many clauses
            # contain both v and w (with sign encoding for polarity)
            # Use a signed incidence approach: for each clause (l1,l2,l3),
            # create a rank-3 update to a matrix that captures the structure
            # Simpler: build B where B[v,w] = sum over clauses containing both
            # of (sign1 * sign2), and diagonal = degree
            B = np.zeros((n_red, n_red), dtype=np.float64)
            for ci, c in enumerate(red_clauses):
                # Convert to new variable indices and signs
                lits_new = []
                for lit in c:
                    v_old = abs(lit) - 1
                    v_new = var_map[v_old]
                    s = 1.0 if lit > 0 else -1.0
                    lits_new.append((v_new, s))
                for i in range(len(lits_new)):
                    vi, si = lits_new[i]
                    B[vi, vi] += 1.0
                    for j in range(i+1, len(lits_new)):
                        vj, sj = lits_new[j]
                        # For a planted SAT instance, the planted solution
                        # satisfies all clauses. The interaction structure
                        # should have the planted vector as a dominant eigenvector.
                        B[vi, vj] += si * sj
                        B[vj, vi] += si * sj
            
            # Power iteration with non-linear sign correction
            x = np.random.randn(n_red)
            x /= np.linalg.norm(x) + 1e-12
            best_x = x.copy()
            best_obj = -1e18
            
            # Objective: maximize sum over clauses of (product of signed values)
            # which is 1 if clause is satisfied by the sign pattern
            def spectral_obj(vec):
                signs = np.sign(vec)
                signs[signs == 0] = 1
                obj = 0.0
                for ci, c in enumerate(red_clauses):
                    prod = 1.0
                    for lit in c:
                        v_old = abs(lit) - 1
                        v_new = var_map[v_old]
                        s = 1.0 if lit > 0 else -1.0
                        prod *= s * signs[v_new]
                    obj += prod
                return obj
            
            best_obj = spectral_obj(x)
            
            for it in range(200):
                if time.time() - start_time > deadline - 10:
                    break
                # Linear step: y = B @ x
                y = B @ x
                # Non-linear sign correction: project toward sign(y)
                # This is like a fixed-point iteration for the max-cut style objective
                y_sign = np.sign(y)
                y_sign[y_sign == 0] = 1.0
                # Blend: x_new = normalize(y + alpha * y_sign)
                alpha = 0.5
                x_new = y + alpha * y_sign
                norm = np.linalg.norm(x_new)
                if norm < 1e-12:
                    break
                x_new /= norm
                x = x_new
                
                obj = spectral_obj(x)
                if obj > best_obj:
                    best_obj = obj
                    best_x = x.copy()
            
            # Convert to boolean assignment
            signs = np.sign(best_x)
            signs[signs == 0] = 1.0
            spectral_assign = [signs[i] > 0 for i in range(n_red)]
            
            # Verify quality
            unsat_spec = 0
            for ci, c in enumerate(red_clauses):
                sat = False
                for lit in c:
                    v_old = abs(lit) - 1
                    v_new = var_map[v_old]
                    val = spectral_assign[v_new]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True
                        break
                if not sat:
                    unsat_spec += 1
        except Exception:
            spectral_assign = None
    
    # Run WalkSAT with spectral initialization as first restart
    result = weighted_walk_sat(
        max_restarts=500, 
        base_steps=200, 
        noise_init=0.3, 
        noise_decay=0.99, 
        weight_factor=1.05,
        spectral_init=spectral_assign
    )

    if result is not None:
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        return [full_assign.get(v, False) for v in range(n)]

    # Fallback
    full_assign = dict(assigned)
    for v in range(n):
        if v not in full_assign:
            full_assign[v] = random.random() < 0.5
    return [full_assign.get(v, False) for v in range(n)]
