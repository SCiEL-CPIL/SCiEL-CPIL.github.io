import random
import time
import math

def solve(instance):
    start_time = time.time()
    deadline = 325.0  # Safety margin

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # --- Preprocessing: Pure Literal + Unit Propagation ---
    work_clauses = [list(c) for c in clauses]
    assigned = {}
    active = [True] * m
    
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
                return [False] * n
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

    def get_initial_assign(strategy):
        assign = [False] * n_red
        if strategy == 'polarity':
            for v in range(n_red):
                assign[v] = pos_counts[v] >= neg_counts[v]
        elif strategy == 'random':
            for v in range(n_red):
                assign[v] = random.random() < 0.5
        else:
            pass
        return assign

    best_unsat_global = float('inf')
    best_assign_global = None
    phase = [False] * n_red

    def compute_initial_state(assign):
        """Compute clause_sat_count and unsat_count, and build unsat_clause_list."""
        clause_sat_count = [0] * m_red
        unsat_count = 0
        unsat_clause_list = []
        for ci in range(m_red):
            cnt = 0
            for v, p in clause_lits[ci]:
                if assign[v] == (p == 1):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat_count += 1
                unsat_clause_list.append(ci)
        return clause_sat_count, unsat_count, unsat_clause_list

    def flip_variable(assign, clause_sat_count, unsat_count, unsat_clause_list, v, ci_hint=None):
        """
        Flip variable v. Update clause_sat_count, unsat_count, unsat_clause_list.
        ci_hint is the clause we're trying to satisfy (for optimization).
        Returns new unsat_count.
        """
        old_val = assign[v]
        assign[v] = not old_val
        new_val = assign[v]

        for (cci, cp) in var_occ[v]:
            old_sat = old_val == (cp == 1)
            new_sat = new_val == (cp == 1)
            if old_sat and not new_sat:
                clause_sat_count[cci] -= 1
                if clause_sat_count[cci] == 0:
                    unsat_count += 1
                    unsat_clause_list.append(cci)
            elif not old_sat and new_sat:
                clause_sat_count[cci] += 1
                if clause_sat_count[cci] == 1:
                    unsat_count -= 1
                    # Remove cci from unsat_clause_list
                    # O(unsat) removal; we'll do it lazily or use a set
                    # For now, just mark it; we'll filter when picking
        return unsat_count

    def pick_unsat_clause(unsat_clause_list, clause_sat_count, m_red):
        """Pick an unsatisfied clause from the list, cleaning up stale entries."""
        while unsat_clause_list:
            ci = unsat_clause_list.pop()
            if clause_sat_count[ci] == 0:
                return ci
        # Should not happen if unsat_count > 0, but fallback
        for ci in range(m_red):
            if clause_sat_count[ci] == 0:
                return ci
        return -1

    def greedy_flip_score(assign, clause_sat_count, v, ci):
        """
        Score for flipping variable v in context of clause ci.
        We want to minimize damage (breaking currently-satisfied clauses)
        and maximize gain (satisfying currently-unsatisfied clauses other than ci).
        """
        old_val = assign[v]
        damage = 0
        gain = 0
        for (cci, cp) in var_occ[v]:
            if cci == ci:
                continue
            if clause_sat_count[cci] == 1:
                if old_val == (cp == 1):
                    damage += 1
            elif clause_sat_count[cci] == 0:
                if old_val != (cp == 1):
                    gain += 1
        # Minimize damage, maximize gain
        # Score = damage - gain (lower is better)
        return damage - gain

    def walk_sat_run(max_restarts, base_steps, noise_init, noise_decay, weight_factor, start_assign=None):
        nonlocal best_unsat_global, best_assign_global, phase
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Initialize
            if start_assign is not None and restart == 0:
                assign = start_assign[:]
            elif restart == 0:
                assign = get_initial_assign('polarity')
            elif restart == 1:
                assign = get_initial_assign('random')
            else:
                assign = phase[:]
                for v in range(n_red):
                    if random.random() < 0.1:
                        assign[v] = not assign[v]

            clause_sat_count, unsat_count, unsat_clause_list = compute_initial_state(assign)
            
            if unsat_count == 0:
                return assign
                
            if unsat_count < best_unsat_global:
                best_unsat_global = unsat_count
                best_assign_global = assign[:]
                phase = assign[:]

            weights = [1.0] * m_red
            
            steps = 0
            max_steps = base_steps + restart * 50
            noise = noise_init * (noise_decay ** restart)
            
            while steps < max_steps and time.time() - start_time < deadline:
                if unsat_count == 0:
                    return assign

                # Pick unsat clause
                ci = pick_unsat_clause(unsat_clause_list, clause_sat_count, m_red)
                if ci < 0:
                    break
                
                clause = clause_lits[ci]
                
                # Choose variable to flip
                if random.random() < noise:
                    v, p = random.choice(clause)
                else:
                    # Greedy: minimize (damage - gain)
                    best_v = -1
                    best_score = float('inf')
                    for v, p in clause:
                        score = greedy_flip_score(assign, clause_sat_count, v, ci)
                        if score < best_score:
                            best_score = score
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
                            unsat_clause_list.append(cci)
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

    # Run main WalkSAT
    result = walk_sat_run(
        max_restarts=500, 
        base_steps=200, 
        noise_init=0.3, 
        noise_decay=0.99, 
        weight_factor=1.05
    )

    # Polish pass: if we got close, do a tighter local search
    if result is not None and best_unsat_global <= 10 and time.time() - start_time < deadline - 30:
        polish_result = walk_sat_run(
            max_restarts=200,
            base_steps=100,
            noise_init=0.15,
            noise_decay=0.995,
            weight_factor=1.02,
            start_assign=result
        )
        if polish_result is not None:
            result = polish_result

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
