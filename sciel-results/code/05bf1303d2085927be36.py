# EVOLVE-BLOCK-START
import time
import random


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(n, clauses_opt, var_clauses, deadline):
    """DPLL with VSIDS and unit propagation."""
    # clauses_opt: list of [(var_idx, polarity), ...]
    # var_clauses: list of [(clause_idx, polarity), ...]
    
    # Track assignments: None = unassigned, True/False = assigned
    assignment = [None] * n
    
    # For each variable, track which clauses contain it and their current status
    # sat_count[ci] = number of satisfied literals in clause ci
    # unsat_count[ci] = number of unsatisfied literals in clause ci
    # unassigned_count[ci] = number of unassigned literals in clause ci
    
    m = len(clauses_opt)
    sat_count = [0] * m
    unsat_count = [0] * m
    unassigned_count = [0] * m
    
    # Initialize clause counts
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            unassigned_count[ci] += 1
    
    # Unit propagation queue
    def propagate(assignment, sat_count, unsat_count, unassigned_count):
        """Perform unit propagation. Returns False on contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            for ci in range(m):
                if sat_count[ci] > 0:
                    continue  # clause already satisfied
                if unassigned_count[ci] == 0:
                    return False  # all literals unsatisfied -> contradiction
                if unassigned_count[ci] == 1:
                    # Find the unassigned literal
                    for v, p in clauses_opt[ci]:
                        if assignment[v] is None:
                            # Must set this literal to true
                            new_val = p
                            if assignment[v] is not None:
                                if assignment[v] != new_val:
                                    return False  # contradiction
                                continue
                            assignment[v] = new_val
                            # Update all clauses containing v
                            for ci2, p2 in var_clauses[v]:
                                unassigned_count[ci2] -= 1
                                if p2 == new_val:
                                    sat_count[ci2] += 1
                                else:
                                    unsat_count[ci2] += 1
                            changed = True
                            break
        return True
    
    # VSIDS activity scores
    activity = [1.0] * n
    last_touched = [0] * n
    clock = 0
    
    def bump_activity(v):
        nonlocal clock
        activity[v] *= 2.0
        last_touched[v] = clock
        clock += 1
    
    # Select next variable using VSIDS heuristic
    def select_variable(assignment, sat_count, unsat_count, unassigned_count):
        """Select variable with highest activity among unassigned vars in unsat clauses."""
        best_v = -1
        best_score = -1.0
        # Look at variables in currently unsatisfied or partially unassigned clauses
        candidates = set()
        for ci in range(m):
            if sat_count[ci] > 0:
                continue
            for v, p in clauses_opt[ci]:
                if assignment[v] is None:
                    candidates.add(v)
        
        if not candidates:
            return -1
        
        # Pick from candidates with highest activity
        for v in candidates:
            if activity[v] > best_score:
                best_score = activity[v]
                best_v = v
        return best_v
    
    # Recursive DPLL
    def dpll(assignment, sat_count, unsat_count, unassigned_count, depth):
        nonlocal clock
        
        if time.time() > deadline:
            return None
        
        # Check if all clauses satisfied
        all_sat = True
        for ci in range(m):
            if sat_count[ci] == 0 and unassigned_count[ci] == 0:
                all_sat = False
                break
        if all_sat:
            return assignment[:]
        
        # Unit propagation
        if not propagate(assignment, sat_count, unsat_count, unassigned_count):
            return None
        
        # Check again after propagation
        all_sat = True
        for ci in range(m):
            if sat_count[ci] == 0 and unassigned_count[ci] == 0:
                all_sat = False
                break
        if all_sat:
            return assignment[:]
        
        # Select variable
        v = select_variable(assignment, sat_count, unsat_count, unassigned_count)
        if v < 0:
            # No unassigned variables in unsat clauses - check if solved
            for ci in range(m):
                if sat_count[ci] == 0:
                    return None
            return assignment[:]
        
        # Try both values, prefer the one that satisfies more
        # Determine preferred polarity: which value satisfies more currently unsat clauses
        pos_sat = 0
        neg_sat = 0
        for ci, p in var_clauses[v]:
            if sat_count[ci] == 0 and unassigned_count[ci] > 0:
                if p:
                    pos_sat += 1
                else:
                    neg_sat += 1
        
        if pos_sat >= neg_sat:
            order = [True, False]
        else:
            order = [False, True]
        
        for val in order:
            # Save state
            old_sat = sat_count[:]
            old_unsat = unsat_count[:]
            old_unassigned = unassigned_count[:]
            
            # Assign
            assignment[v] = val
            for ci, p in var_clauses[v]:
                unassigned_count[ci] -= 1
                if p == val:
                    sat_count[ci] += 1
                else:
                    unsat_count[ci] += 1
            
            # Bump activity
            bump_activity(v)
            
            # Recurse
            result = dpll(assignment, sat_count, unsat_count, unassigned_count, depth + 1)
            if result is not None:
                return result
            
            # Backtrack
            assignment[v] = None
            sat_count[:] = old_sat
            unsat_count[:] = old_unsat
            unassigned_count[:] = old_unassigned
        
        return None
    
    # Run DPLL
    result = dpll(assignment, sat_count, unsat_count, unassigned_count, 0)
    if result is not None:
        # Convert to boolean list (handle any remaining None)
        return [bool(x) if x is not None else False for x in result]
    return None


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 320  # safety margin before 330s kill

    # Precompute clauses as (var_idx 0-based, polarity bool)
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    # Precompute var -> list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Weights per variable
    weights = [len(l) for l in var_clauses]
    total_w = sum(weights) or 1

    def count_unsat(ass):
        cnt = 0
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                cnt += 1
        return cnt

    def build_state(ass):
        # sat[v] = number of satisfied literals in clause v
        sat = [0] * m
        for ci, cl in enumerate(clauses_opt):
            s = 0
            for v, p in cl:
                if ass[v] == p:
                    s += 1
            sat[ci] = s
        return sat

    def flip_var(ass, var, sat):
        old = ass[var]
        new = not old
        ass[var] = new
        for ci, p in var_clauses[var]:
            if p == new:
                sat[ci] += 1
            else:
                sat[ci] -= 1

    def check_all_sat(sat):
        for s in sat:
            if s == 0:
                return False
        return True

    def unsat_clauses_list(sat):
        return [ci for ci in range(m) if sat[ci] == 0]

    def unit_propagate(ass, remaining_vars, n_remaining, sat):
        # Repeat: find clauses with exactly one unfixed unsat literal.
        # Returns True if no contradiction (ass updated in place).
        # Uses remaining_vars set.
        changed = True
        while changed and n_remaining > 0:
            changed = False
            # For each clause still unsat, count unfixed literals
            for ci in range(m):
                if sat[ci] > 0:
                    continue
                cl = clauses_opt[ci]
                unfixed = []
                for v, p in cl:
                    if v not in remaining_vars:
                        # fixed: if it's satisfied, clause would be sat; but sat==0 so it's unsat
                        continue
                    unfixed.append((v, p))
                if len(unfixed) == 1:
                    v, p = unfixed[0]
                    if ass[v] != p:
                        flip_var(ass, v, sat)
                    remaining_vars.discard(v)
                    n_remaining -= 1
                    changed = True
        return True  # no contradiction handling needed (planted SAT)

    def quick_check(ass):
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                return False
        return True

    # Phase 0: Try DPLL with time limit
    dpll_deadline = start_time + 60  # Give DPLL up to 60 seconds
    dpll_result = _dpll_solve(n, clauses_opt, var_clauses, dpll_deadline)
    if dpll_result is not None:
        # Verify the solution
        if quick_check(dpll_result):
            return dpll_result

    # Phase 1: Stochastic local search if DPLL didn't solve
    best_assignment = None
    best_unsat = float('inf')

    def track(ass):
        nonlocal best_assignment, best_unsat
        u = count_unsat(ass)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        return u

    max_restarts = 200
    rng_master = random.Random(123456789)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 131)
        ass = [rng.random() < 0.5 for _ in range(n)]

        # Try to bias toward positive literals (planted instances often)
        # Occasionally use phase bias: prefer the literal that appears more
        if restart % 5 == 0:
            # phase save bias: set var to polarity with more occurrences
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                ass[v] = (pos >= len(var_clauses[v]) - pos) if len(var_clauses[v]) else False

        u = track(ass)
        if u == 0:
            return ass

        # Unit propagation to reduce
        remaining_vars = set(range(n))
        n_remaining = n
        sat = build_state(ass)
        unit_propagate(ass, remaining_vars, n_remaining, sat)
        u = track(ass)
        if u == 0:
            return ass

        # Determine heuristic mode for this restart
        mode = restart % 3  # 0: min-break, 1: phase-satisfy, 2: max-weight
        noise = 0.2 + 0.1 * (restart % 4)

        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break

            # Find an unsatisfied clause - use longest unsat for stronger guidance
            best_ci = -1
            best_len = 0
            for ci in range(m):
                if sat[ci] == 0:
                    ln = len(clauses_opt[ci])
                    if ln > best_len:
                        best_len = ln
                        best_ci = ci
                        if ln >= 3:
                            # 3 is max for 3-SAT, can break
                            break
            if best_ci < 0:
                return ass

            clause = clauses_opt[best_ci]

            # Choose variable
            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                # Evaluate each literal
                best_v = clause[0][0]
                best_score = None
                for v, p in clause:
                    # new value would be p (to satisfy this literal)
                    if ass[v] == p:
                        # already satisfied? shouldn't be in unsat clause but be safe
                        continue
                    # cost = number of currently-satisfied other clauses broken
                    cost = 0
                    for ci2, p2 in var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    # benefit weight
                    if mode == 0:
                        score = (cost, -weights[v])
                    elif mode == 1:
                        # prefer phase: lower cost, then prefer polarity that matches majority
                        pos = sum(1 for ci2, p2 in var_clauses[v] if p2)
                        phase_match = 1 if (p == (pos >= len(var_clauses[v]) - pos)) else 0
                        score = (cost, -phase_match, -weights[v])
                    else:
                        score = (cost, -weights[v])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_v = v
                v = best_v
            flip_var(ass, v, sat)

            u = 0
            # quick check via sat
            all_sat = True
            for s in sat:
                if s == 0:
                    all_sat = False
                    break
            if all_sat:
                return ass

        track(ass)

    # Phase 2: if time remains, do targeted deep walks from best_assignment
    if best_assignment is not None and time.time() < deadline - 5:
        ass = best_assignment[:]
        sat = build_state(ass)
        u = count_unsat(ass)
        if u > 0:
            rng2 = random.Random(987654321)
            noise = 0.15
            while time.time() < deadline - 2:
                # find unsat clauses
                unsat_list = [ci for ci in range(m) if sat[ci] == 0]
                if not unsat_list:
                    return ass
                # pick a random unsat clause, but bias to shorter ones (easier)
                ci = unsat_list[rng2.randrange(len(unsat_list))]
                clause = clauses_opt[ci]
                if rng2.random() < noise:
                    v, p = clause[rng2.randrange(len(clause))]
                else:
                    best_v = clause[0][0]
                    best_cost = float('inf')
                    for v, p in clause:
                        cost = 0
                        for ci2, p2 in var_clauses[v]:
                            if p2 == ass[v] and sat[ci2] == 1:
                                cost += 1
                        if cost < best_cost:
                            best_cost = cost
                            best_v = v
                    v = best_v
                flip_var(ass, v, sat)
                # quick full check
                all_sat = True
                for s in sat:
                    if s == 0:
                        all_sat = False
                        break
                if all_sat:
                    return ass
                u2 = sum(1 for s in sat if s == 0)
                if u2 < best_unsat:
                    best_unsat = u2
                    best_assignment = ass[:]

    if best_assignment is not None:
        return best_assignment
    return ass
# EVOLVE-BLOCK-END
