# EVOLVE-BLOCK-START
import time
import random


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(n, clauses_opt, var_clauses, deadline):
    """DPLL with VSIDS and watched-literal unit propagation."""
    m = len(clauses_opt)
    
    # Assignment: 0 = unassigned, 1 = True, -1 = False
    assignment = [0] * n
    # Clause status: for each clause, track count of unassigned literals
    unassigned_count = [len(cl) for cl in clauses_opt]
    
    # Watched literals: for each clause, track two literals that are unassigned
    # watches[ci] = list of (var, polarity) that are currently unassigned
    watches = []
    for ci, cl in enumerate(clauses_opt):
        watches.append([list(cl[0]), list(cl[1])])
    
    # Assignment trail for backtracking
    trail = []  # list of (var, old_value)
    
    # VSIDS
    activity = [1.0] * n
    decay = 0.95
    
    def bump(v):
        activity[v] *= 2.0
    
    def decay_activities():
        for v in range(n):
            activity[v] *= decay
    
    # Unit propagation using watched literals
    def propagate():
        """Returns False on conflict, True otherwise."""
        worklist = []
        # Initialize worklist with all unit clauses
        for ci in range(m):
            if unassigned_count[ci] == 1:
                worklist.append(ci)
        
        while worklist:
            ci = worklist.pop()
            if unassigned_count[ci] == 0:
                return False  # conflict
            if unassigned_count[ci] > 1:
                continue  # no longer unit
            
            # Find the unassigned literal in this clause
            w = watches[ci]
            unassigned_lit = None
            for i in range(2):
                v, p = w[i]
                if assignment[v] == 0:
                    unassigned_lit = (v, p)
                    break
            
            if unassigned_lit is None:
                # Both watched literals are assigned - this shouldn't happen if unassigned_count==1
                # But let's find the actual unassigned one
                for v, p in clauses_opt[ci]:
                    if assignment[v] == 0:
                        unassigned_lit = (v, p)
                        break
            
            if unassigned_lit is None:
                return False  # conflict
            
            v, p = unassigned_lit
            new_val = 1 if p else -1
            
            if assignment[v] != 0:
                if assignment[v] != new_val:
                    return False  # conflict
                continue
            
            # Assign v
            assignment[v] = new_val
            trail.append((v, 0))
            bump(v)
            
            # Update watches for all clauses containing v
            for ci2, p2 in var_clauses[v]:
                if unassigned_count[ci2] == 0 or ci2 == ci:
                    continue
                # Check if v is in watches[ci2]
                w2 = watches[ci2]
                found_idx = -1
                for i in range(2):
                    if w2[i][0] == v:
                        found_idx = i
                        break
                if found_idx == -1:
                    continue
                # v was satisfied or unsatisfied; since we just assigned it,
                # if p2 == new_val, clause is satisfied, no need to update
                if p2 == (new_val == 1):
                    continue  # clause satisfied
                # v is now unsatisfied in this clause, need to find new watch
                unassigned_count[ci2] -= 1
                # Find a new unassigned literal to watch
                new_watch = None
                for vv, pp in clauses_opt[ci2]:
                    if assignment[vv] == 0 and vv != v:
                        new_watch = (vv, pp)
                        break
                if new_watch is not None:
                    w2[found_idx] = list(new_watch)
                else:
                    # No other unassigned literal - clause is now unit
                    if unassigned_count[ci2] == 1:
                        worklist.append(ci2)
                    elif unassigned_count[ci2] == 0:
                        return False
        return True
    
    def select_variable():
        """Select variable with highest activity among unassigned vars in active clauses."""
        best_v = -1
        best_score = -1.0
        for ci in range(m):
            if unassigned_count[ci] == 0:
                continue
            for v, p in clauses_opt[ci]:
                if assignment[v] == 0 and activity[v] > best_score:
                    best_score = activity[v]
                    best_v = v
        return best_v
    
    def all_satisfied():
        for ci in range(m):
            if unassigned_count[ci] == 0:
                return False
        return True
    
    def backtrack_to(level):
        while len(trail) > level:
            v, old_val = trail.pop()
            assignment[v] = old_val
            # Update unassigned_count for clauses containing v
            for ci, p in var_clauses[v]:
                unassigned_count[ci] += 1
            # Rebuild watches if needed (simplified: just mark for rebuild)
    
    def dpll(level):
        if time.time() > deadline:
            return None
        
        if all_satisfied():
            # Fill in remaining unassigned vars
            for v in range(n):
                if assignment[v] == 0:
                    assignment[v] = 1
            return assignment[:]
        
        if not propagate():
            return None
        
        if all_satisfied():
            for v in range(n):
                if assignment[v] == 0:
                    assignment[v] = 1
            return assignment[:]
        
        v = select_variable()
        if v == -1:
            return None
        
        # Determine preferred polarity
        pos_score = 0
        neg_score = 0
        for ci, p in var_clauses[v]:
            if unassigned_count[ci] > 0:
                if p:
                    pos_score += 1
                else:
                    neg_score += 1
        
        if pos_score >= neg_score:
            order = [1, -1]
        else:
            order = [-1, 1]
        
        for val in order:
            trail_level = len(trail)
            assignment[v] = val
            trail.append((v, 0))
            bump(v)
            
            # Update unassigned counts
            for ci, p in var_clauses[v]:
                unassigned_count[ci] -= 1
                if unassigned_count[ci] < 0:
                    unassigned_count[ci] = 0
            
            result = dpll(level + 1)
            if result is not None:
                return result
            
            # Backtrack
            backtrack_to(trail_level)
        
        return None
    
    result = dpll(0)
    if result is not None:
        return [bool(x) for x in result]
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

    # Phase 0: Try DPLL with short time limit
    dpll_deadline = start_time + 5
    dpll_result = _dpll_solve(n, clauses_opt, var_clauses, dpll_deadline)
    if dpll_result is not None:
        if quick_check(dpll_result):
            return dpll_result

    # Phase 1: WalkSAT with Phase Saving and Dynamic Noise
    best_assignment = None
    best_unsat = float('inf')
    phase = [None] * n

    def track(ass):
        nonlocal best_assignment, best_unsat
        u = count_unsat(ass)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
            for v in range(n):
                phase[v] = ass[v]
        return u

    max_restarts = 1000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 131)
        
        ass = []
        for v in range(n):
            if phase[v] is not None and rng.random() < 0.8:
                ass.append(phase[v])
            else:
                ass.append(rng.random() < 0.5)

        u = track(ass)
        if u == 0:
            return ass

        sat = build_state(ass)
        unsat_clauses = [ci for ci in range(m) if sat[ci] == 0]

        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return ass

            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = clauses_opt[ci]

            progress = step / steps
            noise = 0.01 + 0.4 * progress

            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                best_v = clause[0][0]
                best_cost = float('inf')
                best_phase_score = 0
                for v, p in clause:
                    cost = 0
                    for ci2, p2 in var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    
                    new_val = not ass[v]
                    if phase[v] is not None and new_val == phase[v]:
                        phase_score = 1
                    else:
                        phase_score = 0

                    if cost < best_cost or (cost == best_cost and (phase_score > best_phase_score or (phase_score == best_phase_score and weights[v] > weights[best_v]))):
                        best_cost = cost
                        best_phase_score = phase_score
                        best_v = v
                
                v = best_v
            
            old_val = ass[v]
            new_val = not old_val
            ass[v] = new_val
            
            for ci2, p2 in var_clauses[v]:
                old_sat = sat[ci2]
                if p2 == new_val:
                    sat[ci2] = old_sat + 1
                else:
                    sat[ci2] = old_sat - 1
                
                new_sat = sat[ci2]
                if old_sat == 0 and new_sat > 0:
                    try:
                        unsat_clauses.remove(ci2)
                    except ValueError:
                        pass
                elif old_sat > 0 and new_sat == 0:
                    unsat_clauses.append(ci2)
            
            if not unsat_clauses:
                return ass

        track(ass)

    if best_assignment is not None:
        return best_assignment
    return ass
# EVOLVE-BLOCK-END
