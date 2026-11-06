"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock kill, leave 10s margin
    deadline = time.time() + 320.0
    
    # Preprocess: convert clauses to 0-based indices and store signs
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = []
        signs = []
        for lit in c:
            v = abs(lit) - 1
            s = lit > 0
            vars_.append(v)
            signs.append(s)
        clause_vars.append(vars_)
        clause_signs.append(signs)
        
    # Precompute sign of each variable in each clause for fast delta calculation
    # var_sign_in_clause[v][c] = sign of v in clause c (True/False), or None if not present
    # To save memory and time, we store for each clause the list of (var, sign)
    # and build a reverse map: var -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_vars[ci][j]
            s = clause_signs[ci][j]
            var_to_clauses[v].append((ci, s))
            
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True
        
    def dpll_solve(assignment, time_limit):
        """
        DPLL solver with unit propagation and pure literal elimination.
        Returns a satisfying assignment if found within time_limit, else None.
        """
        dpll_assign = assignment[:]
        
        def propagate(assignment, active_clauses):
            """
            Perform unit propagation and pure literal elimination.
            Returns (success, new_assignment, new_active_clauses)
            """
            changed = True
            while changed:
                changed = False
                
                # Unit propagation
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assignment[v] is None:
                            unassigned.append((v, s))
                        elif assignment[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return False, assignment, active_clauses
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        assignment[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                pure_vars = {}
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assignment[v] is None:
                            if v not in pure_vars:
                                pure_vars[v] = set()
                            pure_vars[v].add(s)
                
                for v, signs_set in pure_vars.items():
                    if len(signs_set) == 1:
                        s = signs_set.pop()
                        assignment[v] = s
                        changed = True
                
                if changed:
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if assignment[v] is not None and assignment[v] == s:
                                satisfied = True
                                break
                        if not satisfied:
                            new_active.append(ci)
                    active_clauses = new_active
            
            return True, assignment, active_clauses
        
        success, dpll_assign, active_clauses = propagate(dpll_assign, list(range(m)))
        
        if not success:
            return None
        
        if not active_clauses:
            for v in range(n):
                if dpll_assign[v] is None:
                    dpll_assign[v] = False
            return dpll_assign
        
        # Recursive DPLL with improved branching heuristic
        def dpll_recursive(current_assign, active_clauses, time_limit):
            if time.time() > time_limit:
                return None
            
            # Propagate
            changed = True
            while changed:
                changed = False
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is None:
                            unassigned.append((v, s))
                        elif current_assign[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return None
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        current_assign[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                pure_vars = {}
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is None:
                            if v not in pure_vars:
                                pure_vars[v] = set()
                            pure_vars[v].add(s)
                
                for v, signs_set in pure_vars.items():
                    if len(signs_set) == 1:
                        s = signs_set.pop()
                        current_assign[v] = s
                        changed = True
                
                if changed:
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is not None and current_assign[v] == s:
                                satisfied = True
                                break
                        if not satisfied:
                            new_active.append(ci)
                    active_clauses = new_active
            
            if not active_clauses:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = False
                return current_assign
            
            # Pick branch variable using weighted degree heuristic
            var_score = {}
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                for j in range(3):
                    v = vars_[j]
                    if current_assign[v] is None:
                        var_score[v] = var_score.get(v, 0) + 1
            
            if not var_score:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = False
                return current_assign
            
            branch_var = max(var_score, key=var_score.get)
            
            for val in [True, False]:
                if time.time() > time_limit:
                    return None
                
                saved_assign = current_assign[:]
                current_assign[branch_var] = val
                
                # Propagate
                changed = True
                while changed:
                    changed = False
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        
                        unassigned = []
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is None:
                                unassigned.append((v, s))
                            elif current_assign[v] == s:
                                satisfied = True
                                break
                        
                        if satisfied:
                            continue
                        elif len(unassigned) == 0:
                            current_assign = saved_assign
                            break
                        elif len(unassigned) == 1:
                            v, s = unassigned[0]
                            current_assign[v] = s
                            changed = True
                        else:
                            new_active.append(ci)
                    
                    if current_assign is not saved_assign:
                        active_clauses = new_active
                    
                    # Pure literal elimination
                    pure_vars = {}
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is None:
                                if v not in pure_vars:
                                    pure_vars[v] = set()
                                pure_vars[v].add(s)
                    
                    for v, signs_set in pure_vars.items():
                        if len(signs_set) == 1:
                            s = signs_set.pop()
                            current_assign[v] = s
                            changed = True
                    
                    if changed:
                        new_active = []
                        for ci in active_clauses:
                            vars_ = clause_vars[ci]
                            signs = clause_signs[ci]
                            satisfied = False
                            for j in range(3):
                                v = vars_[j]
                                s = signs[j]
                                if current_assign[v] is not None and current_assign[v] == s:
                                    satisfied = True
                                    break
                            if not satisfied:
                                new_active.append(ci)
                        active_clauses = new_active
                
                if current_assign is not saved_assign:
                    if not active_clauses:
                        for v in range(n):
                            if current_assign[v] is None:
                                current_assign[v] = False
                        return current_assign
                    else:
                        result = dpll_recursive(current_assign, active_clauses, time_limit)
                        if result is not None:
                            return result
                
                current_assign = saved_assign
            
            return None
        
        return dpll_recursive(dpll_assign, active_clauses, time_limit)
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random()
    
    # Phase 1: DPLL with time limit
    dpll_deadline = time.time() + 20.0
    dpll_result = dpll_solve([None] * n, dpll_deadline)
    if dpll_result is not None:
        if is_satisfied(dpll_result):
            return dpll_result
    
    # Phase 2: Efficient WalkSAT with incremental tracking
    max_restarts = 1000
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial sat_count
        sat_count = [0] * m
        for ci in range(m):
            vars_ = clause_vars[ci]
            signs = clause_signs[ci]
            cnt = 0
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    cnt += 1
            sat_count[ci] = cnt
            
        current_unsat = sum(1 for c in sat_count if c == 0)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            
        # Adaptive noise parameter
        noise = 0.1 + 0.1 * (restart / max_restarts)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 1.0:
                break
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                
            # Pick a random unsatisfied clause
            unsat_clauses_list = [ci for ci in range(m) if sat_count[ci] == 0]
            if not unsat_clauses_list:
                break
            clause_idx = rng.choice(unsat_clauses_list)
            vars_, signs = clause_vars[clause_idx], clause_signs[clause_idx]
            
            # Find false literals in this clause
            false_vars = []
            for j in range(3):
                if assignment[vars_[j]] != signs[j]:
                    false_vars.append(vars_[j])
            
            if not false_vars:
                v = rng.choice(vars_)
            else:
                # Use min-unsat heuristic with randomization
                if rng.random() < noise:
                    v = rng.choice(false_vars)
                else:
                    # Evaluate each false variable flip using sat_count
                    best_v = false_vars[0]
                    best_delta = float('inf')
                    for fv in false_vars:
                        delta = 0
                        # Iterate over all clauses containing fv
                        for ci, s in var_to_clauses[fv]:
                            # s is the sign of fv in clause ci
                            # Current value of fv
                            cur_val = assignment[fv]
                            
                            # Is the literal (fv with sign s) currently satisfied?
                            # If cur_val == s, the literal is True.
                            # If cur_val != s, the literal is False.
                            
                            if cur_val == s:
                                # Literal is True. Flipping makes it False.
                                # sat_count[ci] decreases by 1.
                                if sat_count[ci] == 1:
                                    # Clause becomes unsat
                                    delta += 1
                            else:
                                # Literal is False. Flipping makes it True.
                                # sat_count[ci] increases by 1.
                                if sat_count[ci] == 0:
                                    # Clause becomes sat
                                    delta -= 1
                                    
                        if delta < best_delta:
                            best_delta = delta
                            best_v = fv
                    v = best_v
            
            # Flip the variable
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update sat_count incrementally
            for ci, s in var_to_clauses[v]:
                # s is the sign of v in clause ci
                # After flip, is the literal satisfied?
                if assignment[v] == s:
                    # Literal is now True
                    sat_count[ci] += 1
                    # If it was 0, it becomes 1 (unsat -> sat)
                    if sat_count[ci] == 1:
                        current_unsat -= 1
                else:
                    # Literal is now False
                    sat_count[ci] -= 1
                    # If it was 1, it becomes 0 (sat -> unsat)
                    if sat_count[ci] == 0:
                        current_unsat += 1
    
    # Phase 3: Polish with DPLL if close to solution
    if best_assignment is not None and best_unsat <= 10:
        polish_deadline = time.time() + 20.0
        polish_result = dpll_solve(best_assignment, polish_deadline)
        if polish_result is not None:
            if is_satisfied(polish_result):
                return polish_result
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
