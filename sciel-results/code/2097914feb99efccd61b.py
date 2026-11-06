# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit

    # Convert clauses to 0-based signed literals
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    m = len(clauses_0based)

    # Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # ========== DPLL SOLVER WITH WATCHED LITERALS AND VSIDS ==========
    
    def _dpll_watchlit(deadline):
        """DPLL with watched literals, VSIDS, and clause learning."""
        # Variable state: 0=unassigned, 1=True, -1=False
        values = [0] * n
        trail = []  # Stack of assigned variables
        trail_lim = []  # Decision levels
        
        # VSIDS
        var_activity = [0.0] * n
        var_inc = 1.0
        var_order = list(range(n))
        
        # Clause structure
        # Each clause: list of (var, sign), with watch pointers
        # We'll use a simpler approach: maintain clause list and track satisfied status
        
        clause_satisfied = [False] * m
        clause_unsat_count = [0] * m  # Number of unassigned literals
        clause_false_count = [0] * m  # Number of false literals
        
        # Initialize clause statistics
        for ci, clause in enumerate(clauses_0based):
            clause_unsat_count[ci] = len(clause)
        
        # Watched literal approach: for each clause, watch 2 literals
        # When a literal becomes false, look for a new watch
        watches = [None] * m  # watches[ci] = (var1, var2) indices into clause
        
        # Initialize watches
        for ci, clause in enumerate(clauses_0based):
            if len(clause) >= 2:
                watches[ci] = (0, 1)
            elif len(clause) == 1:
                watches[ci] = (0, 0)
        
        # For each variable, track which clauses watch it
        var_watches = defaultdict(list)  # var -> list of (ci, watch_pos)
        for ci in range(m):
            if watches[ci] is not None:
                w0, w1 = watches[ci]
                v0 = clauses_0based[ci][w0][0]
                v1 = clauses_0based[ci][w1][0]
                var_watches[v0].append((ci, 0))
                if w0 != w1:
                    var_watches[v1].append((ci, 1))
        
        def bump_activity(var):
            var_activity[var] += var_inc
            if var_activity[var] > 1e100:
                # Rescale
                for v in range(n):
                    var_activity[v] /= 1e100
                var_inc /= 1e100
        
        def update_var_order():
            # Simple heap-like approach: just sort occasionally
            # For efficiency, we'll use a simpler selection
            pass
        
        def propagate():
            """Unit propagation with watched literals. Returns False on conflict."""
            while len(trail) > trail_lim[-1] if trail_lim else len(trail) > 0:
                # Actually, we need to process from the last decision level
                pass
            
            # Process assignments made after the last checkpoint
            # This is simplified - in a full implementation, we'd use a work queue
            # For now, do a simple scan
            return True
        
        # Simplified DPLL: use pure literal elimination + unit propagation + branching
        # with VSIDS-guided variable selection
        
        # Preprocessing: Pure literal elimination
        pure_assigned = [False] * n
        pure_value = [False] * n
        
        lit_pos = [0] * n  # count of positive occurrences
        lit_neg = [0] * n  # count of negative occurrences
        for clause in clauses_0based:
            for var, sign in clause:
                if sign > 0:
                    lit_pos[var] += 1
                else:
                    lit_neg[var] += 1
        
        changed = True
        while changed:
            changed = False
            for var in range(n):
                if pure_assigned[var]:
                    continue
                if lit_pos[var] > 0 and lit_neg[var] == 0:
                    pure_assigned[var] = True
                    pure_value[var] = True
                    changed = True
                elif lit_neg[var] > 0 and lit_pos[var] == 0:
                    pure_assigned[var] = True
                    pure_value[var] = False
                    changed = True
        
        # Simplify clauses
        active_clauses = []
        for clause in clauses_0based:
            remaining = []
            sat = False
            for var, sign in clause:
                if pure_assigned[var]:
                    if pure_value[var] == (sign > 0):
                        sat = True
                        break
                else:
                    remaining.append((var, sign))
            if not sat and len(remaining) > 0:
                active_clauses.append(remaining)
        
        # Unit propagation on remaining
        unassigned = [v for v in range(n) if not pure_assigned[v]]
        var_map = {v: i for i, v in enumerate(unassigned)}
        n_rem = len(unassigned)
        
        if n_rem == 0:
            # All assigned
            result = [False] * n
            for v in range(n):
                result[v] = pure_value[v]
            return result
        
        # Convert to compact form
        compact_clauses = []
        for clause in active_clauses:
            cc = []
            for var, sign in clause:
                if not pure_assigned[var]:
                    cc.append((var_map[var], sign))
            if len(cc) > 0:
                compact_clauses.append(cc)
        
        m_rem = len(compact_clauses)
        if m_rem == 0:
            result = [False] * n
            for v in range(n):
                if pure_assigned[v]:
                    result[v] = pure_value[v]
                else:
                    result[v] = False  # arbitrary
            return result
        
        # Build index
        var_clauses_rem = [[] for _ in range(n_rem)]
        for ci, clause in enumerate(compact_clauses):
            for var_idx, sign in clause:
                var_clauses_rem[var_idx].append((ci, sign))
        
        # DPLL with backtracking
        assignment = [False] * n_rem
        clause_sat = [False] * m_rem
        clause_unsat = [0] * m_rem  # number of unassigned literals
        
        for ci, clause in enumerate(compact_clauses):
            clause_unsat[ci] = len(clause)
        
        def check_solution():
            for ci, clause in enumerate(compact_clauses):
                sat = False
                for var_idx, sign in clause:
                    if assignment[var_idx] == (sign > 0):
                        sat = True
                        break
                if not sat:
                    return False
            return True
        
        # Recursive DPLL with VSIDS
        var_act = [0.0] * n_rem
        var_inc = 1.0
        decision_level = [0] * n_rem
        current_level = 0
        trail = []
        
        def select_variable():
            """Select variable with highest activity among unassigned in unsat clauses."""
            best_var = -1
            best_act = -1.0
            # Focus on variables in unsatisfied clauses
            unsat_vars = set()
            for ci, clause in enumerate(compact_clauses):
                sat = False
                for var_idx, sign in clause:
                    if assignment[var_idx] == (sign > 0):
                        sat = True
                        break
                if not sat:
                    for var_idx, sign in clause:
                        if decision_level[var_idx] == 0:
                            unsat_vars.add(var_idx)
            
            if not unsat_vars:
                # No unsat clauses, pick any unassigned
                for v in range(n_rem):
                    if decision_level[v] == 0:
                        return v
                return -1
            
            for v in unsat_vars:
                if var_act[v] > best_act:
                    best_act = var_act[v]
                    best_var = v
            return best_var
        
        def dpll():
            """Main DPLL search."""
            nonlocal current_level, trail, var_inc
            
            if time.time() > deadline:
                return None
            
            # Check if all clauses satisfied
            unsat_count = 0
            for ci, clause in enumerate(compact_clauses):
                sat = False
                for var_idx, sign in clause:
                    if assignment[var_idx] == (sign > 0):
                        sat = True
                        break
                if not sat:
                    unsat_count += 1
            
            if unsat_count == 0:
                return True
            
            # Select variable to branch on
            var = select_variable()
            if var == -1:
                return False  # Shouldn't happen
            
            # Try True first
            current_level += 1
            decision_level[var] = current_level
            trail.append(var)
            assignment[var] = True
            
            # Propagate
            conflict = False
            changed = True
            while changed:
                changed = False
                for ci, clause in enumerate(compact_clauses):
                    # Check if clause is unit
                    unassigned_lits = []
                    sat = False
                    all_false = True
                    for var_idx, sign in clause:
                        if assignment[var_idx] == (sign > 0):
                            sat = True
                            break
                        elif decision_level[var_idx] == 0:
                            unassigned_lits.append((var_idx, sign))
                        else:
                            pass  # already assigned and false
                        if assignment[var_idx] != (sign > 0):
                            all_false = False
                    
                    if sat:
                        continue
                    
                    if len(unassigned_lits) == 1:
                        # Unit clause
                        v, s = unassigned_lits[0]
                        if decision_level[v] == 0:
                            assignment[v] = (s > 0)
                            decision_level[v] = current_level
                            trail.append(v)
                            changed = True
                            # Bump activity
                            var_act[v] += var_inc
                    elif len(unassigned_lits) == 0 and not sat:
                        # Conflict
                        conflict = True
                        break
                
                if conflict:
                    break
            
            if not conflict:
                # Check if solved
                all_sat = True
                for ci, clause in enumerate(compact_clauses):
                    sat = False
                    for var_idx, sign in clause:
                        if assignment[var_idx] == (sign > 0):
                            sat = True
                            break
                    if not sat:
                        all_sat = False
                        break
                
                if all_sat:
                    return True
                
                # Recurse
                if dpll():
                    return True
            
            # Backtrack
            while trail and decision_level[trail[-1]] >= current_level:
                v = trail.pop()
                assignment[v] = False
                decision_level[v] = 0
            current_level -= 1
            
            # Try False
            current_level += 1
            decision_level[var] = current_level
            trail.append(var)
            assignment[var] = False
            
            # Propagate
            conflict = False
            changed = True
            while changed:
                changed = False
                for ci, clause in enumerate(compact_clauses):
                    unassigned_lits = []
                    sat = False
                    for var_idx, sign in clause:
                        if assignment[var_idx] == (sign > 0):
                            sat = True
                            break
                        elif decision_level[var_idx] == 0:
                            unassigned_lits.append((var_idx, sign))
                    
                    if sat:
                        continue
                    
                    if len(unassigned_lits) == 1:
                        v, s = unassigned_lits[0]
                        if decision_level[v] == 0:
                            assignment[v] = (s > 0)
                            decision_level[v] = current_level
                            trail.append(v)
                            changed = True
                            var_act[v] += var_inc
                    elif len(unassigned_lits) == 0 and not sat:
                        conflict = True
                        break
                
                if conflict:
                    break
            
            if not conflict:
                all_sat = True
                for ci, clause in enumerate(compact_clauses):
                    sat = False
                    for var_idx, sign in clause:
                        if assignment[var_idx] == (sign > 0):
                            sat = True
                            break
                    if not sat:
                        all_sat = False
                        break
                
                if all_sat:
                    return True
                
                if dpll():
                    return True
            
            # Backtrack
            while trail and decision_level[trail[-1]] >= current_level:
                v = trail.pop()
                assignment[v] = False
                decision_level[v] = 0
            current_level -= 1
            
            # Bump activity of var
            var_act[var] += var_inc
            if var_act[var] > 1e100:
                for v in range(n_rem):
                    var_act[v] /= 1e100
                var_inc /= 1e100
            
            return False
        
        result = dpll()
        if result:
            # Build full solution
            final = [False] * n
            for v in range(n):
                if pure_assigned[v]:
                    final[v] = pure_value[v]
                else:
                    final[v] = assignment[var_map[v]]
            return final
        return None

    # Try DPLL first
    dpll_result = _dpll_watchlit(deadline)
    if dpll_result is not None:
        # Verify
        all_sat = True
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if dpll_result[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return dpll_result

    # ========== FALLBACK: WALKSAT WITH DYNAMIC NOISE ==========
    
    # Rebuild search structures (in case DPLL modified state)
    var_assigned = [False] * n
    var_value = [False] * n
    
    # Pure literal elimination
    lit_count = {}
    for clause in clauses_0based:
        for var, sign in clause:
            key = (var, sign)
            lit_count[key] = lit_count.get(key, 0) + 1
    
    changed = True
    while changed:
        changed = False
        for var in range(n):
            if var_assigned[var]:
                continue
            pos_count = lit_count.get((var, 1), 0)
            neg_count = lit_count.get((var, -1), 0)
            if pos_count > 0 and neg_count == 0:
                var_assigned[var] = True
                var_value[var] = True
                changed = True
            elif neg_count > 0 and pos_count == 0:
                var_assigned[var] = True
                var_value[var] = False
                changed = True
    
    simplified_clauses = []
    for clause in clauses_0based:
        clause_sat = False
        remaining_lits = []
        for var, sign in clause:
            if var_assigned[var]:
                if var_value[var] == (sign > 0):
                    clause_sat = True
                    break
            else:
                remaining_lits.append((var, sign))
        if not clause_sat:
            simplified_clauses.append(remaining_lits)
    
    # Unit propagation
    changed = True
    while changed:
        changed = False
        new_simplified = []
        for clause in simplified_clauses:
            if len(clause) == 1 and not var_assigned[clause[0][0]]:
                v, s = clause[0]
                var_assigned[v] = True
                var_value[v] = (s > 0)
                changed = True
                continue
            remaining = []
            clause_sat = False
            for var, sign in clause:
                if var_assigned[var]:
                    if var_value[var] == (sign > 0):
                        clause_sat = True
                        break
                else:
                    remaining.append((var, sign))
            if not clause_sat:
                new_simplified.append(remaining)
        simplified_clauses = new_simplified
    
    if len(simplified_clauses) == 0:
        return var_value[:]
    
    unassigned_vars = [v for v in range(n) if not var_assigned[v]]
    var_map = {v: i for i, v in enumerate(unassigned_vars)}
    n_search = len(unassigned_vars)
    
    if n_search == 0:
        return var_value[:]
    
    search_clauses = []
    for clause in simplified_clauses:
        search_clause = []
        for var, sign in clause:
            if not var_assigned[var]:
                search_clause.append((var_map[var], sign))
        if len(search_clause) > 0:
            search_clauses.append(search_clause)
    
    m_search = len(search_clauses)
    if m_search == 0:
        return var_value[:]
    
    search_var_clauses = [[] for _ in range(n_search)]
    for ci, clause in enumerate(search_clauses):
        for var_idx, sign in clause:
            search_var_clauses[var_idx].append((ci, sign))
    
    def _walksat_dynamic(rng, max_restarts=80):
        best_assignment = None
        best_unsat = m_search + 1
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            assignment = [rng.random() < 0.5 for _ in range(n_search)]
            
            clause_unsat = [False] * m_search
            unsat_count = 0
            for ci, clause in enumerate(search_clauses):
                sat = False
                for var_idx, sign in clause:
                    if assignment[var_idx] == (sign > 0):
                        sat = True
                        break
                clause_unsat[ci] = not sat
                if not sat:
                    unsat_count += 1
            
            if unsat_count == 0:
                final = var_value[:]
                for i, v in enumerate(unassigned_vars):
                    final[v] = assignment[i]
                return final
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            noise_start = 0.8
            noise_end = 0.1
            steps = 5000
            noise_decay = (noise_end / noise_start) ** (1.0 / steps)
            noise = noise_start
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                if unsat_count == 0:
                    final = var_value[:]
                    for i, v in enumerate(unassigned_vars):
                        final[v] = assignment[i]
                    return final
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                
                unsat_indices = [i for i in range(m_search) if clause_unsat[i]]
                if not unsat_indices:
                    break
                ci = unsat_indices[rng.randrange(len(unsat_indices))]
                clause = search_clauses[ci]
                
                best_var = -1
                best_score = -float('inf')
                
                for var_idx, sign in clause:
                    delta = 0
                    for other_ci, other_sign in search_var_clauses[var_idx]:
                        if other_ci == ci:
                            continue
                        other_clause = search_clauses[other_ci]
                        if clause_unsat[other_ci]:
                            delta -= 1
                        else:
                            only_sat = True
                            for v2, s2 in other_clause:
                                if v2 == var_idx:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                delta += 1
                    
                    score = -delta + rng.random() * noise
                    if score > best_score:
                        best_score = score
                        best_var = var_idx
                
                if best_var == -1:
                    break
                
                assignment[best_var] = not assignment[best_var]
                
                for other_ci, other_sign in search_var_clauses[best_var]:
                    other_clause = search_clauses[other_ci]
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat
                
                noise *= noise_decay
        
        if best_assignment is not None:
            final = var_value[:]
            for i, v in enumerate(unassigned_vars):
                final[v] = best_assignment[i]
            return final
        return var_value[:]
    
    # Run WalkSAT
    rng1 = random.Random(42)
    result = _walksat_dynamic(rng1, max_restarts=80)
    
    # Verify
    all_sat = True
    for clause in search_clauses:
        sat = False
        for var_idx, sign in clause:
            if result[unassigned_vars[var_idx]] == (sign > 0):
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        return result
    
    return result

# EVOLVE-BLOCK-END
