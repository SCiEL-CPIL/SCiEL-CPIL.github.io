import time
import random
import sys
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin
    
    if m == 0:
        return [True] * n
    
    # Convert to 0-based signed literals
    # lit > 0 means variable (lit-1) is True
    # lit < 0 means variable (lit-1) is False
    clause_data = []
    for clause in clauses:
        cd = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cd.append((v, sign))
        clause_data.append(cd)
    
    # Variable occurrence: var_occ[v] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for (v, sign) in cd:
            var_occ[v].append((i, sign))
    
    # ============ CDCL SOLVER (MiniSAT-style) ============
    # Assignment: 0 = unassigned, 1 = True, -1 = False
    assignment = [0] * n
    # Trail: list of (var, level) in assignment order
    trail = []
    trail_lim = []  # Decision levels
    # Level: level[v] = decision level when v was assigned
    level = [0] * n
    # Reason: reason[v] = clause that forced v's assignment (None for decisions)
    reason = [None] * n
    # Watched literals for each clause: watch_list[ci] = list of clause indices watching this literal
    # We use a simpler approach: for each clause, store which variables are watched
    # Actually, let's use a proper watched literals scheme
    
    # For simplicity and correctness, use a basic CDCL without full watched literals
    # but with clause learning and non-chronological backtracking
    
    # Clause database: original clauses + learned clauses
    # Each clause is a list of (var, sign)
    clause_db = [cd[:] for cd in clause_data]
    
    # Activity scores for variable selection (VSIDS)
    activity = [0.0] * n
    # Decay factor
    decay = 0.95
    # Increment for activity
    inc = 1.0
    # Heap-like structure for activities (simplified: just sort periodically)
    
    # Decision level
    decision_level = 0
    
    def assign_var(v, val, lvl, rs):
        """Assign variable v to val at level lvl with reason rs"""
        assignment[v] = val
        level[v] = lvl
        reason[v] = rs
        trail.append((v, lvl))
    
    def unassign_var(v):
        """Unassign variable v"""
        assignment[v] = 0
        level[v] = 0
        reason[v] = None
    
    def propagate_unit():
        """Perform unit propagation. Returns conflict clause or None."""
        while True:
            found_unit = False
            conflict = None
            
            for ci, cd in enumerate(clause_db):
                # Check if clause is satisfied
                sat = False
                unassigned = []
                false_count = 0
                for (v, sign) in cd:
                    if assignment[v] == 0:
                        unassigned.append((v, sign))
                    elif sign == 1:
                        if assignment[v] == 1:
                            sat = True
                            break
                        else:
                            false_count += 1
                    else:
                        if assignment[v] == -1:
                            sat = True
                            break
                        else:
                            false_count += 1
                
                if sat:
                    continue
                
                if len(unassigned) == 0:
                    # Conflict: all literals false
                    conflict = ci
                    break
                
                if len(unassigned) == 1:
                    # Unit clause
                    v, sign = unassigned[0]
                    val = 1 if sign == 1 else -1
                    assign_var(v, val, decision_level, ci)
                    found_unit = True
            
            if conflict is not None:
                return conflict
            if not found_unit:
                return None
    
    def analyze_conflict(conflict_ci):
        """Analyze conflict and return (backtrack_level, learned_clause)."""
        # Simplified analysis: find the conflict clause and learn from it
        # This is a simplified version; full MiniSAT analysis is more complex
        
        # Get the conflict clause
        conflict_clause = clause_db[conflict_ci]
        
        # Find the highest level variable in the conflict clause
        max_level = 0
        for (v, sign) in conflict_clause:
            if assignment[v] != 0 and level[v] > max_level:
                max_level = level[v]
        
        # Backtrack level is one less than the highest level in the conflict
        # But we need to ensure we're not below the second-highest level
        levels_in_clause = set()
        for (v, sign) in conflict_clause:
            if assignment[v] != 0:
                levels_in_clause.add(level[v])
        
        if len(levels_in_clause) >= 2:
            sorted_levels = sorted(levels_in_clause, reverse=True)
            backtrack_level = sorted_levels[1]
        else:
            backtrack_level = 0
        
        # Learn the conflict clause (simplified: just add it)
        # In a full implementation, we would minimize the clause
        learned = [(v, sign) for (v, sign) in conflict_clause]
        
        return backtrack_level, learned
    
    def backtrack_to_level(target_level):
        """Backtrack to the specified decision level."""
        while trail and trail[-1][1] > target_level:
            v, lvl = trail.pop()
            unassign_var(v)
        while len(trail_lim) > target_level:
            trail_lim.pop()
    
    def pick_branch_variable():
        """Pick a variable to branch on using VSIDS-like heuristic."""
        # Find unassigned variables
        unassigned = [v for v in range(n) if assignment[v] == 0]
        if not unassigned:
            return None
        
        # Use activity to guide selection
        # Simplified: pick the variable with highest activity among unassigned
        # For planted SAT, random might be fine, but activity helps
        # Let's use a mix: mostly random with some activity bias
        
        # Get top candidates by activity
        unassigned.sort(key=lambda v: activity[v], reverse=True)
        
        # With some probability, pick from top 10%
        if len(unassigned) > 10 and random.random() < 0.3:
            top_k = max(1, len(unassigned) // 10)
            return unassigned[random.randint(0, top_k - 1)]
        
        return unassigned[random.randint(0, len(unassigned) - 1)]
    
    def restart():
        """Restart the search from root."""
        global decision_level
        # Unassign all
        while trail:
            v, lvl = trail.pop()
            unassign_var(v)
        trail_lim = []
        decision_level = 0
    
    # Main CDCL loop
    max_conflicts = 100000
    conflicts = 0
    restart_interval = 1000
    restart_multiplier = 1.5
    found_solution = False
    
    # Initial propagation
    conflict = propagate_unit()
    if conflict is None:
        # Check if solved
        all_assigned = all(assignment[v] != 0 for v in range(n))
        if all_assigned:
            # Verify solution
            for ci, cd in enumerate(clause_data):
                sat = False
                for (v, sign) in cd:
                    if sign == 1 and assignment[v] == 1:
                        sat = True
                        break
                    elif sign == -1 and assignment[v] == -1:
                        sat = True
                        break
                if not sat:
                    break
            else:
                found_solution = True
    elif conflict is not None:
        # Conflict at root: UNSAT (shouldn't happen for planted SAT)
        pass
    
    if not found_solution and conflict is None:
        while time.time() < deadline:
            # Check if solved
            all_assigned = all(assignment[v] != 0 for v in range(n))
            if all_assigned:
                sat = True
                for ci, cd in enumerate(clause_data):
                    clause_sat = False
                    for (v, sign) in cd:
                        if sign == 1 and assignment[v] == 1:
                            clause_sat = True
                            break
                        elif sign == -1 and assignment[v] == -1:
                            clause_sat = True
                            break
                    if not clause_sat:
                        sat = False
                        break
                if sat:
                    found_solution = True
                    break
            
            # Pick branch variable
            v = pick_branch_variable()
            if v is None:
                break
            
            # Make decision
            decision_level += 1
            trail_lim.append(len(trail))
            
            # Choose polarity: use activity-based heuristic or random
            # For planted SAT, try the value that appears more often in clauses
            pos_count = sum(1 for (ci, s) in var_occ[v] if s == 1)
            neg_count = sum(1 for (ci, s) in var_occ[v] if s == -1)
            if pos_count > neg_count * 1.2:
                val = 1
            elif neg_count > pos_count * 1.2:
                val = -1
            else:
                val = 1 if random.random() < 0.5 else -1
            
            assign_var(v, val, decision_level, None)
            
            # Propagate
            conflict = propagate_unit()
            
            if conflict is not None:
                # Analyze conflict
                conflicts += 1
                
                if conflicts % 1000 == 0:
                    # Decay activities
                    for i in range(n):
                        activity[i] *= decay
                    inc *= 1.0 / decay
                
                # Check if conflict at root
                if decision_level == 0:
                    # UNSAT - shouldn't happen for planted SAT
                    break
                
                # Analyze and learn
                bt_level, learned = analyze_conflict(conflict)
                
                # Add learned clause
                if learned:
                    clause_db.append(learned)
                
                # Boost activity of variables in learned clause
                for (v2, s2) in learned:
                    activity[v2] += inc
                
                # Backtrack
                backtrack_to_level(bt_level)
                
                # Restart if needed
                if conflicts % restart_interval == 0:
                    restart()
                    restart_interval = int(restart_interval * restart_multiplier)
                    conflicts = 0
            
            # Check time
            if time.time() > deadline - 30:
                break
    
    if found_solution:
        return [assignment[v] == 1 for v in range(n)]
    
    # ============ FALLBACK: WalkSAT ============
    # Use the original WalkSAT approach as fallback
    
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    sat_counts = [0] * m
    clause_active = [True] * m
    
    for ci in range(m):
        sc = 0
        for (v, sign) in clause_data[ci]:
            if sign == 1:
                if assignment[v]:
                    sc += 1
            else:
                if not assignment[v]:
                    sc += 1
        sat_counts[ci] = sc
        if sc > 0:
            clause_active[ci] = False
    
    unsat_clauses = [ci for ci in range(m) if clause_active[ci] and sat_counts[ci] == 0]
    
    if not unsat_clauses:
        return assignment
    
    best_unsat = len(unsat_clauses)
    best_assignment = assignment[:]
    
    max_restarts = 500
    max_steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart > 0 and restart % 10 == 0:
            assignment = best_assignment[:]
            for _ in range(min(n // 5, 10)):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
            for ci in range(m):
                sc = 0
                for (v, sign) in clause_data[ci]:
                    if sign == 1:
                        if assignment[v]:
                            sc += 1
                    else:
                        if not assignment[v]:
                            sc += 1
                sat_counts[ci] = sc
            clause_active = [sat_counts[ci] == 0 for ci in range(m)]
            unsat_clauses = [ci for ci in range(m) if clause_active[ci]]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            for ci in range(m):
                sc = 0
                for (v, sign) in clause_data[ci]:
                    if sign == 1:
                        if assignment[v]:
                            sc += 1
                    else:
                        if not assignment[v]:
                            sc += 1
                sat_counts[ci] = sc
            clause_active = [sat_counts[ci] == 0 for ci in range(m)]
            unsat_clauses = [ci for ci in range(m) if clause_active[ci]]
        
        if not unsat_clauses:
            return assignment
        
        if len(unsat_clauses) < best_unsat:
            best_unsat = len(unsat_clauses)
            best_assignment = assignment[:]
        
        p_greedy = 0.5 + 0.2 * (1 - len(unsat_clauses) / m)
        
        stuck_count = 0
        last_unsat_count = len(unsat_clauses)
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
            
            if len(unsat_clauses) >= last_unsat_count:
                stuck_count += 1
            else:
                stuck_count = 0
                last_unsat_count = len(unsat_clauses)
            
            c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            cd_c = clause_data[c_idx]
            
            best_gain = -100
            gains = []
            for (v, sign) in cd_c:
                gain = 0
                for (ci2, sign2) in var_occ[v]:
                    if sign2 == 1:
                        is_true = assignment[v]
                    else:
                        is_true = not assignment[v]
                    
                    if is_true:
                        if sat_counts[ci2] == 1:
                            gain -= 1
                    else:
                        if sat_counts[ci2] == 0:
                            gain += 1
                gains.append(gain)
                if gain > best_gain:
                    best_gain = gain
            
            if rng.random() < p_greedy:
                candidates = [(v, g) for (v, g) in zip([v for (v, s) in cd_c], gains) if g == best_gain]
                v_to_flip = candidates[rng.randint(0, len(candidates) - 1)][0]
            else:
                v_to_flip = cd_c[rng.randint(0, len(cd_c) - 1)][0]
            
            old_val = assignment[v_to_flip]
            assignment[v_to_flip] = not old_val
            
            for (ci2, sign2) in var_occ[v_to_flip]:
                if sign2 == 1:
                    is_true_now = assignment[v_to_flip]
                else:
                    is_true_now = not assignment[v_to_flip]
                
                if is_true_now:
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1 and clause_active[ci2]:
                        clause_active[ci2] = False
                        try:
                            idx = unsat_clauses.index(ci2)
                            unsat_clauses.pop(idx)
                        except ValueError:
                            pass
                else:
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        clause_active[ci2] = True
                        unsat_clauses.append(ci2)
    
    if best_assignment is not None:
        return best_assignment
    
    return [True] * n
