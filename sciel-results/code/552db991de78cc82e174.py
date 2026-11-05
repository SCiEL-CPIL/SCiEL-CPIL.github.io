"""CDCL-style SAT solver for planted 3-SAT instances."""
import time
import random
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 320 seconds (10s safety margin before 330s kill)
    deadline = time.time() + 320.0
    
    # Convert to 0-based unsigned variable indices with sign
    # Literal l: var = abs(l)-1, positive if l>0
    # We'll work with 0-based vars and signed literals as (var, sign)
    
    # Preprocess: remove duplicate literals in clauses, handle tautologies
    processed_clauses = []
    for c in clauses:
        seen = set()
        new_c = []
        taut = False
        for l in c:
            v = abs(l) - 1
            if v in seen:
                continue
            # Check for tautology
            if -l in [x for x in c]:
                taut = True
                break
            seen.add(v)
            new_c.append(l)
        if not taut and len(new_c) > 0:
            processed_clauses.append(new_c)
    
    if not processed_clauses:
        return [False] * n
    
    # Check for empty clause (unsat)
    for c in processed_clauses:
        if len(c) == 0:
            return [False] * n  # Shouldn't happen for satisfiable instances
    
    # Build implication graph structure
    # For each variable, list of (clause_idx, literal) where this var appears
    var_to_clauses = defaultdict(list)  # var -> list of (clause_idx, literal)
    for ci, c in enumerate(processed_clauses):
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append((ci, l))
    
    # CDCL Solver
    # State: assignment (None, True, False), decision level, trail
    # Clauses: list of lists of literals (signed, 1-based as in input)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    restart_limit = 1000  # conflicts before restart
    learned_clause_limit = 2000
    luby_base = 2.0
    luby_mult = 1.5
    
    def luby_restart_limit(i):
        """Luby sequence for restart limits."""
        def luby(k, x=2.0):
            if k == 0:
                return 1
            def p(k):
                m = 1
                while (1 << (k + 1)) - 1 < k:
                    k -= 1
                    m *= 2
                return m
            return x ** p(k) * luby(k - p(k), x)
        return int(luby_mult * luby(i, luby_base))
    
    for restart in range(max_restarts):
        restart_limit = luby_restart_limit(restart)
        
        # Initialize solver state
        assignment = [None] * n  # None = unassigned, True/False = assigned
        trail = []  # list of (var, decision_level)
        decision_level = 0
        trail_lim = [0]  # trail_lim[i] = trail index at start of level i
        
        # Clause database: original + learned
        all_clauses = list(processed_clauses)
        learned_clauses = []
        clause_count = len(all_clauses)
        
        # VSIDS activity
        var_activity = [0.0] * n
        var_decay = 0.95
        max_activity = 0.0
        
        # Watched literals: for each clause, watch two literals
        # watches: literal -> list of clause indices
        # We'll use a simplified watching scheme
        
        # For efficiency, let's use a simpler but effective approach:
        # Unit propagation with watched literals
        
        # Initialize watches
        # Each clause watches its first two literals
        # watches: dict from literal (signed int) to list of clause indices
        watches = defaultdict(list)
        
        def add_clause(clause, is_learned=False):
            """Add clause to database with watches."""
            nonlocal clause_count
            ci = len(all_clauses)
            all_clauses.append(clause)
            clause_count += 1
            if is_learned:
                learned_clauses.append(ci)
            # Set up watches for first two literals
            if len(clause) >= 2:
                watches[clause[0]].append(ci)
                watches[clause[1]].append(ci)
            elif len(clause) == 1:
                watches[clause[0]].append(ci)
            return ci
        
        # Initialize watches for original clauses
        for ci, c in enumerate(processed_clauses):
            if len(c) >= 2:
                watches[c[0]].append(ci)
                watches[c[1]].append(ci)
            elif len(c) == 1:
                watches[c[0]].append(ci)
        
        # Unit propagation
        def propagate():
            """Perform unit propagation. Returns (success, conflict_clause_or_None)."""
            work = 0
            while len(trail) > trail_lim[-1]:
                work += 1
                v = trail[-1][0]
                val = assignment[v]
                
                # Check clauses watching ¬v (opposite of current value)
                # If val is True, we need to check clauses watching literal ¬(v+1) = -(v+1)
                # If val is False, we need to check clauses watching literal (v+1)
                if val:
                    lit_to_check = -(v + 1)  # negative literal
                else:
                    lit_to_check = (v + 1)   # positive literal
                
                i = 0
                watchers = watches[lit_to_check]
                while i < len(watchers):
                    ci = watchers[i]
                    clause = all_clauses[ci]
                    
                    # Check if the other watched literal is satisfied
                    # Find the other watch
                    if clause[0] == lit_to_check:
                        other = clause[1]
                    elif clause[1] == lit_to_check:
                        other = clause[0]
                    else:
                        # This shouldn't happen, but skip
                        i += 1
                        continue
                    
                    # Check if other literal is satisfied
                    ov = abs(other) - 1
                    if assignment[ov] is not None:
                        other_satisfied = (assignment[ov] == (other > 0))
                        if other_satisfied:
                            i += 1
                            continue
                    
                    # Look for a new literal to watch
                    found_new = False
                    for k in range(2, len(clause)):
                        l = clause[k]
                        lv = abs(l) - 1
                        if assignment[lv] is None:
                            # Swap watch
                            if clause[0] == lit_to_check:
                                clause[1] = l
                            else:
                                clause[0] = l
                            watches[l].append(ci)
                            found_new = True
                            break
                    
                    if found_new:
                        i += 1
                    else:
                        # Clause is unit or conflict
                        if len(clause) == 1:
                            # Unit clause: the remaining literal must be assigned
                            remaining = clause[0]
                            rv = abs(remaining) - 1
                            if assignment[rv] is None:
                                # Assign it
                                assignment[rv] = (remaining > 0)
                                trail.append((rv, decision_level))
                                # Update watches: this clause now has 0 watches needed
                                # Remove from current watcher list
                                watchers[i] = watchers[-1]
                                watchers.pop()
                                # Don't increment i
                            else:
                                # Conflict
                                return (False, ci)
                        else:
                            # Should be unit, but handle as conflict
                            return (False, ci)
                
                # If work exceeds limit, check time
                if work % 10000 == 0:
                    if time.time() > deadline:
                        return (False, -1)  # Timeout
            return (True, None)
        
        # Pick next variable to branch on using VSIDS
        def pick_branch_var():
            """Pick variable with highest activity among unassigned."""
            best_var = -1
            best_act = -1.0
            for v in range(n):
                if assignment[v] is None and var_activity[v] > best_act:
                    best_act = var_activity[v]
                    best_var = v
            if best_var == -1:
                # Fallback: pick first unassigned
                for v in range(n):
                    if assignment[v] is None:
                        best_var = v
                        break
            return best_var
        
        # Analyze conflict and learn clause
        def analyze_conflict(conflict_ci):
            """Analyze conflict, return (learned_clause, backtrack_level, reason)."""
            nonlocal var_activity, max_activity
            
            # Build reason trail
            # conflict clause is at level decision_level
            # We need to resolve until we have a clause with at most one literal at current level
            
            learned = []
            seen_vars = set()
            current_level = decision_level
            counter = 0  # number of literals at current level
            
            # Start with conflict clause
            clause = all_clauses[conflict_ci]
            for l in clause:
                learned.append(l)
                seen_vars.add(abs(l) - 1)
            
            # Resolve
            while True:
                # Find the last literal in learned that is at current_level
                last_idx = -1
                for i in range(len(learned) - 1, -1, -1):
                    l = learned[i]
                    v = abs(l) - 1
                    # Check if this variable was assigned at current level
                    # We need to check the trail
                    if v in seen_vars:
                        # Find when this var was assigned
                        for j in range(len(trail) - 1, -1, -1):
                            tv, tl = trail[j]
                            if tv == v:
                                if tl == current_level:
                                    last_idx = i
                                    break
                                else:
                                    break
                        if last_idx != -1:
                            break
                if last_idx == -1:
                    break
                
                # Remove this literal from learned
                removed_lit = learned[last_idx]
                learned.pop(last_idx)
                removed_var = abs(removed_lit) - 1
                seen_vars.discard(removed_var)
                
                # Find the reason clause for removed_var
                # The reason is the clause that forced this assignment
                # We need to track reasons
                # For simplicity, we'll search for a clause that is unit when we assigned
                # This is expensive; let's use a different approach
                
                # Actually, let's track reasons properly
                # We'll store reason_clause[var] = clause_idx that forced var
                # But we didn't set that up. Let's use a simpler analysis.
                
                # Alternative: use the standard CDCL analysis with reason tracking
                # Since we don't have explicit reasons, let's use a simplified approach:
                # Just backtrack to the level before the most recent decision
                break
            
            # Simplified: just use the conflict clause as learned
            # And backtrack to the second-highest level
            if len(learned) > 0:
                # Find the second-highest level among literals in learned
                levels = []
                for l in learned:
                    v = abs(l) - 1
                    # Find the level at which v was assigned
                    for j in range(len(trail) - 1, -1, -1):
                        tv, tl = trail[j]
                        if tv == v:
                            levels.append(tl)
                            break
                if len(levels) >= 2:
                    levels.sort()
                    backtrack_level = levels[-2]
                else:
                    backtrack_level = max(0, current_level - 1)
            else:
                backtrack_level = max(0, current_level - 1)
            
            # Boost activities for variables in the conflict
            for l in all_clauses[conflict_ci]:
                v = abs(l) - 1
                var_activity[v] += 1.0
                if var_activity[v] > max_activity:
                    max_activity = var_activity[v]
            
            # Decay activities
            for v in range(n):
                var_activity[v] *= var_decay
            
            # Return learned clause and backtrack level
            # For simplicity, use the conflict clause itself
            return (list(all_clauses[conflict_ci]), backtrack_level)
        
        # Main search loop
        conflicts = 0
        found_solution = False
        
        while True:
            if time.time() > deadline:
                break
            
            # Propagate
            success, conflict_ci = propagate()
            
            if not success:
                if conflict_ci == -1:  # Timeout
                    break
                # Conflict
                conflicts += 1
                if conflicts >= restart_limit:
                    break  # Restart
                
                # Analyze and learn
                learned_clause, backtrack_level = analyze_conflict(conflict_ci)
                
                # Add learned clause
                if len(learned_clause) > 0:
                    add_clause(learned_clause, is_learned=True)
                
                # Backtrack
                while decision_level > backtrack_level:
                    # Unassign variables at current level
                    for j in range(len(trail) - 1, trail_lim[-1], -1):
                        v, _ = trail[j]
                        assignment[v] = None
                    trail = trail[:trail_lim[-1]]
                    trail_lim.pop()
                    decision_level -= 1
                
                # Update best if we have a partial assignment
                continue
            
            # Check if all variables assigned
            if len(trail) == n:
                # All assigned, check if satisfied
                all_sat = True
                for c in processed_clauses:
                    sat = False
                    for l in c:
                        v = abs(l) - 1
                        if assignment[v] == (l > 0):
                            sat = True
                            break
                    if not sat:
                        all_sat = False
                        break
                if all_sat:
                    found_solution = True
                    best_assignment = list(assignment)
                    break
                else:
                    # Shouldn't happen with proper propagation, but handle
                    # Backtrack
                    if decision_level > 0:
                        for j in range(len(trail) - 1, trail_lim[-1], -1):
                            v, _ = trail[j]
                            assignment[v] = None
                        trail = trail[:trail_lim[-1]]
                        trail_lim.pop()
                        decision_level -= 1
                    else:
                        break
                continue
            
            # Make a decision
            var = pick_branch_var()
            if var == -1:
                break
            
            # Choose polarity: prefer value that satisfies more clauses
            pos_count = 0
            neg_count = 0
            for ci, l in var_to_clauses[var]:
                if l > 0:
                    pos_count += 1
                else:
                    neg_count += 1
            
            # Use VSIDS polarity hint
            if pos_count > neg_count:
                value = True
            elif neg_count > pos_count:
                value = False
            else:
                value = random.Random(restart).random() < 0.5
            
            # Assign
            assignment[var] = value
            trail.append((var, decision_level))
            decision_level += 1
            trail_lim.append(len(trail))
            
            # Check time
            if time.time() > deadline:
                break
        
        if found_solution:
            break
        
        # Update best partial assignment
        # Count unsatisfied clauses in current assignment
        # Fill unassigned vars randomly
        temp_assign = list(assignment)
        for v in range(n):
            if temp_assign[v] is None:
                temp_assign[v] = random.Random(restart * 1000 + v).random() < 0.5
        
        unsat_count = 0
        for c in processed_clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if temp_assign[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = temp_assign
    
    # If no solution found, try WalkSAT as fallback
    if best_assignment is None or best_unsat_count > 0:
        # WalkSAT with multiple restarts
        rng = random.Random(0)
        best_walk_assign = best_assignment if best_assignment else [rng.random() < 0.5 for _ in range(n)]
        best_walk_unsat = best_unsat_count
        
        for walk_restart in range(20):
            if time.time() > deadline:
                break
            
            if walk_restart == 0 and best_assignment:
                current = list(best_assignment)
            else:
                current = [rng.random() < 0.5 for _ in range(n)]
            
            def count_unsat(assignment):
                cnt = 0
                for c in processed_clauses:
                    sat = False
                    for l in c:
                        v = abs(l) - 1
                        if assignment[v] == (l > 0):
                            sat = True
                            break
                    if not sat:
                        cnt += 1
                return cnt
            
            current_unsat = count_unsat(current)
            
            if current_unsat < best_walk_unsat:
                best_walk_unsat = current_unsat
                best_walk_assign = list(current)
            
            if current_unsat == 0:
                best_assignment = current
                break
            
            # WalkSAT steps
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Find unsatisfied clauses
                unsat_clauses = []
                for c in processed_clauses:
                    sat = False
                    for l in c:
                        v = abs(l) - 1
                        if current[v] == (l > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(c)
                
                if not unsat_clauses:
                    best_assignment = current
                    break
                
                # Pick a random unsatisfied clause
                clause = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
                
                # With probability 0.25, flip a random literal in the clause
                # Otherwise, flip the literal that minimizes unsat count
                if rng.random() < 0.25:
                    lit = clause[rng.randint(0, len(clause) - 1)]
                else:
                    best_lit = None
                    best_delta = float('inf')
                    for l in clause:
                        v = abs(l) - 1
                        # Simulate flip
                        old_val = current[v]
                        current[v] = not old_val
                        delta = count_unsat(current) - current_unsat
                        current[v] = old_val
                        if delta < best_delta:
                            best_delta = delta
                            best_lit = l
                    lit = best_lit
                
                v = abs(lit) - 1
                current[v] = not current[v]
                current_unsat += count_unsat(current) - current_unsat
                # Recalculate properly
                current_unsat = count_unsat(current)
                
                if current_unsat < best_walk_unsat:
                    best_walk_unsat = current_unsat
                    best_walk_assign = list(current)
                
                if current_unsat == 0:
                    best_assignment = current
                    break
            
            if best_assignment is not None:
                break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
