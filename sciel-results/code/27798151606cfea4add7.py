"""
Improved 3-SAT solver for planted instances (50 vars).
Uses DPLL with watched literals + local search portfolio (WalkSAT, min-conflicts).
"""
import random
import time
import math
import sys
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 290.0
    DPLL_TIME_BUDGET = 30.0  # Limit DPLL to give more time to local search
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    if time_remaining() <= 0:
        return [True] * n
    
    m = len(clauses)
    
    # Precompute clause data: list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    # Build variable-to-clause index for fast lookup
    var_to_clauses = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for var, pos in cd:
            var_to_clauses[var].append(i)
    
    def is_satisfied(assignment, clause_idx):
        for var, pos in clause_data[clause_idx]:
            if assignment[var] == pos:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            sat = False
            for var, pos in clause_data[i]:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def init_clause_sat(assignment):
        """Return list of bool: whether each clause is satisfied."""
        sat = []
        for i in range(m):
            s = False
            for var, pos in clause_data[i]:
                if assignment[var] == pos:
                    s = True
                    break
            sat.append(s)
        return sat
    
    def count_unsat_from_sat(sat_list):
        return m - sum(sat_list)
    
    def full_satisfied(assignment):
        for i in range(m):
            sat = False
            for var, pos in clause_data[i]:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # DPLL with watched literals, unit propagation, and pure literal elimination
    def dpll_solve():
        """DPLL solver with watched literals."""
        # assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n
        
        # Watched literals: for each clause, watch 2 literals
        # Represent literal as signed 1-based: var+1 for positive, -(var+1) for negative
        # clause_watches[i] = [lit1, lit2] where lit is signed 1-based literal
        clause_watches = []
        for i, c in enumerate(clauses):
            if len(c) >= 2:
                clause_watches.append([c[0], c[1]])
            elif len(c) == 1:
                clause_watches.append([c[0], c[0]])
            else:
                return None  # Empty clause = UNSAT
        
        # active_clauses: set of clause indices that are still active (not satisfied)
        active_clauses = set(range(m))
        
        # Queue for unit propagation
        unit_queue = []  # list of (var_idx, value) where value is 0 or 1
        
        def literal_satisfied(lit, var_idx, val):
            """Check if a signed 1-based literal is satisfied by assignment."""
            if val == -1:
                return False
            if lit > 0:
                return val == 1
            else:
                return val == 0
        
        def literal_falsified(lit, var_idx, val):
            """Check if a signed 1-based literal is falsified by assignment."""
            if val == -1:
                return False
            if lit > 0:
                return val == 0
            else:
                return val == 1
        
        def find_new_watch(clause_idx, bad_lit):
            """Find a new literal in clause to watch after bad_lit is falsified."""
            for lit in clauses[clause_idx]:
                if lit != bad_lit:
                    var_idx = abs(lit) - 1
                    if assignment[var_idx] == -1:
                        return lit
            return None
        
        def propagate():
            """Unit propagation. Returns False if conflict, True if OK."""
            while unit_queue:
                var_idx, val = unit_queue.pop()
                
                # Check for conflict
                if assignment[var_idx] != -1 and assignment[var_idx] != val:
                    return False
                
                assignment[var_idx] = val
                
                # Process all clauses watching this variable
                for ci in var_to_clauses[var_idx]:
                    if ci not in active_clauses:
                        continue
                    
                    w0, w1 = clause_watches[ci]
                    
                    # Check if either watch is satisfied
                    if literal_satisfied(w0, abs(w0)-1, assignment[abs(w0)-1]) or \
                       literal_satisfied(w1, abs(w1)-1, assignment[abs(w1)-1]):
                        continue
                    
                    # Check if either watch is on the just-assigned variable
                    bad = None
                    other = None
                    if abs(w0) - 1 == var_idx:
                        bad = w0
                        other = w1
                    elif abs(w1) - 1 == var_idx:
                        bad = w1
                        other = w0
                    else:
                        # Neither watch is on this variable, but clause might be satisfied
                        # by another literal
                        sat = False
                        for lit in clauses[ci]:
                            v = abs(lit) - 1
                            if literal_satisfied(lit, v, assignment[v]):
                                sat = True
                                break
                        if sat:
                            active_clauses.discard(ci)
                        continue
                    
                    if literal_falsified(bad, var_idx, val):
                        # Bad literal is falsified, need to find new watch
                        new_lit = find_new_watch(ci, bad)
                        if new_lit is None:
                            # All literals falsified -> conflict
                            return False
                        else:
                            # Replace bad with new
                            if w0 == bad:
                                clause_watches[ci] = [new_lit, w1]
                            else:
                                clause_watches[ci] = [w0, new_lit]
                            # If new literal is unit (other is also falsified or on assigned var)
                            other_val = assignment[abs(other)-1]
                            if literal_falsified(other, abs(other)-1, other_val):
                                # Both watches falsified? Shouldn't happen if new_lit found
                                # Actually, check if other is also on an assigned variable
                                pass
                        # Check if the new watch makes this a unit clause
                        nw0, nw1 = clause_watches[ci]
                        n0_sat = literal_satisfied(nw0, abs(nw0)-1, assignment[abs(nw0)-1])
                        n1_sat = literal_satisfied(nw1, abs(nw1)-1, assignment[abs(nw1)-1])
                        if n0_sat or n1_sat:
                            active_clauses.discard(ci)
                        else:
                            # Check if one watch is unassigned and other is falsified
                            n0_unassigned = assignment[abs(nw0)-1] == -1
                            n1_unassigned = assignment[abs(nw1)-1] == -1
                            n0_fals = literal_falsified(nw0, abs(nw0)-1, assignment[abs(nw0)-1])
                            n1_fals = literal_falsified(nw1, abs(nw1)-1, assignment[abs(nw1)-1])
                            if n0_fals and n1_unassigned:
                                unit_queue.append((abs(nw1)-1, 1 if nw1 > 0 else 0))
                            elif n1_fals and n0_unassigned:
                                unit_queue.append((abs(nw0)-1, 1 if nw0 > 0 else 0))
                    else:
                        # Bad literal is satisfied, clause is satisfied
                        active_clauses.discard(ci)
            
            return True
        
        # Initial unit propagation: find any unit clauses
        for i in range(m):
            if i not in active_clauses:
                continue
            # Check if clause is already satisfied
            sat = False
            for lit in clauses[i]:
                v = abs(lit) - 1
                if literal_satisfied(lit, v, assignment[v]):
                    sat = True
                    break
            if sat:
                active_clauses.discard(i)
                continue
            # Check for unit clause
            unassigned_lits = []
            for lit in clauses[i]:
                v = abs(lit) - 1
                if assignment[v] == -1:
                    unassigned_lits.append(lit)
                elif literal_falsified(lit, v, assignment[v]):
                    pass
            if len(unassigned_lits) == 1:
                lit = unassigned_lits[0]
                var_idx = abs(lit) - 1
                val = 1 if lit > 0 else 0
                if assignment[var_idx] == -1:
                    unit_queue.append((var_idx, val))
        
        if not propagate():
            return None
        
        # Check if all satisfied
        for i in range(m):
            if i in active_clauses:
                sat = False
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if literal_satisfied(lit, v, assignment[v]):
                        sat = True
                        break
                if not sat:
                    break
        else:
            # All clauses satisfied
            return [bool(a) for a in assignment]
        
        # Pure literal elimination
        def eliminate_pure():
            """Find and assign pure literals. Returns list of (var, val) or None if conflict."""
            # For each unassigned variable, check if it only appears with one sign
            pure_assignments = []
            for var_idx in range(n):
                if assignment[var_idx] != -1:
                    continue
                has_pos = False
                has_neg = False
                for ci in var_to_clauses[var_idx]:
                    if ci not in active_clauses:
                        continue
                    for lit in clauses[ci]:
                        if abs(lit) - 1 == var_idx:
                            if lit > 0:
                                has_pos = True
                            else:
                                has_neg = True
                            break
                if has_pos and not has_neg:
                    pure_assignments.append((var_idx, 1))
                elif has_neg and not has_pos:
                    pure_assignments.append((var_idx, 0))
            
            for var_idx, val in pure_assignments:
                if assignment[var_idx] != -1:
                    if assignment[var_idx] != val:
                        return None  # Conflict
                    continue
                assignment[var_idx] = val
                unit_queue.append((var_idx, val))
            
            return pure_assignments
        
        # Apply pure literal elimination
        while True:
            if time_remaining() < 5.0:
                break
            pl = eliminate_pure()
            if pl is None:
                return None
            if not pl:
                break
            if not propagate():
                return None
        
        # Recursively search
        def dpll_rec(depth):
            if time_remaining() < 5.0:
                return None
            
            # Check if all clauses satisfied
            all_sat = True
            for i in range(m):
                if i not in active_clauses:
                    continue
                sat = False
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if literal_satisfied(lit, v, assignment[v]):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                return [bool(a) for a in assignment]
            
            # Choose a variable to branch on
            # Use a heuristic: pick variable in most active clauses
            var_counts = [0] * n
            for i in active_clauses:
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if assignment[v] == -1:
                        var_counts[v] += 1
            
            # Find unassigned variable with highest count
            best_var = -1
            best_count = -1
            for v in range(n):
                if assignment[v] == -1 and var_counts[v] > best_count:
                    best_count = var_counts[v]
                    best_var = v
            
            if best_var == -1:
                # No unassigned variables but not all satisfied - shouldn't happen
                return None
            
            # Try both values
            for val in [1, 0]:
                if time_remaining() < 5.0:
                    return None
                
                # Save state
                saved_assignment = assignment[:]
                saved_active = active_clauses.copy()
                saved_watches = [w[:] for w in clause_watches]
                saved_queue = unit_queue[:]
                
                # Assign
                assignment[best_var] = val
                unit_queue.append((best_var, val))
                
                if propagate():
                    result = dpll_rec(depth + 1)
                    if result is not None:
                        return result
                
                # Backtrack
                assignment[:] = saved_assignment
                active_clauses.clear()
                active_clauses.update(saved_active)
                for i in range(m):
                    clause_watches[i] = saved_watches[i][:]
                unit_queue[:] = saved_queue
            
            return None
        
        return dpll_rec(0)
    
    # Try DPLL first with time limit
    dpll_start = time.time()
    dpll_result = dpll_solve()
    dpll_elapsed = time.time() - dpll_start
    if dpll_result is not None and full_satisfied(dpll_result):
        return dpll_result
    # If DPLL took too long, we still have time for local search
    # If it was fast, great - more time for local search
    
    # If DPLL didn't find a solution, use local search portfolio
    best_result = None
    best_unsat_count = float('inf')
    
    def try_solver(solver_func, seeds, steps_per_call):
        nonlocal best_result, best_unsat_count
        for seed in seeds:
            if time_remaining() < 5.0:
                return
            result = solver_func(seed, steps_per_call)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
        return None
    
    def walksat(seed, max_steps, noise=0.3):
        """WalkSAT with weighted clause selection and incremental tracking."""
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Clause weights
        weights = [1.0] * m
        
        # Track satisfaction incrementally
        clause_sat = init_clause_sat(assignment)
        current_unsat = m - sum(clause_sat)
        if current_unsat == 0:
            return assignment[:]
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        
        # Build list of unsatisfied clause indices
        unsat_list = [i for i in range(m) if not clause_sat[i]]
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break
            
            if not unsat_list:
                return assignment[:]
            
            # Weighted selection of an unsatisfied clause
            total_w = sum(weights[i] for i in unsat_list)
            if total_w > 0:
                r = rng.random() * total_w
                cum = 0
                chosen = unsat_list[0]
                for idx in unsat_list:
                    cum += weights[idx]
                    if r <= cum:
                        chosen = idx
                        break
            else:
                chosen = rng.choice(unsat_list)
            
            # Increase weight of chosen clause
            weights[chosen] *= 1.15
            # Decay other weights slightly (only for unsat ones to save time)
            for i in unsat_list:
                if i != chosen:
                    weights[i] = max(1.0, weights[i] * 0.98)
            
            clause = clause_data[chosen]
            
            # Choose which variable to flip
            # Use a min-conflicts-like heuristic with noise
            best_var = -1
            best_score = -1e9
            
            for var, pos in clause:
                # Score: how many clauses would be improved by flipping var
                score = 0
                for ci in var_to_clauses[var]:
                    if not clause_sat[ci]:
                        # Clause is unsatisfied; would flipping var satisfy it?
                        # The literal for var in this clause: find it
                        for v2, p2 in clause_data[ci]:
                            if v2 == var:
                                if (not assignment[var]) == p2:
                                    score += 1
                                break
                    else:
                        # Clause is satisfied; would flipping var unsatisfy it?
                        # Check if var is the only satisfying literal
                        only_sat = True
                        for v2, p2 in clause_data[ci]:
                            if v2 == var:
                                continue
                            if assignment[v2] == p2:
                                only_sat = False
                                break
                        if only_sat:
                            score -= 1
                
                # Add noise
                score += rng.gauss(0, noise)
                
                if score > best_score:
                    best_score = score
                    best_var = var
            
            if best_var < 0:
                best_var, _ = rng.choice(clause)
            
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            # Update clause satisfaction incrementally
            # Only need to update clauses containing best_var
            for ci in var_to_clauses[best_var]:
                if clause_sat[ci]:
                    # Was satisfied, check if still satisfied after flip
                    still_sat = False
                    for v2, p2 in clause_data[ci]:
                        if v2 == best_var:
                            if new_val == p2:
                                still_sat = True
                                break
                        else:
                            if assignment[v2] == p2:
                                still_sat = True
                                break
                    if not still_sat:
                        clause_sat[ci] = False
                        unsat_list.append(ci)
                        current_unsat += 1
                else:
                    # Was unsatisfied, check if now satisfied
                    now_sat = False
                    for v2, p2 in clause_data[ci]:
                        if v2 == best_var:
                            if new_val == p2:
                                now_sat = True
                                break
                        else:
                            if assignment[v2] == p2:
                                now_sat = True
                                break
                    if now_sat:
                        clause_sat[ci] = True
                        unsat_list.remove(ci)
                        current_unsat -= 1
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            if current_unsat == 0:
                return assignment[:]
        
        return best_assignment
    
    def min_conflicts(seed, max_steps):
        """Min-conflicts local search with incremental tracking."""
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_sat = init_clause_sat(assignment)
        current_unsat = m - sum(clause_sat)
        if current_unsat == 0:
            return assignment[:]
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        unsat_list = [i for i in range(m) if not clause_sat[i]]
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break
            
            if not unsat_list:
                return assignment[:]
            
            # Pick a random unsatisfied clause
            chosen = rng.choice(unsat_list)
            clause = clause_data[chosen]
            
            # For each variable in the clause, compute the number of unsatisfied clauses it appears in
            # that would be fixed by flipping it
            best_var = -1
            best_gain = -1e9
            
            for var, pos in clause:
                gain = 0
                for ci in var_to_clauses[var]:
                    if not clause_sat[ci]:
                        # Would flipping var satisfy this clause?
                        for v2, p2 in clause_data[ci]:
                            if v2 == var:
                                if (not assignment[var]) == p2:
                                    gain += 1
                                break
                    else:
                        # Would flipping var unsatisfy this clause?
                        only_sat = True
                        for v2, p2 in clause_data[ci]:
                            if v2 == var:
                                continue
                            if assignment[v2] == p2:
                                only_sat = False
                                break
                        if only_sat:
                            gain -= 1
                
                gain += rng.gauss(0, 0.1)
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            if best_var < 0:
                best_var, _ = rng.choice(clause)
            
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            # Update clause satisfaction incrementally
            for ci in var_to_clauses[best_var]:
                if clause_sat[ci]:
                    still_sat = False
                    for v2, p2 in clause_data[ci]:
                        if v2 == best_var:
                            if new_val == p2:
                                still_sat = True
                                break
                        else:
                            if assignment[v2] == p2:
                                still_sat = True
                                break
                    if not still_sat:
                        clause_sat[ci] = False
                        unsat_list.append(ci)
                        current_unsat += 1
                else:
                    now_sat = False
                    for v2, p2 in clause_data[ci]:
                        if v2 == best_var:
                            if new_val == p2:
                                now_sat = True
                                break
                        else:
                            if assignment[v2] == p2:
                                now_sat = True
                                break
                    if now_sat:
                        clause_sat[ci] = True
                        unsat_list.remove(ci)
                        current_unsat -= 1
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            if current_unsat == 0:
                return assignment[:]
        
        return best_assignment
    
    # Portfolio: try multiple strategies with different seeds
    best_result = None
    best_unsat_count = float('inf')
    
    def try_solver(solver_func, seeds, steps_per_call):
        nonlocal best_result, best_unsat_count
        for seed in seeds:
            if time_remaining() < 5.0:
                return
            result = solver_func(seed, steps_per_call)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
        return None
    
    # Strategy 1: WalkSAT with moderate noise
    seeds1 = [42, 123, 777, 2024, 555, 999, 31337, 12345, 67890, 24680, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
    steps1 = max(500, n * 15)
    
    for seed in seeds1:
        if time_remaining() < 5.0:
            break
        result = walksat(seed, steps1, noise=0.3)
        if result is not None:
            if full_satisfied(result):
                return result
            else:
                cnt = count_unsat(result)
                if cnt < best_unsat_count:
                    best_unsat_count = cnt
                    best_result = result[:]
    
    # Strategy 2: WalkSAT with lower noise (more deterministic)
    if time_remaining() > 10.0:
        seeds2 = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
        steps2 = max(300, n * 10)
        for seed in seeds2:
            if time_remaining() < 5.0:
                break
            result = walksat(seed, steps2, noise=0.1)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
    
    # Strategy 3: Min-conflicts
    if time_remaining() > 10.0:
        seeds3 = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
        steps3 = max(300, n * 10)
        for seed in seeds3:
            if time_remaining() < 5.0:
                break
            result = min_conflicts(seed, steps3)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
    
    # Strategy 4: WalkSAT with higher noise (more exploration)
    if time_remaining() > 10.0:
        seeds4 = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
        steps4 = max(200, n * 8)
        for seed in seeds4:
            if time_remaining() < 5.0:
                break
            result = walksat(seed, steps4, noise=0.5)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
    
    # Strategy 5: Short bursts of WalkSAT with many restarts
    if time_remaining() > 5.0:
        for seed in range(2000, 2050):
            if time_remaining() < 3.0:
                break
            result = walksat(seed, 100, noise=0.3)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
    
    # Strategy 6: Final polish - try to fix remaining unsat clauses
    if best_result is not None and best_unsat_count > 0 and time_remaining() > 5.0:
        # Try targeted flips on variables in unsatisfied clauses
        assignment = best_result[:]
        clause_sat = init_clause_sat(assignment)
        current_unsat = m - sum(clause_sat)
        unsat_list = [i for i in range(m) if not clause_sat[i]]
        
        # Try flipping each variable in unsat clauses to see if it helps
        improved = True
        while improved and time_remaining() > 3.0 and unsat_list:
            improved = False
            for ci in unsat_list[:]:
                if time_remaining() < 3.0:
                    break
                for var, pos in clause_data[ci]:
                    # Try flipping var
                    old_val = assignment[var]
                    assignment[var] = not old_val
                    
                    # Check if this improves things
                    new_unsat = 0
                    new_unsat_list = []
                    for i in range(m):
                        sat = False
                        for v2, p2 in clause_data[i]:
                            if assignment[v2] == p2:
                                sat = True
                                break
                        if not sat:
                            new_unsat += 1
                            new_unsat_list.append(i)
                    
                    if new_unsat < current_unsat:
                        current_unsat = new_unsat
                        unsat_list = new_unsat_list
                        improved = True
                        if current_unsat == 0:
                            return assignment
                    else:
                        assignment[var] = old_val
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_result = assignment[:]
    
    if best_result is not None:
        return best_result
    
    return [True] * n
