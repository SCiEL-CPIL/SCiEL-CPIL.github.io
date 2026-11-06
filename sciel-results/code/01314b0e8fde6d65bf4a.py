import sys
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Increase recursion limit for DPLL
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    time_limit = 290.0
    
    # Preprocess: 0-based literals
    processed_clauses = []
    for c in clauses:
        pc = [l - 1 if l > 0 else -l - 1 for l in c]
        processed_clauses.append(pc)
        
    num_clauses = len(processed_clauses)
    
    # DPLL Solver
    def dpll(assignment, active_clauses):
        if time.time() - start_time > time_limit:
            return None, False
            
        # 1. Check for empty clause
        for i in active_clauses:
            if not processed_clauses[i]:
                return None, False
                
        # 2. Unit Propagation & Pure Literal Elimination
        # We will collect unit assignments and pure assignments
        unit_assignments = {}
        pure_assignments = {}
        
        pos_count = [0] * n
        neg_count = [0] * n
        
        # First pass: Identify unit clauses and count occurrences
        for i in active_clauses:
            clause = processed_clauses[i]
            unassigned = []
            satisfied = False
            
            for l in clause:
                v = abs(l)
                if assignment[v] != -1:
                    if (l > 0 and assignment[v] == 1) or (l < 0 and assignment[v] == 0):
                        satisfied = True
                        break
                else:
                    unassigned.append(l)
                    if l > 0:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1
                        
            if satisfied:
                continue
                
            if len(unassigned) == 0:
                return None, False # Contradiction
                
            if len(unassigned) == 1:
                unit_assignments[abs(unassigned[0])] = unassigned[0]
                
        # Second pass: Pure literals (only if no unit assignments to avoid conflict? 
        # Actually, pure literals can be applied before or after units. 
        # If a var is unit, it's not pure unless the other sign doesn't appear.
        # Let's apply units first, then re-evaluate? 
        # Standard DPLL: Apply units. If conflict, fail. Then apply pure.
        
        if unit_assignments:
            new_assignment = assignment[:]
            for v, l in unit_assignments.items():
                new_assignment[v] = 1 if l > 0 else 0
            # Recurse with new assignment. Active clauses remain the same?
            # Yes, because we haven't removed satisfied clauses from the set yet, 
            # but the check `satisfied` above handles it.
            return dpll(new_assignment, active_clauses)
            
        # Check for pure literals
        for v in range(n):
            if assignment[v] == -1:
                if pos_count[v] > 0 and neg_count[v] == 0:
                    pure_assignments[v] = 1
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    pure_assignments[v] = 0
                    
        if pure_assignments:
            new_assignment = assignment[:]
            for v, val in pure_assignments.items():
                new_assignment[v] = val
            return dpll(new_assignment, active_clauses)
            
        # 3. Branching
        # Choose variable with max occurrences
        max_count = 0
        best_var = -1
        for v in range(n):
            if assignment[v] == -1:
                count = pos_count[v] + neg_count[v]
                if count > max_count:
                    max_count = count
                    best_var = v
                    
        if best_var == -1:
            # All variables assigned
            # Verify solution
            for i in active_clauses:
                sat = False
                for l in processed_clauses[i]:
                    v = abs(l)
                    if (l > 0 and assignment[v] == 1) or (l < 0 and assignment[v] == 0):
                        sat = True
                        break
                if not sat:
                    return None, False
            return assignment, True
            
        # Branch
        new_assign_1 = assignment[:]
        new_assign_1[best_var] = 1
        res, success = dpll(new_assign_1, active_clauses)
        if success:
            return res, True
            
        new_assign_0 = assignment[:]
        new_assign_0[best_var] = 0
        res, success = dpll(new_assign_0, active_clauses)
        if success:
            return res, True
            
        return None, False

    initial_assignment = [-1] * n
    active_clauses = set(range(num_clauses))
    
    result, success = dpll(initial_assignment, active_clauses)
    
    if success:
        return [bool(v) for v in result]
        
    # Fallback: WalkSAT
    rng = random.Random(12345)
    best_assign = None
    best_unsat = float('inf')
    
    def count_unsat(assign):
        count = 0
        for c in processed_clauses:
            sat = False
            for l in c:
                v = abs(l)
                if (l > 0 and assign[v]) or (l < 0 and not assign[v]):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        assign = [rng.random() < 0.5 for _ in range(n)]
        unsat = count_unsat(assign)
        if unsat == 0:
            return assign
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assign[:]
            
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                break
            unsat_clauses = []
            for i, c in enumerate(processed_clauses):
                sat = False
                for l in c:
                    v = abs(l)
                    if (l > 0 and assign[v]) or (l < 0 and not assign[v]):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assign
                
            c_idx = rng.choice(unsat_clauses)
            c = processed_clauses[c_idx]
            v = rng.choice([abs(l) for l in c])
            assign[v] = not assign[v]
            
            unsat = count_unsat(assign)
            if unsat == 0:
                return assign
            if unsat < best_unsat:
                best_unsat = unsat
                best_assign = assign[:]
                
    return [bool(v) for v in best_assign] if best_assign else [False] * n
