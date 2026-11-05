"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause data: list of (var_idx, sign) for each clause
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            v = abs(l) - 1
            s = (l > 0)
            cd.append((v, s))
        clause_data.append(cd)
    
    # Build variable-to-clauses index for fast update
    var_to_clauses = [[] for _ in range(n)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            var_to_clauses[v].append(ci)
    
    # Pre-compute sign lookup: clause_var_sign[ci][v] = True/False/None
    clause_var_sign = [[None]*n for _ in range(num_clauses)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            clause_var_sign[ci][v] = s
    
    # Deadline guard: 300s limit, leave 15s margin
    start_time = time.time()
    deadline = start_time + 285.0
    
    def check_time():
        return time.time() < deadline
    
    def init_state(assignment):
        sat_counts = [0] * num_clauses
        unsat_list = []
        for ci, cd in enumerate(clause_data):
            cnt = 0
            for v, s in cd:
                if assignment[v] == s:
                    cnt += 1
            sat_counts[ci] = cnt
            if cnt == 0:
                unsat_list.append(ci)
        satisfied = num_clauses - len(unsat_list)
        return sat_counts, satisfied, unsat_list
    
    def walksat_run(assignment, max_steps, p):
        """
        Perform WalkSAT steps with maintained unsat list.
        Returns (solved: bool, assignment, satisfied_count, sat_counts)
        """
        sat_counts, satisfied, unsat_list = init_state(assignment)
        
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts
        
        for _ in range(max_steps):
            if not check_time():
                return False, assignment, satisfied, sat_counts
            
            if not unsat_list:
                return True, assignment, satisfied, sat_counts
            
            # Pick random unsat clause
            ci = unsat_list[random.randrange(len(unsat_list))]
            cd = clause_data[ci]
            
            if random.random() < p:
                best_var = -1
                best_cost = float('inf')
                
                for v, s in cd:
                    cost = 0
                    for c_idx in var_to_clauses[v]:
                        if sat_counts[c_idx] > 0:
                            s_c = clause_var_sign[c_idx][v]
                            if s_c is not None and assignment[v] == s_c:
                                if sat_counts[c_idx] == 1:
                                    cost += 1
                    
                    total_cost = cost + random.random()
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_var = v
                v_to_flip = best_var
            else:
                v_to_flip = cd[random.randrange(len(cd))][0]
            
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            for c_idx in var_to_clauses[v_to_flip]:
                s_c = clause_var_sign[c_idx][v_to_flip]
                if s_c is not None:
                    was_sat = (old_val == s_c)
                    is_sat = (new_val == s_c)
                    
                    if was_sat and not is_sat:
                        sat_counts[c_idx] -= 1
                        if sat_counts[c_idx] == 0:
                            satisfied -= 1
                            unsat_list.append(c_idx)
                    elif not was_sat and is_sat:
                        sat_counts[c_idx] += 1
                        if sat_counts[c_idx] == 1:
                            satisfied += 1
                            # Remove from unsat_list: find and swap-remove
                            for j in range(len(unsat_list)):
                                if unsat_list[j] == c_idx:
                                    unsat_list[j] = unsat_list[-1]
                                    unsat_list.pop()
                                    break
            
            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts
        
        return False, assignment, satisfied, sat_counts
    
    # Main loop: Restarts with varying parameters
    best_assignment = None
    best_satisfied = -1
    
    # Parameter sweep with tuned values
    p_values = [0.55, 0.65, 0.75, 0.85]
    steps_list = [500, 2000, 10000, 50000]
    
    for p in p_values:
        for steps in steps_list:
            if not check_time():
                break
            
            while check_time():
                assignment = [random.random() < 0.5 for _ in range(n)]
                solved, assignment, satisfied, _ = walksat_run(assignment, steps, p)
                
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = assignment[:]
                
                if solved:
                    return assignment
    
    # If we get here, we didn't solve. Return best assignment found.
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
