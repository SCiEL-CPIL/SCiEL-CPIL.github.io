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
    # sign is True if literal is positive (var must be True), False if negative
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
    
    # Deadline guard: 300s limit, leave 10s margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    def check_time():
        return time.time() < deadline
    
    # State: assignment as list of bools
    # satisfied_count: number of satisfied clauses
    # For each clause, we track how many literals are satisfied (0, 1, 2, 3)
    # If count > 0, clause is satisfied.
    
    def init_state(assignment):
        sat_counts = [0] * num_clauses
        for ci, cd in enumerate(clause_data):
            cnt = 0
            for v, s in cd:
                if assignment[v] == s:
                    cnt += 1
            sat_counts[ci] = cnt
        satisfied = sum(1 for c in sat_counts if c > 0)
        return sat_counts, satisfied
    
    def update_flip(var_idx, sat_counts, satisfied):
        """Flip var_idx in assignment, update sat_counts and satisfied count.
        Returns new satisfied count. Assumes assignment[var_idx] is already flipped."""
        # Note: caller must flip assignment[var_idx] before or after? 
        # Let's say caller flips assignment first, then calls this.
        # We need to know the OLD value to compute the change.
        # Actually, it's easier to pass the old value or compute delta based on new value.
        # Let's restructure: 
        # 1. Record old value
        # 2. Flip assignment
        # 3. For each clause containing var_idx:
        #    - If old assignment satisfied the literal, decrement count
        #    - If new assignment satisfies the literal, increment count
        #    - Update satisfied count if count crosses 0 threshold
        pass # This logic will be implemented in the main loop for speed

    # Optimized WalkSAT step
    # We maintain:
    # assignment: list of bools
    # sat_counts: list of ints (0-3) for each clause
    # satisfied: int
    
    def walksat_run(assignment, max_steps, p):
        """
        Perform WalkSAT steps.
        Returns (solved: bool, assignment, satisfied_count, sat_counts)
        """
        sat_counts, satisfied = init_state(assignment)
        
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts
        
        for _ in range(max_steps):
            if not check_time():
                return False, assignment, satisfied, sat_counts
            
            # Find unsatisfied clauses
            # Optimization: keep a list of unsat clause indices?
            # For n=50, num_clauses ~ 200-500, scanning is fast.
            # But we can maintain a set/list of unsat clauses for O(1) random pick.
            # Let's try scanning first, it's simple.
            
            unsat_indices = [i for i in range(num_clauses) if sat_counts[i] == 0]
            
            if not unsat_indices:
                return True, assignment, satisfied, sat_counts
            
            # Pick random unsat clause
            ci = random.choice(unsat_indices)
            cd = clause_data[ci]
            
            # Choose literal to flip
            # With prob p: pick literal that minimizes breakage (number of satisfied clauses that would become unsatisfied)
            # With prob 1-p: pick random literal
            
            if random.random() < p:
                best_var = -1
                best_cost = float('inf')
                best_rand = 0
                
                for v, s in cd:
                    # Calculate cost of flipping v
                    # Cost = number of satisfied clauses containing v that would become unsatisfied
                    # A clause becomes unsatisfied if:
                    # 1. It is currently satisfied (sat_counts[c] > 0)
                    # 2. The literal (v, s_v) in that clause is currently satisfied
                    # 3. After flip, (v, s_v) is no longer satisfied
                    # 4. No other literal in that clause is satisfied (i.e., sat_counts[c] == 1 and this is the only one)
                    
                    # Actually, simpler:
                    # For each clause c containing v:
                    # If sat_counts[c] > 0:
                    #   Check if literal (v, s_c) is currently satisfied.
                    #   If yes, and sat_counts[c] == 1, then flipping v will make it unsat. Cost +1.
                    #   If sat_counts[c] > 1, flipping v might make it unsat only if it was the only one? No.
                    #   If sat_counts[c] == 1 and the satisfied literal is (v, s_c), then flipping v makes it 0.
                    #   If sat_counts[c] > 1, flipping v reduces count by 1, but it stays > 0 unless it was 1.
                    
                    cost = 0
                    for c_idx in var_to_clauses[v]:
                        if sat_counts[c_idx] > 0:
                            # Check if this clause's literal for v is satisfied
                            # Find the sign of v in clause c_idx
                            # We need to know the sign. We have var_to_clauses[v] = list of clause indices.
                            # We need a map from (clause, var) to sign.
                            # Let's precompute clause_lit_sign[ci][v] = sign (True/False) or None.
                            # Or just loop through cd of c_idx? That's slow.
                            # Let's precompute a dict or list.
                            # For now, let's assume we can check quickly.
                            # We'll precompute `clause_var_sign` as a list of dicts or parallel lists.
                            # Since n is small, let's just iterate clause_data[c_idx] to find v? 
                            # That's O(3) per clause. Total O(3 * num_clauses_per_var).
                            # num_clauses_per_var is ~ 3 * num_clauses / n.
                            # This is acceptable.
                            
                            # Find sign of v in clause c_idx
                            s_c = None
                            for vv, ss in clause_data[c_idx]:
                                if vv == v:
                                    s_c = ss
                                    break
                            
                            if s_c is not None:
                                # Is the literal currently satisfied?
                                # Current assignment of v is assignment[v]
                                # Literal is satisfied if assignment[v] == s_c
                                if assignment[v] == s_c:
                                    # This literal is satisfied.
                                    # If sat_counts[c_idx] == 1, then flipping v makes it unsat.
                                    if sat_counts[c_idx] == 1:
                                        cost += 1
                    
                    # Add small random tie-breaker
                    rand_val = random.random()
                    total_cost = cost + rand_val
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_var = v
                        best_rand = rand_val
                v_to_flip = best_var
            else:
                # Random literal from clause
                v_to_flip = random.choice(cd)[0]
            
            # Flip v_to_flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update sat_counts and satisfied
            # For each clause containing v_to_flip:
            for c_idx in var_to_clauses[v_to_flip]:
                # Find sign of v_to_flip in clause c_idx
                s_c = None
                for vv, ss in clause_data[c_idx]:
                    if vv == v_to_flip:
                        s_c = ss
                        break
                
                if s_c is not None:
                    # Was the literal satisfied before?
                    was_sat = (old_val == s_c)
                    # Is the literal satisfied now?
                    is_sat = (new_val == s_c)
                    
                    if was_sat and not is_sat:
                        # Lost a satisfaction
                        sat_counts[c_idx] -= 1
                        if sat_counts[c_idx] == 0:
                            satisfied -= 1
                    elif not was_sat and is_sat:
                        # Gained a satisfaction
                        sat_counts[c_idx] += 1
                        if sat_counts[c_idx] == 1:
                            satisfied += 1
            
            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts
        
        return False, assignment, satisfied, sat_counts

    # Main loop: Restarts with varying parameters
    best_assignment = None
    best_satisfied = -1
    
    # Strategy:
    # 1. Quick random restarts with short steps
    # 2. If not solved, longer runs
    # 3. Use increasing p (greediness) or decreasing?
    # Standard WalkSAT: p around 0.5-0.8 works well.
    
    # Pre-compute sign lookup for speed: clause_var_sign[ci][v]
    # Using a list of dicts for clarity, or a 2D list.
    # Since n=50, a 2D list of size [num_clauses][n] with None for absent is fine.
    clause_var_sign = [[None]*n for _ in range(num_clauses)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            clause_var_sign[ci][v] = s
    
    # Refine walksat_run to use clause_var_sign
    def walksat_run_opt(assignment, max_steps, p):
        sat_counts, satisfied = init_state(assignment)
        
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts
        
        for _ in range(max_steps):
            if not check_time():
                return False, assignment, satisfied, sat_counts
            
            # Find unsat clauses
            # Optimization: maintain a list of unsat clauses?
            # For now, scan.
            unsat_indices = []
            for i in range(num_clauses):
                if sat_counts[i] == 0:
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return True, assignment, satisfied, sat_counts
            
            ci = random.choice(unsat_indices)
            cd = clause_data[ci]
            
            if random.random() < p:
                best_var = -1
                best_cost = float('inf')
                
                for v, s in cd:
                    cost = 0
                    for c_idx in var_to_clauses[v]:
                        if sat_counts[c_idx] > 0:
                            s_c = clause_var_sign[c_idx][v]
                            if s_c is not None:
                                if assignment[v] == s_c:
                                    if sat_counts[c_idx] == 1:
                                        cost += 1
                    
                    total_cost = cost + random.random()
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_var = v
                v_to_flip = best_var
            else:
                v_to_flip = random.choice(cd)[0]
            
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
                    elif not was_sat and is_sat:
                        sat_counts[c_idx] += 1
                        if sat_counts[c_idx] == 1:
                            satisfied += 1
            
            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts
        
        return False, assignment, satisfied, sat_counts

    # Parameter sweep
    # p values: 0.5, 0.6, 0.7, 0.8
    # steps: 100, 500, 1000, 5000, 10000
    
    # We'll do a few restarts for each p, with increasing steps.
    
    p_values = [0.5, 0.7, 0.8]
    steps_list = [200, 1000, 5000, 20000]
    
    for p in p_values:
        for steps in steps_list:
            if not check_time():
                break
            
            # Do a few restarts for this setting
            # Number of restarts depends on time.
            # Let's do 1 restart first, then more if time allows.
            # Actually, just loop until time runs out or solved.
            
            while check_time():
                assignment = [random.random() < 0.5 for _ in range(n)]
                solved, assignment, satisfied, _ = walksat_run_opt(assignment, steps, p)
                
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
