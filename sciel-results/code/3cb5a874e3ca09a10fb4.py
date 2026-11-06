# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Precompute variable to clause mapping and literal signs
    var_to_clauses = [[] for _ in range(n)]
    clause_var_info = []
    var_sign_in_clause = []
    
    for c_idx, clause in enumerate(clauses):
        info = []
        sign_map = {}
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            info.append((v, s))
            sign_map[v] = s
            var_to_clauses[v].append(c_idx)
        clause_var_info.append(info)
        var_sign_in_clause.append(sign_map)

    m = len(clauses)
    
    # Deadline: 325 seconds to be safe against the 330s kill
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    noise = 0.5
    restart_threshold = 5000
    TIME_CHECK_INTERVAL = 50
    POLISH_THRESHOLD = 10

    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial clause weights (number of satisfied literals)
        clause_weights = [0] * m
        unsat_set = set()
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_set.add(i)
        
        total_unsat = len(unsat_set)
        
        if total_unsat == 0:
            return assignment
        
        # Unit propagation: force variables in clauses with only one unsatisfied literal
        # Actually, for 3-SAT, unit clauses are rare. Instead, do a simple forcing pass:
        # If a variable appears in exactly one clause and that clause is unsat, flip it.
        # This is a weak form but can help.
        
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        recent_flips = [0] * n  # Track when each variable was last flipped
        step_count = 0
        
        for step in range(100000):
            step_count = step
            if step % TIME_CHECK_INTERVAL == 0 and time.time() >= deadline:
                break
            
            if total_unsat == 0:
                return assignment
            
            # Track best solution
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]
            
            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > restart_threshold:
                break
            
            # Find an unsatisfied clause using O(1) random selection
            if total_unsat > 0:
                unsat_list = list(unsat_set)
                clause_idx = unsat_list[rng.randrange(len(unsat_list))]
            else:
                return assignment
            
            clause_info = clause_var_info[clause_idx]
            
            # Heuristic: Pick variable that minimizes new unsat clauses
            if rng.random() < noise:
                # Random walk step - avoid recently flipped variables
                candidates = []
                for v, s in clause_info:
                    if step - recent_flips[v] > 3:  # Not flipped in last 3 steps
                        candidates.append(v)
                if candidates:
                    var_idx = candidates[rng.randrange(len(candidates))]
                else:
                    v, s = rng.choice(clause_info)
                    var_idx = v
            else:
                # Greedy step: Find variable in clause that gives best gain
                best_gain = float('inf')
                best_var_idx = -1
                best_freq = float('inf')
                
                for v, s in clause_info:
                    var_idx = v
                    current_val = assignment[var_idx]
                    
                    # Calculate delta in total_unsat
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_sign_in_clause[c_idx][var_idx]
                        
                        is_sat = (assignment[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1
                    
                    freq = len(var_to_clauses[var_idx])
                    if delta < best_gain or (delta == best_gain and freq < best_freq):
                        best_gain = delta
                        best_var_idx = var_idx
                        best_freq = freq
                        if best_gain == -1:
                            break
                            
                var_idx = best_var_idx
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            recent_flips[var_idx] = step
            
            # Update clause weights
            for c_idx in var_to_clauses[var_idx]:
                old_w = clause_weights[c_idx]
                c_sign = var_sign_in_clause[c_idx][var_idx]
                is_sat = (assignment[var_idx] == (c_sign == 1))
                
                if is_sat:
                    clause_weights[c_idx] = old_w + 1
                else:
                    clause_weights[c_idx] = old_w - 1
                    
                if old_w == 0 and clause_weights[c_idx] > 0:
                    total_unsat -= 1
                    unsat_set.discard(c_idx)
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    total_unsat += 1
                    unsat_set.add(c_idx)
            
            # POLISH PASS: When very close to solution, do exhaustive local search
            if total_unsat <= POLISH_THRESHOLD and total_unsat > 0:
                polish_success = False
                polish_deadline = time.time() + 8.0  # Max 8s for polish
                
                while time.time() < polish_deadline:
                    if total_unsat == 0:
                        return assignment
                    
                    # Try flipping each variable in unsat clauses
                    flipped = False
                    unsat_clauses = list(unsat_set)
                    rng.shuffle(unsat_clauses)
                    
                    for c_idx in unsat_clauses:
                        if time.time() >= polish_deadline:
                            break
                        
                        if c_idx not in unsat_set:
                            continue
                        
                        clause_info = clause_var_info[c_idx]
                        best_delta = 0
                        best_v = -1
                        
                        for v, s in clause_info:
                            delta = 0
                            for cc_idx in var_to_clauses[v]:
                                old_w = clause_weights[cc_idx]
                                c_sign = var_sign_in_clause[cc_idx][v]
                                is_sat = (assignment[v] == (c_sign == 1))
                                
                                if is_sat:
                                    if old_w == 1:
                                        delta += 1
                                else:
                                    if old_w == 0:
                                        delta -= 1
                            
                            if delta < best_delta:
                                best_delta = delta
                                best_v = v
                        
                        if best_v != -1 and best_delta < 0:
                            v = best_v
                            assignment[v] = not assignment[v]
                            
                            for cc_idx in var_to_clauses[v]:
                                old_w = clause_weights[cc_idx]
                                c_sign = var_sign_in_clause[cc_idx][v]
                                is_sat = (assignment[v] == (c_sign == 1))
                                
                                if is_sat:
                                    clause_weights[cc_idx] = old_w + 1
                                else:
                                    clause_weights[cc_idx] = old_w - 1
                                    
                                if old_w == 0 and clause_weights[cc_idx] > 0:
                                    total_unsat -= 1
                                    unsat_set.discard(cc_idx)
                                elif old_w > 0 and clause_weights[cc_idx] == 0:
                                    total_unsat += 1
                                    unsat_set.add(cc_idx)
                            
                            flipped = True
                            if total_unsat == 0:
                                return assignment
                            break
                    
                    if not flipped:
                        # Try 2-opt: flip pairs of variables
                        pair_found = False
                        unsat_clauses2 = list(unsat_set)
                        if len(unsat_clauses2) >= 2:
                            # Sample a few pairs
                            for _ in range(min(20, len(unsat_clauses2) * (len(unsat_clauses2) - 1) // 2)):
                                if time.time() >= polish_deadline:
                                    break
                                i = rng.randrange(len(unsat_clauses2))
                                j = rng.randrange(len(unsat_clauses2) - 1)
                                if j >= i:
                                    j += 1
                                c1 = unsat_clauses2[i]
                                c2 = unsat_clauses2[j]
                                if c1 not in unsat_set or c2 not in unsat_set:
                                    continue
                                
                                # Get variables from both clauses
                                vars1 = [v for v, s in clause_var_info[c1]]
                                vars2 = [v for v, s in clause_var_info[c2]]
                                
                                for v1 in vars1:
                                    if time.time() >= polish_deadline:
                                        break
                                    for v2 in vars2:
                                        if v1 == v2:
                                            continue
                                        # Calculate delta for flipping both v1 and v2
                                        delta = 0
                                        for cc_idx in set(var_to_clauses[v1] + var_to_clauses[v2]):
                                            old_w = clause_weights[cc_idx]
                                            # Check if v1 or v2 appears in this clause
                                            c_sign_v1 = var_sign_in_clause[cc_idx].get(v1, None)
                                            c_sign_v2 = var_sign_in_clause[cc_idx].get(v2, None)
                                            
                                            # Current satisfaction
                                            old_sat = 0
                                            if c_sign_v1 is not None:
                                                if assignment[v1] == (c_sign_v1 == 1):
                                                    old_sat += 1
                                            if c_sign_v2 is not None:
                                                if assignment[v2] == (c_sign_v2 == 1):
                                                    old_sat += 1
                                            
                                            # New satisfaction after flipping both
                                            new_sat = 0
                                            if c_sign_v1 is not None:
                                                if (not assignment[v1]) == (c_sign_v1 == 1):
                                                    new_sat += 1
                                            if c_sign_v2 is not None:
                                                if (not assignment[v2]) == (c_sign_v2 == 1):
                                                    new_sat += 1
                                            
                                            if old_sat > 0 and new_sat == 0:
                                                delta += 1
                                            elif old_sat == 0 and new_sat > 0:
                                                delta -= 1
                                        
                                        if delta < 0:
                                            # Flip both
                                            assignment[v1] = not assignment[v1]
                                            assignment[v2] = not assignment[v2]
                                            
                                            for cc_idx in var_to_clauses[v1]:
                                                old_w = clause_weights[cc_idx]
                                                c_sign = var_sign_in_clause[cc_idx][v1]
                                                is_sat = (assignment[v1] == (c_sign == 1))
                                                if is_sat:
                                                    clause_weights[cc_idx] = old_w + 1
                                                else:
                                                    clause_weights[cc_idx] = old_w - 1
                                                if old_w == 0 and clause_weights[cc_idx] > 0:
                                                    total_unsat -= 1
                                                    unsat_set.discard(cc_idx)
                                                elif old_w > 0 and clause_weights[cc_idx] == 0:
                                                    total_unsat += 1
                                                    unsat_set.add(cc_idx)
                                            
                                            for cc_idx in var_to_clauses[v2]:
                                                old_w = clause_weights[cc_idx]
                                                c_sign = var_sign_in_clause[cc_idx][v2]
                                                is_sat = (assignment[v2] == (c_sign == 1))
                                                if is_sat:
                                                    clause_weights[cc_idx] = old_w + 1
                                                else:
                                                    clause_weights[cc_idx] = old_w - 1
                                                if old_w == 0 and clause_weights[cc_idx] > 0:
                                                    total_unsat -= 1
                                                    unsat_set.discard(cc_idx)
                                                elif old_w > 0 and clause_weights[cc_idx] == 0:
                                                    total_unsat += 1
                                                    unsat_set.add(cc_idx)
                                            
                                            if total_unsat == 0:
                                                return assignment
                                            pair_found = True
                                            break
                                    if pair_found:
                                        break
                                if pair_found:
                                    break
                            if pair_found:
                                flipped = True
                        if not flipped:
                            break
                
                if total_unsat == 0:
                    return assignment

    return best_assignment
# EVOLVE-BLOCK-END
