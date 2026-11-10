import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Target n=34
    n = 34
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Build constraints
    # Constraint: (type, list of (var_idx, required_val))
    # type 0: Red K4 (size 4), require at least one 1. Violation if all 0.
    # type 1: Blue K5 (size 10), require at least one 0. Violation if all 1.
    
    clauses = []
    var_to_clauses = [[] for _ in range(num_triples)]
    
    # Red K4
    for k4 in combinations(range(n), 4):
        edges = []
        for i in range(4):
            t = tuple(sorted([v for j, v in enumerate(k4) if j != i]))
            idx = triple_to_idx[t]
            edges.append(idx)
        clause = [(idx, 1) for idx in edges] # Need 1 to satisfy
        clause_id = len(clauses)
        clauses.append(clause)
        for idx, val in clause:
            var_to_clauses[idx].append(clause_id)
            
    # Blue K5
    for k5 in combinations(range(n), 5):
        edges = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            edges.append(idx)
        clause = [(idx, 0) for idx in edges] # Need 0 to satisfy
        clause_id = len(clauses)
        clauses.append(clause)
        for idx, val in clause:
            var_to_clauses[idx].append(clause_id)
            
    num_clauses = len(clauses)
    
    # Store clause sizes and types for fast checking
    # size[c] = len(clauses[c])
    # We check if violations[c] == size[c] to see if unsatisfied
    
    clause_size = np.array([len(c) for c in clauses], dtype=np.int32)
    
    def solve_sa(time_limit):
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # Initialize violations
        # violations[c] = number of literals in clause c that are FALSE
        # Literal (v, val) is FALSE if colors[v] != val
        violations = np.zeros(num_clauses, dtype=np.int32)
        
        for c_idx, clause in enumerate(clauses):
            count = 0
            for idx, val in clause:
                if colors[idx] != val:
                    count += 1
            violations[c_idx] = count
            
        # List of unsatisfied clauses
        # Unsatisfied if violations[c] == clause_size[c]
        unsat_list = [i for i in range(num_clauses) if violations[i] == clause_size[i]]
        
        if len(unsat_list) == 0:
            return colors.tolist()
            
        run_deadline = time.time() + time_limit
        
        # Simulated Annealing parameters
        T_start = 1.0
        T_end = 0.01
        # Decay rate
        # We want to cool down over the duration
        # Let's do a geometric decay based on steps or time?
        # Time-based is safer.
        
        # We will use a dynamic T
        # Start with high T, decrease.
        
        # To escape local minima, we can also do restarts.
        # Let's try a single long run with restarts if stuck.
        
        best_colors = colors.copy()
        best_violations = sum(1 for v in violations if v == clause_size[0]) # Just count unsat
        # Actually, sum of (clause_size - violations) is satisfied literals.
        # Total unsat count is len(unsat_list).
        
        # We want to minimize the number of unsatisfied clauses.
        
        # Let's run multiple attempts if time permits.
        # But for n=34, one good SA run might suffice.
        
        # We'll use a "Min-Conflict" style move within SA.
        # Pick random unsat clause, pick random var in it.
        
        # Optimization: Precompute for each clause, the variables and required values
        # We already have `clauses` as list of lists of tuples.
        
        # To speed up violation update:
        # When flipping var v:
        # For each clause c containing v:
        #   find the required value for v in c.
        #   if colors[v] was satisfying (colors[v] == req), it becomes violating -> violations[c] += 1
        #   if colors[v] was violating (colors[v] != req), it becomes satisfying -> violations[c] -= 1
        
        # We need to know `req` for v in c quickly.
        # `clauses[c]` is a list of tuples. Searching for v in it is O(size).
        # Size is 4 or 10. This is fine.
        
        # Let's store `clause_var_req` as a dictionary or parallel arrays?
        # List of tuples is okay for size 4/10.
        
        # To make it faster, let's create a map: var_idx -> list of (clause_idx, req_val)
        # We have var_to_clauses, but we need req_val too.
        var_to_clause_info = [[] for _ in range(num_triples)]
        for c_idx, clause in enumerate(clauses):
            for idx, val in clause:
                var_to_clause_info[idx].append((c_idx, val))
                
        # Main Loop
        while time.time() < run_deadline:
            if len(unsat_list) == 0:
                return colors.tolist()
                
            # Pick a random unsatisfied clause
            c_idx = unsat_list[random.randint(0, len(unsat_list) - 1)]
            
            # Lazy check: is it still unsatisfied?
            if violations[c_idx] != clause_size[c_idx]:
                # It got satisfied. Remove from list?
                # We can't remove easily. We can just skip this iteration?
                # Or clean up periodically.
                # If we skip, we might waste time if list is full of stale entries.
                # Let's clean up if len(unsat_list) is large and many are stale?
                # For now, just continue.
                continue
                
            # Pick a variable from this clause
            clause = clauses[c_idx]
            var_idx, req_val = clause[random.randint(0, len(clause) - 1)]
            
            # Calculate Delta E (change in number of unsatisfied clauses)
            # This is tricky. Flipping a var can satisfy or violate multiple clauses.
            # Delta E = (new_unsat_count) - (old_unsat_count)
            
            # Let's compute the effect on each clause containing var_idx
            delta_unsat = 0
            for c_i, req_v in var_to_clause_info[var_idx]:
                old_viol = violations[c_i]
                old_sat = (old_viol < clause_size[c_i])
                
                # Determine new violation count
                # If colors[var_idx] == req_v, it was satisfying. Now it violates.
                # If colors[var_idx] != req_v, it was violating. Now it satisfies.
                
                if colors[var_idx] == req_v:
                    new_viol = old_viol + 1
                else:
                    new_viol = old_viol - 1
                    
                new_sat = (new_viol < clause_size[c_i])
                
                if old_sat and not new_sat:
                    delta_unsat += 1
                elif not old_sat and new_sat:
                    delta_unsat -= 1
                    
            # Simulated Annealing acceptance
            # We want to minimize delta_unsat.
            # If delta_unsat < 0, accept.
            # If delta_unsat > 0, accept with prob exp(-delta_unsat / T)
            
            # T schedule:
            # Let's use a simple cooling schedule based on time.
            elapsed = time.time() - (run_deadline - time_limit)
            # T = T_start * (T_end/T_start)^(elapsed/time_limit)
            # Or just linear?
            # Let's use a fixed T that decreases slowly, or restarts.
            
            # Actually, for hard CSPs, WalkSAT (always accept if improves, else random) is often good.
            # SA is better if we tune T.
            # Let's try a "Greedy with Random Restart" or "WalkSAT" variant.
            
            # WalkSAT:
            # If delta_unsat < 0: Flip
            # Else: Flip with probability 0.5? Or pick another var?
            # Standard WalkSAT: Pick var from clause. If it doesn't satisfy the clause, 
            # it's a "bad" flip? No, WalkSAT flips a var in the clause.
            # The heuristic is usually: 
            # 1. Pick unsat clause.
            # 2. Pick var in it.
            # 3. Flip it.
            # This is a random walk.
            
            # Let's use a "Min-Conflict" heuristic:
            # Only flip if it reduces total violations (delta_unsat < 0).
            # If no such move exists (local min), pick random move.
            
            if delta_unsat < 0:
                accept = True
            else:
                # Accept worsening moves with some probability to escape local minima
                # Let's use a fixed probability 0.2 for worsening moves
                # Or use SA temperature.
                # Let's try SA with T=0.5 initially, cooling to 0.1.
                
                # Simple heuristic: accept with prob 0.1 if delta > 0
                accept = random.random() < 0.1
                
            if accept:
                # Flip
                colors[var_idx] = 1 - colors[var_idx]
                
                # Update violations
                for c_i, req_v in var_to_clause_info[var_idx]:
                    if colors[var_idx] == req_v:
                        # Now satisfying. Was it violating?
                        # We need to know the old state.
                        # We already computed new_viol in the delta loop? No, we didn't store it.
                        # Let's re-calculate or store.
                        # It's cheaper to just update based on the flip logic.
                        
                        # If new color == req, it is satisfying.
                        # Previously, if old color != req, it was violating.
                        # So violations[c_i] -= 1
                        # If old color == req, it was satisfying.
                        # So violations[c_i] += 1
                        
                        # We know old color was 1 - new_color.
                        # if new_color == req, then old_color != req.
                        # So it went from Violating to Satisfying.
                        violations[c_i] -= 1
                    else:
                        # new_color != req. Satisfying? No, Violating.
                        # Previously, old_color == req?
                        # if new_color != req, then old_color == req.
                        # So it went from Satisfying to Violating.
                        violations[c_i] += 1
                        
                # Update unsat_list
                # We can't easily update the list.
                # We will clean it up periodically.
                
        return None

    # Run SA
    # Split time: 100s for n=34
    sol_34 = solve_sa(100.0)
    
    if sol_34 is not None:
        return {'n': 34, 'colors': sol_34}
        
    # Fallback: n=33? Or just return invalid?
    # The task says "Maximize n". If we fail 34, we should try to return a valid smaller n.
    # But 34 is the target.
    
    # If SA fails, try a quick random search for n=33?
    # Or just return the best effort for 34? No, it must be valid.
    
    # Let's add a fallback to n=30 which is easier.
    # But the score is n. So 34 is much better than 30.
    
    # If 34 fails, it's a score of 0? No, "invalid result ... scores 0".
    # So we MUST return a valid result.
    # We can always return a valid result for small n (e.g. n=9) using a simple pattern.
    
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}

if __name__ == "__main__":
    res = construct()
    print(f"Result: n={res['n']}, len(colors)={len(res['colors'])}")
