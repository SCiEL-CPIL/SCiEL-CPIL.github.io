"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: must finish well within 330s wall-clock
    deadline = time.time() + 290.0
    
    # Preprocess: convert to 0-based, build clause list
    # Each clause: list of signed literals (1-based)
    # We'll work with a fast representation
    
    m = len(clauses)
    
    # Build literal -> clause indices mapping for efficiency
    # lit is 1-based signed; var = abs(lit)-1, sign = lit > 0
    # We'll track for each variable which clauses contain it and with what sign
    
    # For WalkSAT with efficient tracking:
    # assignment: list of bools, True = positive literal true
    # For each clause, track number of true literals
    # For each variable, track which clauses it appears in
    
    var_clauses = [[] for _ in range(n)]  # var_clauses[v] = list of (clause_idx, sign) where sign=1 means positive lit
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v].append((ci, sign))
    
    def try_solve(seed_offset=0, max_restarts=100):
        rng = random.Random(42 + seed_offset)
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial satisfied counts per clause
        # sat_count[c] = number of true literals in clause c
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            cnt = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    cnt += 1
            sat_count[ci] = cnt
        
        # unsat_clauses: set of clause indices that are unsatisfied
        unsat = set(ci for ci in range(m) if sat_count[ci] == 0)
        
        if not unsat:
            return assignment
        
        best_unsat = len(unsat)
        best_assignment = assignment[:]
        
        # WalkSAT with adaptive parameters
        # Use a combination of greedy and random flips
        step = 0
        max_steps = 50000
        restart_after = 5000
        no_improve_count = 0
        last_best = len(unsat)
        
        # Adaptive: track weight for random vs greedy
        # w = probability of picking a random literal from the clause
        # vs picking the literal that maximizes satisfied clauses (greedy)
        
        while step < max_steps:
            if time.time() > deadline:
                return best_assignment
            
            if not unsat:
                return assignment
            
            # Check if we've improved
            if len(unsat) < last_best:
                last_best = len(unsat)
                no_improve_count = 0
                if len(unsat) < best_unsat:
                    best_unsat = len(unsat)
                    best_assignment = assignment[:]
            else:
                no_improve_count += 1
            
            # Adaptive restart
            if no_improve_count > restart_after:
                # Restart with new random assignment
                assignment = [rng.random() < 0.5 for _ in range(n)]
                sat_count = [0] * m
                for ci, clause in enumerate(clauses):
                    cnt = 0
                    for lit in clause:
                        v = abs(lit) - 1
                        if assignment[v] == (lit > 0):
                            cnt += 1
                    sat_count[ci] = cnt
                unsat = set(ci for ci in range(m) if sat_count[ci] == 0)
                no_improve_count = 0
                last_best = len(unsat)
                step = 0
                if time.time() > deadline:
                    return best_assignment
                continue
            
            # Pick an unsatisfied clause
            clause_idx = rng.choice(list(unsat))
            clause = clauses[clause_idx]
            
            # Compute for each literal in the clause: how many clauses would become satisfied/unsatisfied
            # Greedy: pick literal that maximizes net gain
            best_gain = -999
            best_var = -1
            best_sign = 1  # sign of the flip: 1 means set to True, -1 means set to False
            
            for lit in clause:
                v = abs(lit) - 1
                # If we flip variable v, what happens?
                # Current value: assignment[v]
                # New value: not assignment[v]
                # For each clause containing v, check if satisfaction changes
                
                # The literal in this clause is currently false (since clause is unsat)
                # After flip, this literal becomes true -> clause becomes satisfied
                # So +1 for this clause
                
                # But other clauses containing v might change
                # If v is set to new_val, for each (ci2, sign2) in var_clauses[v]:
                #   old contribution: assignment[v] == (sign2 == 1)
                #   new contribution: (not assignment[v]) == (sign2 == 1)
                #   If old was True and new is False: clause ci2 loses one true literal
                #   If old was False and new is True: clause ci2 gains one true literal
                
                gain = 1  # this clause becomes satisfied
                
                new_val = not assignment[v]
                for ci2, sign2 in var_clauses[v]:
                    if ci2 == clause_idx:
                        continue
                    old_true = assignment[v] == (sign2 == 1)
                    new_true = new_val == (sign2 == 1)
                    if old_true and not new_true:
                        # This clause loses a true literal
                        if sat_count[ci2] == 1:
                            # It becomes unsatisfied
                            gain -= 1
                    elif not old_true and new_true:
                        # This clause gains a true literal
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
                    best_sign = 1 if (not assignment[v]) else -1
            
            # With some probability, pick a random literal instead (for exploration)
            # Adaptive: increase randomness when stuck
            rand_prob = min(0.5, 0.1 + 0.05 * (no_improve_count / restart_after))
            if rng.random() < rand_prob:
                lit = rng.choice(clause)
                best_var = abs(lit) - 1
                best_sign = 1 if (not assignment[best_var]) else -1
            else:
                if best_var == -1:
                    lit = rng.choice(clause)
                    best_var = abs(lit) - 1
                    best_sign = 1 if (not assignment[best_var]) else -1
            
            # Flip the variable
            v = best_var
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val
            
            # Update sat_count and unsat set
            for ci2, sign2 in var_clauses[v]:
                old_true = old_val == (sign2 == 1)
                new_true = new_val == (sign2 == 1)
                
                if old_true and not new_true:
                    sat_count[ci2] -= 1
                    if sat_count[ci2] == 0:
                        unsat.add(ci2)
                elif not old_true and new_true:
                    sat_count[ci2] += 1
                    if sat_count[ci2] == 1 and ci2 in unsat:
                        unsat.discard(ci2)
            
            step += 1
        
        return best_assignment
    
    # Try multiple restarts with different seeds
    best_result = None
    best_unsat_count = m + 1
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        result = try_solve(seed_offset=attempt)
        
        # Check how many clauses are satisfied
        unsat_count = 0
        for ci, clause in enumerate(clauses):
            if not any(result[abs(l) - 1] == (l > 0) for l in clause):
                unsat_count += 1
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_result = result
        
        if unsat_count == 0:
            return result
    
    if best_result is None:
        # Fallback: random assignment
        rng = random.Random(0)
        best_result = [rng.random() < 0.5 for _ in range(n)]
    
    return best_result
# EVOLVE-BLOCK-END
