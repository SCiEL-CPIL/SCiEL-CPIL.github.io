"""Planted 3-SAT solver: efficient WalkSAT with clause weighting, min-conflicts, and time management."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: return best result with safety margin before 330s
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Preprocess: convert clauses to 0-based, track literal signs
    # clause: list of (var_idx, is_positive)
    clauses_pp = []
    for c in clauses:
        pp_c = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pp_c.append((var_idx, is_pos))
        clauses_pp.append(pp_c)
    
    # Track variable occurrences for better heuristics
    var_count = [0] * n
    # Build clause index per variable for efficient impact calculation
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_pp):
        for v, _ in c:
            var_count[v] += 1
            var_to_clauses[v].append(ci)
    
    def is_satisfied(assignment, clause):
        for v, pos in clause:
            if assignment[v] == pos:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for c in clauses_pp:
            if not is_satisfied(assignment, c):
                cnt += 1
        return cnt
    
    def satisfied_clauses(assignment):
        """Return list of unsatisfied clause indices"""
        unsat = []
        for i, c in enumerate(clauses_pp):
            if not is_satisfied(assignment, c):
                unsat.append(i)
        return unsat
    
    # Try multiple restarts with different strategies
    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    rng = random.Random(42)
    
    max_restarts = 100
    steps_per_restart = 3000
    
    # Clause weights for weighted random selection
    weights = [1.0] * num_clauses
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        if restart == 0:
            # First restart: try a deterministic heuristic
            # Set each variable to satisfy the most clauses
            assignment = [False] * n
            for v in range(n):
                true_sat = 0
                false_sat = 0
                for c in clauses_pp:
                    v_pos = None
                    for vv, pos in c:
                        if vv == v:
                            v_pos = pos
                            break
                    if v_pos is not None:
                        if v_pos:
                            true_sat += 1
                        else:
                            false_sat += 1
                assignment[v] = true_sat >= false_sat
        else:
            # Subsequent restarts: random with some noise
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Decay weights on restart
        if restart > 0:
            for i in range(num_clauses):
                weights[i] = max(1.0, weights[i] * 0.85)
        
        # Local search with adaptive parameters
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            unsat_list = satisfied_clauses(assignment)
            if not unsat_list:
                # Found satisfying assignment
                return assignment
            
            # Update best
            if len(unsat_list) < best_unsat_count:
                best_unsat_count = len(unsat_list)
                best_assignment = assignment[:]
            
            # Choose an unsatisfied clause with weighted selection
            unsat_weights = [weights[i] for i in unsat_list]
            total_w = sum(unsat_weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_ci = unsat_list[0]
            for w, ci in zip(unsat_weights, unsat_list):
                cumulative += w
                if r <= cumulative:
                    chosen_ci = ci
                    break
            
            clause = clauses_pp[chosen_ci]
            
            # Update weight of this clause
            weights[chosen_ci] *= 1.05
            
            # Choose a literal to flip using min-conflicts with noise
            # Calculate the impact of flipping each variable in the clause
            best_var = -1
            best_gain = -float('inf')
            best_flip_pos = None
            
            for var, pos in clause:
                # If we flip this variable, what happens?
                current_val = assignment[var]
                # Flip means: new_val = not current_val
                # This literal is satisfied if new_val == pos
                # So flipping satisfies this literal if (not current_val) == pos
                
                # Count how many other unsat clauses this flip affects
                gain = 0
                for other_ci in var_to_clauses[var]:
                    other_clause = clauses_pp[other_ci]
                    # Check if this clause is currently unsatisfied
                    other_unsat = True
                    for v2, p2 in other_clause:
                        if assignment[v2] == p2:
                            other_unsat = False
                            break
                    
                    if not other_unsat:
                        # Clause is satisfied; check if flip would unsatisfy it
                        # Only if this variable is the ONLY satisfying literal
                        only_sat = True
                        for v2, p2 in other_clause:
                            if v2 == var:
                                continue
                            if assignment[v2] == p2:
                                only_sat = False
                                break
                        if only_sat:
                            gain -= 1
                    else:
                        # Clause is unsatisfied; check if flip would satisfy it
                        # This happens if the flipped variable's literal becomes true
                        # i.e., (not current_val) == p2 where p2 is the sign of var in this clause
                        var_sign_in_other = None
                        for v2, p2 in other_clause:
                            if v2 == var:
                                var_sign_in_other = p2
                                break
                        if var_sign_in_other is not None and (not current_val) == var_sign_in_other:
                            gain += 1
                
                # Add small noise to avoid getting stuck
                noise = rng.gauss(0, 0.1)
                total_gain = gain + noise
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_var = var
                    best_flip_pos = pos
            
            if best_var != -1:
                assignment[best_var] = not assignment[best_var]
            else:
                # Fallback: random flip
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
        
        # If we've exhausted this restart, check if we should continue
        if best_unsat_count == 0:
            break
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
