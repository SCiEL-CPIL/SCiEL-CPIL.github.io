"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure for fast updates
    # For each clause, store (var_idx, is_positive) pairs
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    def init_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_unsat_count = [0] * m
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        return assignment, clause_unsat_count
    
    def count_unsat(clause_unsat_count):
        cnt = 0
        for c in clause_unsat_count:
            if c >= 3:
                cnt += 1
        return cnt
    
    def find_unsat_clause(clause_unsat_count):
        for ci in range(m):
            if clause_unsat_count[ci] >= 3:
                return ci
        return -1
    
    def flip_var(vi, assignment, clause_unsat_count):
        old_val = assignment[vi]
        assignment[vi] = not old_val
        for ci, pos in var_clauses[vi]:
            if old_val == pos:
                clause_unsat_count[ci] += 1
            else:
                clause_unsat_count[ci] -= 1
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(0)
    
    # Strategy: WalkSAT with adaptive weights (inspired by WalkSAT's p parameter)
    # Use a hybrid approach:
    # 1. First phase: aggressive search with random restarts and weighted clause selection
    # 2. Second phase: if stuck, use simulated annealing style temperature
    
    max_restarts = 100
    base_steps = 3000
    max_steps = 30000
    
    # Weight for random flips (WalkSAT parameter p)
    p_random = 0.2
    
    # Track recent unsat counts to detect stagnation
    recent_unsat = []
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment, clause_unsat_count = init_state(rng)
        current_unsat = count_unsat(clause_unsat_count)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = list(assignment)
        
        # Adaptive step budget based on restart number
        steps_in_restart = base_steps + restart * 1500
        steps_in_restart = min(steps_in_restart, max_steps)
        
        # Stagnation tracking
        local_best_unsat = current_unsat
        steps_since_improvement = 0
        max_stagnation = 2000
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            unsat_idx = find_unsat_clause(clause_unsat_count)
            if unsat_idx == -1:
                return assignment
            
            # With probability p_random, flip a random variable from the clause
            # Otherwise, use weighted flip selection
            if rng.random() < p_random:
                # Random flip from the unsat clause
                vi, pos = rng.choice(clause_vars[unsat_idx])
            else:
                # Weighted flip: minimize damage
                cv = clause_vars[unsat_idx]
                best_var = -1
                best_score = float('inf')
                
                for vi, pos in cv:
                    damage = 0
                    for cj, cpos in var_clauses[vi]:
                        if cj == unsat_idx:
                            continue
                        if clause_unsat_count[cj] < 3:
                            if assignment[vi] == cpos:
                                if clause_unsat_count[cj] >= 2:
                                    damage += 1
                    
                    # Add small random component for diversity
                    score = damage + rng.random() * 0.05
                    
                    if score < best_score:
                        best_score = score
                        best_var = vi
                
                if best_var == -1:
                    best_var = rng.choice(cv)[0]
                vi = best_var
            
            # Flip the variable
            flip_var(vi, assignment, clause_unsat_count)
            
            # Update unsat count
            current_unsat = count_unsat(clause_unsat_count)
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = list(assignment)
            
            # Stagnation detection
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            # If stuck for too long, break early and restart
            if steps_since_improvement > max_stagnation:
                break
        
        # Check time after restart
        if time.time() > deadline:
            break
    
    # If we haven't found a solution, try a second phase with different strategy
    # Phase 2: More aggressive randomization with simulated annealing
    if best_unsat_count > 0 and time.time() < deadline:
        phase2_start = time.time()
        phase2_deadline = deadline
        max_phase2_restarts = 30
        
        for restart in range(max_phase2_restarts):
            if time.time() > phase2_deadline:
                break
            
            assignment, clause_unsat_count = init_state(rng)
            current_unsat = count_unsat(clause_unsat_count)
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = list(assignment)
            
            # Simulated annealing style: temperature decreases over time
            initial_temp = 1.0
            cooling_rate = 0.9995
            temp = initial_temp
            
            steps_in_restart = 5000 + restart * 500
            steps_in_restart = min(steps_in_restart, 20000)
            
            for step in range(steps_in_restart):
                if time.time() > phase2_deadline:
                    break
                
                unsat_idx = find_unsat_clause(clause_unsat_count)
                if unsat_idx == -1:
                    return assignment
                
                cv = clause_vars[unsat_idx]
                
                # Compute scores for each variable in the clause
                scores = []
                for vi, pos in cv:
                    damage = 0
                    for cj, cpos in var_clauses[vi]:
                        if cj == unsat_idx:
                            continue
                        if clause_unsat_count[cj] < 3:
                            if assignment[vi] == cpos:
                                if clause_unsat_count[cj] >= 2:
                                    damage += 1
                    scores.append((vi, damage))
                
                # Boltzmann distribution for selection
                min_damage = min(d for _, d in scores)
                # Temperature controls randomness
                exp_scores = []
                for vi, d in scores:
                    # Lower damage = higher probability
                    exp_val = math.exp(-(d - min_damage) * temp)
                    exp_scores.append(exp_val)
                
                total = sum(exp_scores)
                if total == 0:
                    vi = rng.choice(scores)[0]
                else:
                    r = rng.random() * total
                    cumulative = 0
                    vi = scores[0][0]
                    for i, (v, d) in enumerate(scores):
                        cumulative += exp_scores[i]
                        if r <= cumulative:
                            vi = v
                            break
                
                flip_var(vi, assignment, clause_unsat_count)
                current_unsat = count_unsat(clause_unsat_count)
                
                if current_unsat == 0:
                    return assignment
                
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = list(assignment)
                
                # Cool down
                temp *= cooling_rate
        
        if time.time() > phase2_deadline:
            break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng2 = random.Random(42)
    return [rng2.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
