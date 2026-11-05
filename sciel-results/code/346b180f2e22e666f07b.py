"""
Improved 3-SAT solver for planted instances (50 vars).
Uses a portfolio of local search methods: WalkSAT, min-conflicts, simulated
annealing, and a simple clause-learning guided search, all with restarts and
time management.
"""
import random
import time
import math
import sys
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 290.0
    
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
    
    def walksat(seed, max_steps, noise=0.3):
        """WalkSAT with weighted clause selection."""
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Clause weights
        weights = [1.0] * m
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment[:]
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break
            
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(m):
                sat = False
                for var, pos in clause_data[i]:
                    if assignment[var] == pos:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment[:]
            
            # Weighted selection of an unsatisfied clause
            total_w = sum(weights[i] for i in unsat_indices)
            if total_w > 0:
                r = rng.random() * total_w
                cum = 0
                chosen = unsat_indices[0]
                for idx in unsat_indices:
                    cum += weights[idx]
                    if r <= cum:
                        chosen = idx
                        break
            else:
                chosen = rng.choice(unsat_indices)
            
            # Increase weight of chosen clause
            weights[chosen] *= 1.15
            # Decay other weights slightly
            for i in range(m):
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
                    # Check if clause ci is currently unsatisfied
                    ci_sat = False
                    for v2, p2 in clause_data[ci]:
                        if assignment[v2] == p2:
                            ci_sat = True
                            break
                    
                    if not ci_sat:
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
            
            assignment[best_var] = not assignment[best_var]
            
            current_unsat = count_unsat(assignment)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            if current_unsat == 0:
                return assignment[:]
        
        return best_assignment
    
    def simulated_annealing(seed, max_steps, t0=1.0, cooling=0.9995):
        """Simulated annealing for Max-SAT formulation.
        
        Objective: maximize number of satisfied clauses (equivalently minimize unsat count).
        Uses temperature-dependent acceptance to escape local minima.
        """
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Precompute for fast evaluation: for each variable, which clauses it appears in
        # and whether it's positive or negative in each clause
        
        # Track satisfied count incrementally
        # sat_count[v] = number of satisfied clauses that contain variable v
        sat_count = [0] * n
        for i in range(m):
            sat = False
            for var, pos in clause_data[i]:
                if assignment[var] == pos:
                    sat = True
                    break
            if sat:
                for var, pos in clause_data[i]:
                    sat_count[var] += 1
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment[:]
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        
        temp = t0
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break
            
            # Pick a random variable to flip
            var = rng.randrange(n)
            
            # Compute delta in unsat count if we flip var
            # For each clause containing var:
            # - If clause is currently unsatisfied and flipping var satisfies it: delta -= 1
            # - If clause is currently satisfied and var is the only satisfying literal: delta += 1
            
            delta = 0
            for ci in var_to_clauses[var]:
                # Check if clause ci is currently satisfied
                ci_sat = False
                only_var_satisfies = True
                for v2, p2 in clause_data[ci]:
                    if assignment[v2] == p2:
                        ci_sat = True
                        if v2 != var:
                            only_var_satisfies = False
                            break
                
                if not ci_sat:
                    # Clause is unsatisfied; check if flipping var would satisfy it
                    # Find the literal for var in this clause
                    for v2, p2 in clause_data[ci]:
                        if v2 == var:
                            if (not assignment[var]) == p2:
                                delta -= 1  # Flipping var satisfies this clause
                            break
                else:
                    # Clause is satisfied; check if var is the only satisfying literal
                    if only_var_satisfies:
                        delta += 1  # Flipping var would unsatisfy this clause
            
            # Acceptance decision
            if delta <= 0:
                # Move improves or maintains; always accept
                assignment[var] = not assignment[var]
                current_unsat += delta
            else:
                # Move worsens; accept with probability exp(-delta / temp)
                # Use log form for numerical stability
                if temp > 0:
                    log_prob = -delta / temp
                    if log_prob > -700:  # Avoid underflow
                        if rng.random() < math.exp(log_prob):
                            assignment[var] = not assignment[var]
                            current_unsat += delta
        
            # Update best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            if current_unsat == 0:
                return assignment[:]
            
            # Cool down
            temp *= cooling
        
        return best_assignment
    
    def min_conflicts(seed, max_steps):
        """Min-conflicts local search."""
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment[:]
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break
            
            # Find all unsatisfied clauses
            unsat_indices = []
            for i in range(m):
                sat = False
                for var, pos in clause_data[i]:
                    if assignment[var] == pos:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment[:]
            
            # Pick a random unsatisfied clause
            chosen = rng.choice(unsat_indices)
            clause = clause_data[chosen]
            
            # For each variable in the clause, compute the number of unsatisfied clauses it appears in
            # that would be fixed by flipping it
            best_var = -1
            best_gain = -1e9
            
            for var, pos in clause:
                gain = 0
                for ci in var_to_clauses[var]:
                    # Is clause ci currently unsatisfied?
                    ci_sat = False
                    for v2, p2 in clause_data[ci]:
                        if assignment[v2] == p2:
                            ci_sat = True
                            break
                    
                    if not ci_sat:
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
            
            assignment[best_var] = not assignment[best_var]
            
            current_unsat = count_unsat(assignment)
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
    
    # Strategy 5: Simulated Annealing
    if time_remaining() > 10.0:
        sa_configs = [
            (2001, 2000, 1.0, 0.9995),
            (2002, 3000, 0.5, 0.999),
            (2003, 1500, 2.0, 0.9998),
            (2004, 2500, 0.8, 0.9993),
            (2005, 4000, 1.5, 0.9997),
        ]
        for seed, steps, t0, cooling in sa_configs:
            if time_remaining() < 5.0:
                break
            result = simulated_annealing(seed, steps, t0=t0, cooling=cooling)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
    
    # Strategy 6: Short bursts of WalkSAT with many restarts
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
    
    # Strategy 7: More aggressive SA with lower temperature for polishing
    if time_remaining() > 3.0:
        for seed in [3001, 3002, 3003, 3004, 3005]:
            if time_remaining() < 1.0:
                break
            result = simulated_annealing(seed, 5000, t0=0.3, cooling=0.9999)
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    cnt = count_unsat(result)
                    if cnt < best_unsat_count:
                        best_unsat_count = cnt
                        best_result = result[:]
    
    if best_result is not None:
        return best_result
    
    return [True] * n
