"""Simulated Annealing based 3-SAT solver."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Pre-compute variable to clause indices
    var_to_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((c_idx, sign))

    # Heuristic initial assignment
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0:
                pos_count[var_idx] += 1
            else:
                neg_count[var_idx] += 1
                
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(1)
        elif neg_count[i] > pos_count[i]:
            assignment.append(0)
        else:
            assignment.append(rng.randint(0, 1))
            
    # Initialize clause satisfaction counts
    clause_sat_count = [0] * len(clauses)
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            if assignment[var_idx] == (1 if lit > 0 else 0):
                clause_sat_count[c_idx] += 1
                
    current_energy = sum(1 for c in clause_sat_count if c == 0)
    
    if current_energy == 0:
        return [bool(x) for x in assignment]

    T_start = 1.0
    T_end = 0.001
    max_iterations = 1000000
    restart_interval = 100000
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
    
    best_energy = current_energy
    best_assignment = assignment.copy()
    
    T = T_start
    steps_since_improvement = 0
    
    for iteration in range(max_iterations):
        if current_energy == 0:
            break
        
        if steps_since_improvement > restart_interval:
            for _ in range(n // 10):
                v = rng.randint(0, n - 1)
                assignment[v] = 1 - assignment[v]
                
            for c_idx in range(len(clauses)):
                clause_sat_count[c_idx] = 0
            for c_idx, clause in enumerate(clauses):
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if assignment[var_idx] == (1 if lit > 0 else 0):
                        clause_sat_count[c_idx] += 1
            current_energy = sum(1 for c in clause_sat_count if c == 0)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_assignment = assignment.copy()
            
            steps_since_improvement = 0
            T = T_start
            
        var = -1
        if current_energy > 0:
            for _ in range(10):
                c_idx = rng.randint(0, len(clauses) - 1)
                if clause_sat_count[c_idx] == 0:
                    clause = clauses[c_idx]
                    var = abs(clause[rng.randint(0, len(clause) - 1)]) - 1
                    break
        if var == -1 or rng.random() > 0.8:
            var = rng.randint(0, n - 1)
            
        old_val = assignment[var]
        new_val = 1 - old_val
        
        delta = 0
        updates = []
        
        for c_idx, sign in var_to_clauses[var]:
            old_count = clause_sat_count[c_idx]
            
            var_sat_now = (old_val == 1) if sign == 1 else (old_val == 0)
            var_sat_new = (new_val == 1) if sign == 1 else (new_val == 0)
            
            old_contrib = 1 if var_sat_now else 0
            new_contrib = 1 if var_sat_new else 0
            
            new_count = old_count - old_contrib + new_contrib
            
            if old_count == 0 and new_count > 0:
                delta -= 1
            elif old_count > 0 and new_count == 0:
                delta += 1
                
            updates.append((c_idx, new_count))
            
        if delta <= 0:
            assignment[var] = new_val
            current_energy += delta
            for c_idx, new_count in updates:
                clause_sat_count[c_idx] = new_count
                
            if current_energy < best_energy:
                best_energy = current_energy
                best_assignment = assignment.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            if T > 0:
                if rng.random() < math.exp(-delta / T):
                    assignment[var] = new_val
                    current_energy += delta
                    for c_idx, new_count in updates:
                        clause_sat_count[c_idx] = new_count
                    steps_since_improvement += 1
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1
                
        T *= cooling_rate

    if best_energy < current_energy:
        return [bool(x) for x in best_assignment]
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
