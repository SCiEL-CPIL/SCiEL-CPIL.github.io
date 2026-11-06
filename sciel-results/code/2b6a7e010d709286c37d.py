# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Precompute variable to clause mapping and literal signs
    var_to_clauses = [[] for _ in range(n)]
    # clause_var_info[c_idx] = list of (var_idx, sign)
    # var_sign_in_clause[c_idx][var_idx] = sign (for O(1) lookup)
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
    noise = 0.5  # Probability of picking a random variable in the clause
    restart_threshold = 5000
    TIME_CHECK_INTERVAL = 50  # Check time every 50 steps instead of every step
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial clause weights (number of satisfied literals)
        clause_weights = [0] * m
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
        
        total_unsat = sum(1 for w in clause_weights if w == 0)
        
        if total_unsat == 0:
            return assignment
        
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        
        for step in range(100000):
            # Time check every TIME_CHECK_INTERVAL steps (much cheaper than every step)
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
            
            # Find an unsatisfied clause using random probing (O(1) amortized)
            clause_idx = -1
            if total_unsat > 0:
                # Random probe: pick random clause indices until we find unsat
                # Expected probes = m / total_unsat
                for _ in range(min(m, 200)):
                    probe = rng.randrange(m)
                    if clause_weights[probe] == 0:
                        clause_idx = probe
                        break
                if clause_idx == -1:
                    # Fallback: linear scan (rare, when total_unsat is very small)
                    for i in range(m):
                        if clause_weights[i] == 0:
                            clause_idx = i
                            break
                    if clause_idx == -1:
                        return assignment
            
            clause_info = clause_var_info[clause_idx]
            
            # Heuristic: Pick variable that minimizes new unsat clauses
            if rng.random() < noise:
                # Random walk step
                v, s = rng.choice(clause_info)
                var_idx = v
            else:
                # Greedy step: Find variable in clause that gives best gain
                best_gain = float('inf')
                best_var_idx = -1
                
                for v, s in clause_info:
                    var_idx = v
                    current_val = assignment[var_idx]
                    
                    # Calculate delta in total_unsat
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_sign_in_clause[c_idx][var_idx]
                        
                        # Is the literal satisfied currently?
                        is_sat = (assignment[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1
                    
                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = var_idx
                        if best_gain == -1:
                            break
                            
                var_idx = best_var_idx
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause weights using precomputed signs
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
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    total_unsat += 1

    return best_assignment
# EVOLVE-BLOCK-END
