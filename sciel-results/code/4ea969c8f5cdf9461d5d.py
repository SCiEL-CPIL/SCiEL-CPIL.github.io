import time
import random
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin

    if m == 0:
        return [True] * n

    # Preprocessing
    # clauses: list of lists of signed 1-based literals
    # We need:
    # 1. var_occ: list of lists, var_occ[v] = list of (clause_idx, literal_sign)
    #    literal_sign is +1 if literal is v, -1 if literal is -v
    # 2. clause_vars: list of lists, clause_vars[c] = list of variable indices (0-based)
    # 3. clause_sats: list of int, count of true literals in clause
    
    var_occ = [[] for _ in range(n)]
    clause_vars = [0] * m
    clause_lits = [0] * m # Store actual literals for quick access
    
    for i, clause in enumerate(clauses):
        clause_lits[i] = clause
        vars_in_clause = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((i, sign))
            vars_in_clause.append(v)
        clause_vars[i] = vars_in_clause

    # Helper to check if a literal is satisfied given current assignment
    # assignment is a list of booleans
    def is_lit_true(lit, assignment):
        v = abs(lit) - 1
        if lit > 0:
            return assignment[v]
        else:
            return not assignment[v]

    def count_satisfied_clauses(assignment):
        count = 0
        for i in range(m):
            sat = False
            for lit in clause_lits[i]:
                if is_lit_true(lit, assignment):
                    sat = True
                    break
            if sat:
                count += 1
        return count

    # Main WalkSAT Logic
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat = m + 1
    
    # Parameters
    # For N=150, typical clause density is around 4-5.
    # WalkSAT parameters: p_greedy ~ 0.5-0.7, restarts ~ 100-1000
    
    max_restarts = 500
    max_steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random Initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial sat_counts
        # sat_counts[i] = number of true literals in clause i
        sat_counts = [0] * m
        for i in range(m):
            for lit in clause_lits[i]:
                if is_lit_true(lit, assignment):
                    sat_counts[i] += 1
        
        unsat_count = sum(1 for sc in sat_counts if sc == 0)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT Loop
        p_greedy = 0.5
        # Adaptive p_greedy: start higher, decrease if stuck? 
        # Standard is fixed. Let's stick to 0.5.
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = [i for i in range(m) if sat_counts[i] == 0]
            
            if not unsat_indices:
                return assignment
            
            if len(unsat_indices) < best_unsat:
                best_unsat = len(unsat_indices)
                best_assignment = assignment[:]
                
            # Pick a random unsatisfied clause
            c_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            clause_lits_c = clause_lits[c_idx]
            
            # Determine which variable to flip
            # Calculate gain for each variable in the clause
            # Gain = (# clauses that become satisfied) - (# clauses that become unsatisfied)
            
            best_gain = -100
            best_var = None
            gains = []
            
            for v in clause_vars[c_idx]:
                gain = 0
                for (c_prime, sign) in var_occ[v]:
                    # Check if literal (v with sign) is currently true
                    # lit_val = sign * assignment[v] ... wait
                    # If sign == 1, lit is v. True if assignment[v] is True.
                    # If sign == -1, lit is -v. True if assignment[v] is False.
                    
                    # Current state of this literal in clause c_prime:
                    if sign == 1:
                        is_true = assignment[v]
                    else:
                        is_true = not assignment[v]
                    
                    if is_true:
                        # Flipping v makes this literal False.
                        # Clause c_prime loses one true literal.
                        if sat_counts[c_prime] == 1:
                            # It becomes unsatisfied
                            gain -= 1
                    else:
                        # Flipping v makes this literal True.
                        # Clause c_prime gains one true literal.
                        if sat_counts[c_prime] == 0:
                            # It becomes satisfied
                            gain += 1
                gains.append(gain)
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
            
            # Choose variable
            if rng.random() < p_greedy:
                # Pick best gain
                # In case of tie, pick randomly among those with max gain
                max_g = best_gain
                candidates = [v for v, g in zip(clause_vars[c_idx], gains) if g == max_g]
                v_to_flip = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                # Pick random variable from the clause
                v_to_flip = clause_vars[c_idx][rng.randint(0, len(clause_vars[c_idx]) - 1)]
            
            # Flip v_to_flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update sat_counts
            for (c_prime, sign) in var_occ[v_to_flip]:
                # Determine if the literal in c_prime was true before flip
                # We flipped v_to_flip, so the truth value of the literal inverted.
                # If it was True, it's now False (sat_count decreases)
                # If it was False, it's now True (sat_count increases)
                
                # We need to know the state *before* the flip.
                # Current assignment[v_to_flip] is the NEW value.
                # Old value was not assignment[v_to_flip].
                
                # Let's re-evaluate based on NEW value to see if it's true now.
                # If it is true now, it was false before -> sat_count + 1
                # If it is false now, it was true before -> sat_count - 1
                
                if sign == 1:
                    is_true_now = assignment[v_to_flip]
                else:
                    is_true_now = not assignment[v_to_flip]
                
                if is_true_now:
                    # Was false, now true
                    sat_counts[c_prime] += 1
                else:
                    # Was true, now false
                    sat_counts[c_prime] -= 1
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [True] * n
