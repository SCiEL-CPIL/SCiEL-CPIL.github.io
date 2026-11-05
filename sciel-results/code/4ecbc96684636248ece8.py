import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute clause literals as 0-indexed variables and signs
    # lit: (var_idx, is_positive)
    # Store as list of (var, sign) where sign is 1 for pos, -1 for neg
    parsed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            pc.append((v, sign))
        parsed_clauses.append(pc)
    
    # Mapping variable to clauses it appears in
    var_to_clauses = [[] for _ in range(n)]
    for i, pc in enumerate(parsed_clauses):
        for v, _ in pc:
            var_to_clauses[v].append(i)

    rng = random.Random(42)
    deadline = time.time() + 290  # Safety margin for 300s limit
    
    # Weights for clauses and variables
    clause_weights = [1.0] * len(clauses)
    var_weights = [1.0] * n
    
    # Phase saving: remember last value that satisfied a clause
    phase = [False] * n
    
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT parameters
    base_noise = 0.05
    max_noise = 0.5
    noise_step = 0.02
    
    # Luby sequence for restart lengths
    def luby_sequence():
        i = 1
        while True:
            # Luby sequence: 1, 1, 2, 1, 1, 2, 4, ...
            # Simplified: generate blocks of size 2^k
            k = 0
            while (1 << k) < i:
                k += 1
            yield (1 << k) * 1000 # Scale up
            i += 1
            # Reset i for next block? No, Luby is infinite.
            # Actually, let's just use a simple multiplier.
            # A common approach is to double the restart length every few restarts.
            break # This generator is flawed for infinite use in this context. 
            # Let's use a simpler adaptive restart.

    def get_restart_length(restart_idx):
        # Luby-like: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8...
        # Simplified: Double every 3 restarts? Or use a random distribution.
        # Let's use a base length that increases slowly.
        return 1000 + (restart_idx % 10) * 100

    def is_lit_true(v, sign, assignment):
        if sign == 1:
            return assignment[v]
        else:
            return not assignment[v]

    def is_clause_satisfied(clause, assignment):
        for v, sign in clause:
            if is_lit_true(v, sign, assignment):
                return True
        return False

    def count_false_literals(clause, assignment):
        count = 0
        for v, sign in clause:
            if not is_lit_true(v, sign, assignment):
                count += 1
        return count

    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        # Use phase saving if available, else random
        assignment = [phase[i] if rng.random() < 0.5 else rng.random() < 0.5 for i in range(n)]
        
        # Greedy initialization: try to satisfy more clauses initially
        # This is a simple heuristic to start closer to a solution
        for _ in range(10): # A few passes
            improved = False
            for i, c in enumerate(parsed_clauses):
                if is_clause_satisfied(c, assignment):
                    continue
                # Try flipping each variable in the clause
                best_gain = -1
                best_var = None
                for v, s in c:
                    # Check if flipping v satisfies c and doesn't break too much
                    # Simplified: just check if it satisfies c
                    # We'll rely on local search for the heavy lifting
                    pass 
            # Skip complex greedy init to keep overhead low, rely on restarts

        # Calculate initial unsat
        unsat_indices = set()
        for i, c in enumerate(parsed_clauses):
            if not is_clause_satisfied(c, assignment):
                unsat_indices.add(i)
        
        cur_unsat = len(unsat_indices)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment
        
        noise = base_noise
        stall_count = 0
        last_unsat = cur_unsat
        
        # Adaptive restart length
        steps_per_restart = get_restart_length(restart)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return assignment
            
            # Refined Clause Selection:
            # Pick the unsatisfied clause with the minimum number of false literals.
            # If tie, pick the one with the highest weight.
            # This focuses on clauses that are "closest" to being satisfied.
            
            # Optimization: If the number of unsat clauses is small, we can find the min directly.
            # If large, sampling might be better, but for n=50, m~? 
            # Let's assume m is moderate. Iterating over all unsat clauses is O(m).
            # m is typically ~ 4.25 * n = 212 for 3-SAT. This is fast.
            
            min_false = float('inf')
            best_c_idx = None
            best_weight = -1
            
            for c_idx in unsat_indices:
                c = parsed_clauses[c_idx]
                # Count false literals
                false_count = 0
                for v, s in c:
                    if not is_lit_true(v, s, assignment):
                        false_count += 1
                        if false_count > min_false:
                            break
                            
                if false_count < min_false or (false_count == min_false and clause_weights[c_idx] > best_weight):
                    min_false = false_count
                    best_c_idx = c_idx
                    best_weight = clause_weights[c_idx]
                    
            c_idx = best_c_idx
            clause = parsed_clauses[c_idx]
            
            # Unit Clause Heuristic
            unsat_lits_in_clause = [(v, s) for v, s in clause if not is_lit_true(v, s, assignment)]
            
            target_var = None
            if len(unsat_lits_in_clause) == 1:
                # Force flip this variable to satisfy this clause
                target_var = unsat_lits_in_clause[0][0]
            else:
                # Standard WalkSAT: pick variable from this clause
                # Score each variable in the clause
                best_score = float('inf')
                best_var = None
                best_gained = -1
                best_weight = 0
                
                for v, s in unsat_lits_in_clause:
                    # Calculate gain/loss if we flip v
                    broken = 0
                    gained = 0
                    
                    for other_c_idx in var_to_clauses[v]:
                        if other_c_idx == c_idx:
                            continue
                        
                        other_clause = parsed_clauses[other_c_idx]
                        
                        if other_c_idx in unsat_indices:
                            # It is unsat. Flipping v satisfies the literal for v.
                            # Since the clause was unsat, making one literal true satisfies it.
                            gained += 1
                        else:
                            # It is satisfied. Check if it remains satisfied.
                            has_other_true = False
                            for ov, os in other_clause:
                                if ov == v:
                                    continue
                                if is_lit_true(ov, os, assignment):
                                    has_other_true = True
                                    break
                                    
                            if not has_other_true:
                                # v was the only true literal
                                broken += 1
                            
                    # Score: minimize broken. If tie, maximize gained.
                    # If still tie, use weight.
                    if broken < best_score or (broken == best_score and (gained > best_gained or (gained == best_gained and clause_weights[c_idx] * var_weights[v] > best_weight))):
                        best_score = broken
                        best_var = v
                        best_gained = gained
                        best_weight = clause_weights[c_idx] * var_weights[v]
                
                # Noise
                if rng.random() < noise:
                    # Pick random var from unsat literals in clause
                    v, s = rng.choice(unsat_lits_in_clause)
                    target_var = v
                else:
                    target_var = best_var
            
            if target_var is None:
                continue
            
            # Perform flip
            assignment[target_var] = not assignment[target_var]
            phase[target_var] = assignment[target_var]
            
            # Update unsat_indices
            for c_idx in var_to_clauses[target_var]:
                c = parsed_clauses[c_idx]
                sat = is_clause_satisfied(c, assignment)
                if sat:
                    if c_idx in unsat_indices:
                        unsat_indices.remove(c_idx)
                else:
                    if c_idx not in unsat_indices:
                        unsat_indices.add(c_idx)
            
            # Update weights
            # Increase weight of unsat clauses
            for c_idx in unsat_indices:
                clause_weights[c_idx] += 0.1
            # Increase weight of vars in unsat clauses
            for c_idx in unsat_indices:
                for v, s in parsed_clauses[c_idx]:
                    var_weights[v] += 0.1
            
            # Decay weights periodically to prevent overflow
            if step % 1000 == 0:
                for i in range(len(clauses)):
                    clause_weights[i] *= 0.9
                for i in range(n):
                    var_weights[i] *= 0.9
            
            # Stall detection
            if len(unsat_indices) < last_unsat:
                last_unsat = len(unsat_indices)
                stall_count = 0
                noise = max(base_noise, noise - noise_step)
            else:
                stall_count += 1
                if stall_count > 100:
                    noise = min(max_noise, noise + noise_step)
                
        # End local search loop
        if time.time() > deadline:
            break

    if best_assignment:
        return best_assignment
    return [False] * n
