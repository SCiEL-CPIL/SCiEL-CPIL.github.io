import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute clause literals as 0-indexed variables and signs
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
    deadline = time.time() + 270
    
    # Weights for clauses and variables
    clause_weights = [1.0] * len(clauses)
    var_weights = [1.0] * n
    
    # Phase saving
    phase = [False] * n
    
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT parameters
    base_noise = 0.1
    max_noise = 0.4
    noise_step = 0.02
    steps_per_restart = 15000
    max_restarts = 500
    polish_threshold = 10  # When unsat <= this, enter polish mode
    polish_steps = 50000
    polish_noise = 0.01
    
    def is_clause_satisfied_fast(clause, assignment):
        for v, sign in clause:
            if sign == 1:
                if assignment[v]:
                    return True
            else:
                if not assignment[v]:
                    return True
        return False
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment with phase saving
        assignment = [phase[i] if rng.random() < 0.5 else rng.random() < 0.5 for i in range(n)]
        
        # Calculate initial unsat
        unsat_indices = set()
        for i, c in enumerate(parsed_clauses):
            if not is_clause_satisfied_fast(c, assignment):
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
        polish_mode = False
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return assignment
            
            # Switch to polish mode if close to solution
            if not polish_mode and cur_unsat <= polish_threshold:
                polish_mode = True
                noise = polish_noise
                # Extend steps for polish
                steps_per_restart = step + polish_steps
                last_unsat = cur_unsat
                stall_count = 0
            
            # Pick a random unsatisfied clause
            c_idx = rng.choice(list(unsat_indices))
            clause = parsed_clauses[c_idx]
            
            # Get unsatisfied literals in this clause
            unsat_lits_in_clause = []
            for v, s in clause:
                if s == 1:
                    if not assignment[v]:
                        unsat_lits_in_clause.append((v, s))
                else:
                    if assignment[v]:
                        unsat_lits_in_clause.append((v, s))
            
            target_var = None
            
            if len(unsat_lits_in_clause) == 1:
                # Unit clause: force flip
                target_var = unsat_lits_in_clause[0][0]
            else:
                # Score each variable: minimize broken, maximize gained
                best_score = float('inf')
                best_var = None
                
                for v, s in unsat_lits_in_clause:
                    broken = 0
                    gained = 0
                    
                    for other_c_idx in var_to_clauses[v]:
                        if other_c_idx == c_idx:
                            continue
                        
                        other_clause = parsed_clauses[other_c_idx]
                        
                        if other_c_idx in unsat_indices:
                            # Unsat clause: flipping v will satisfy it (gain)
                            # because the literal for v in this unsat clause is currently false,
                            # and flipping v makes it true
                            gained += 1
                        else:
                            # Satisfied clause: check if it breaks
                            has_other_true = False
                            for ov, os in other_clause:
                                if ov == v:
                                    continue
                                if os == 1:
                                    if assignment[ov]:
                                        has_other_true = True
                                        break
                                else:
                                    if not assignment[ov]:
                                        has_other_true = True
                                        break
                            if not has_other_true:
                                broken += 1
                    
                    # Smooth score: use a soft combination
                    score = broken - 0.3 * gained
                    
                    if score < best_score:
                        best_score = score
                        best_var = v
                
                # Noise-based random selection
                if rng.random() < noise:
                    v, s = rng.choice(unsat_lits_in_clause)
                    target_var = v
                else:
                    target_var = best_var
            
            if target_var is None:
                continue
            
            # Perform flip
            assignment[target_var] = not assignment[target_var]
            phase[target_var] = assignment[target_var]
            
            # Update unsat_indices for affected clauses
            for c_idx2 in var_to_clauses[target_var]:
                c = parsed_clauses[c_idx2]
                sat = is_clause_satisfied_fast(c, assignment)
                if sat:
                    if c_idx2 in unsat_indices:
                        unsat_indices.remove(c_idx2)
                else:
                    if c_idx2 not in unsat_indices:
                        unsat_indices.add(c_idx2)
            
            cur_unsat = len(unsat_indices)
            
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
            
            # Update weights periodically
            if step % 500 == 0:
                for c_idx2 in unsat_indices:
                    clause_weights[c_idx2] += 0.05
                    for v, s in parsed_clauses[c_idx2]:
                        var_weights[v] += 0.05
                # Decay
                for i in range(len(clauses)):
                    clause_weights[i] *= 0.95
                for i in range(n):
                    var_weights[i] *= 0.95
            
            # Stall detection and noise adjustment
            if cur_unsat < last_unsat:
                last_unsat = cur_unsat
                stall_count = 0
                if not polish_mode:
                    noise = max(base_noise, noise - noise_step)
            else:
                stall_count += 1
                if stall_count > 50 and not polish_mode:
                    noise = min(max_noise, noise + noise_step)
        
        # End local search loop
        if time.time() > deadline:
            break
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
