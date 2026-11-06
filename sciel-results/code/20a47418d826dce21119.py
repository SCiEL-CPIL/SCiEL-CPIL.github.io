# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Precompute clause data
    pos_mask = [0] * n
    neg_mask = [0] * n
    
    # Mapping from variable to list of clause indices it appears in
    var_to_clauses = [[] for _ in range(n)]
    
    for c, clause in enumerate(clauses):
        bit = 1 << c
        for lit in clause:
            var = abs(lit) - 1
            if lit > 0:
                pos_mask[var] |= bit
            else:
                neg_mask[var] |= bit
            var_to_clauses[var].append(c)

    def count_bits(mask):
        return mask.bit_count()

    rng = random.Random(42)
    
    # Multiple restarts to escape local optima
    max_restarts = 30
    max_steps_per_restart = 80000
    
    all_mask = (1 << m) - 1
    
    for restart in range(max_restarts):
        # Random initial assignment with varying bias for diversity
        bias = 0.5 + 0.3 * (restart % 7) / 6.0
        assignment = [rng.random() < bias for _ in range(n)]
        
        # Initialize sat_mask
        sat_mask = 0
        for i in range(n):
            if assignment[i]:
                sat_mask |= pos_mask[i]
            else:
                sat_mask |= neg_mask[i]
                
        sat_count = count_bits(sat_mask)
        
        if sat_count == m:
            return assignment
        
        # Track progress
        last_best_sat = sat_count
        no_improve_steps = 0
        
        for step in range(max_steps_per_restart):
            if sat_count == m:
                return assignment
            
            # Build list of unsatisfied clauses if needed
            # Use rejection sampling with early bailout
            c_idx = rng.randint(0, m - 1)
            
            # Check if c_idx is unsatisfied
            if sat_mask & (1 << c_idx):
                # Try a few more times, then break to restart
                continue
            
            clause = clauses[c_idx]
            
            best_gain = -float('inf')
            best_vars = []
            
            for lit in clause:
                var = abs(lit) - 1
                
                # Calculate net gain of flipping var
                old_val = assignment[var]
                
                if old_val:
                    old_mask = pos_mask[var]
                    new_mask = neg_mask[var]
                else:
                    old_mask = neg_mask[var]
                    new_mask = pos_mask[var]
                
                # Gain: clauses satisfied by new value that are currently unsat
                gain_mask = new_mask & ~sat_mask
                gain = count_bits(gain_mask)
                
                # Loss: clauses satisfied by old value that are currently sat 
                # and will become unsat
                loss_candidates_mask = old_mask & sat_mask
                loss = 0
                
                # Iterate through loss candidates
                temp_mask = loss_candidates_mask
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    c_idx2 = lsb.bit_length() - 1
                    temp_mask ^= lsb
                    
                    # Check if clause c_idx2 is satisfied by any other literal
                    clause2 = clauses[c_idx2]
                    is_only_satisfied = True
                    for l2 in clause2:
                        v2 = abs(l2) - 1
                        if v2 == var:
                            continue
                        # Check if l2 is true in current assignment
                        if l2 > 0:
                            if assignment[v2]:
                                is_only_satisfied = False
                                break
                        else:
                            if not assignment[v2]:
                                is_only_satisfied = False
                                break
                    
                    if is_only_satisfied:
                        loss += 1
                
                net_gain = gain - loss
                
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_vars = [var]
                elif net_gain == best_gain:
                    best_vars.append(var)
            
            if not best_vars:
                # Should not happen if clause is not empty
                var = rng.choice([abs(l) - 1 for l in clause])
            else:
                var = rng.choice(best_vars)
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update sat_mask incrementally
            if assignment[var]:
                # Now True
                affected_clauses = var_to_clauses[var]
                for c in affected_clauses:
                    # Check if clause c is satisfied
                    is_sat = False
                    for l in clauses[c]:
                        v = abs(l) - 1
                        if l > 0:
                            if assignment[v]:
                                is_sat = True
                                break
                        else:
                            if not assignment[v]:
                                is_sat = True
                                break
                    if is_sat:
                        sat_mask |= (1 << c)
                    else:
                        sat_mask &= ~(1 << c)
            else:
                # Now False
                affected_clauses = var_to_clauses[var]
                for c in affected_clauses:
                    is_sat = False
                    for l in clauses[c]:
                        v = abs(l) - 1
                        if l > 0:
                            if assignment[v]:
                                is_sat = True
                                break
                        else:
                            if not assignment[v]:
                                is_sat = True
                                break
                    if is_sat:
                        sat_mask |= (1 << c)
                    else:
                        sat_mask &= ~(1 << c)
            
            sat_count = count_bits(sat_mask)
            
            # Track improvement
            if sat_count > last_best_sat:
                last_best_sat = sat_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Restart if stuck for too long
            if no_improve_steps > 3000:
                break
            
        # If we finish steps without solving, restart (loop continues)
    
    # If all restarts fail, return last assignment
    return assignment
# EVOLVE-BLOCK-END
