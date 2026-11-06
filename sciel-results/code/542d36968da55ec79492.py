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
    max_restarts = 50
    max_steps_per_restart = 100000
    
    all_mask = (1 << m) - 1
    
    # Precompute for each clause, the list of (var, lit_sign) pairs
    # lit_sign: 1 if positive, -1 if negative
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((var, sign))
        clause_vars.append(cv)
    
    for restart in range(max_restarts):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
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
        
        # Track unsatisfied clauses for efficient selection
        unsat_clauses = []
        for c in range(m):
            if not (sat_mask & (1 << c)):
                unsat_clauses.append(c)
        
        best_sat_count = sat_count
        best_assignment = assignment[:]
        
        for step in range(max_steps_per_restart):
            if sat_count == m:
                return assignment
            
            # If no unsatisfied clauses, we're done
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            c_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            clause = clauses[c_idx]
            cv_list = clause_vars[c_idx]
            
            best_gain = -float('inf')
            best_vars = []
            
            for var, sign in cv_list:
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
                var = cv_list[0][0]
            else:
                var = best_vars[rng.randrange(len(best_vars))]
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update sat_mask incrementally
            affected_clauses = var_to_clauses[var]
            new_sat_mask = sat_mask
            
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
                    new_sat_mask |= (1 << c)
                else:
                    new_sat_mask &= ~(1 << c)
            
            # Update unsat_clauses list
            # Find clauses that changed status
            old_sat_mask = sat_mask
            sat_mask = new_sat_mask
            
            # Rebuild unsat_clauses efficiently
            # Clauses that were unsat and are now sat
            newly_sat = sat_mask & ~old_sat_mask
            newly_unsat = old_sat_mask & ~sat_mask
            
            # Update unsat_clauses
            unsat_set = set(unsat_clauses)
            
            # Remove newly satisfied
            temp = newly_sat
            while temp:
                lsb = temp & (-temp)
                c = lsb.bit_length() - 1
                temp ^= lsb
                unsat_set.discard(c)
            
            # Add newly unsatisfied
            temp = newly_unsat
            while temp:
                lsb = temp & (-temp)
                c = lsb.bit_length() - 1
                temp ^= lsb
                unsat_set.add(c)
            
            unsat_clauses = list(unsat_set)
            
            sat_count = count_bits(sat_mask)
            
            # Track best
            if sat_count > best_sat_count:
                best_sat_count = sat_count
                best_assignment = assignment[:]
        
        # After restart, if we found a better assignment, keep it
        if best_sat_count > sat_count:
            assignment = best_assignment
    
    # If all restarts fail, return best assignment found
    return assignment
# EVOLVE-BLOCK-END
