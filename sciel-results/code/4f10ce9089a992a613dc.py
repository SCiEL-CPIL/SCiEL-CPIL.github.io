# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Precompute clause data
    # clause_vars[c] = list of (var, sign) where sign is 1 for positive, -1 for negative
    # Or just keep the literals
    clause_lits = clauses
    
    # Bitmasks for variable occurrences
    pos_mask = [0] * n
    neg_mask = [0] * n
    
    for c, clause in enumerate(clauses):
        bit = 1 << c
        for lit in clause:
            var = abs(lit) - 1
            if lit > 0:
                pos_mask[var] |= bit
            else:
                neg_mask[var] |= bit
    
    def compute_sat_mask(assign):
        sat_mask = 0
        for i in range(n):
            if assign[i]:
                sat_mask |= pos_mask[i]
            else:
                sat_mask |= neg_mask[i]
        return sat_mask

    def count_bits(mask):
        return bin(mask).count('1')

    rng = random.Random(42)
    
    # Multiple restarts to escape local optima
    max_restarts = 10
    max_steps_per_restart = 20000
    
    for restart in range(max_restarts):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_sat_mask = compute_sat_mask(assignment)
        current_sat_count = count_bits(current_sat_mask)
        
        if current_sat_count == m:
            return assignment
        
        all_mask = (1 << m) - 1
        
        for step in range(max_steps_per_restart):
            if current_sat_count == m:
                return assignment
            
            # Pick a random unsatisfied clause
            # Rejection sampling: pick random index, check if unsat
            # To speed up, if sat_count is high, we might miss unsat clauses often.
            # But for 3-SAT, usually sat_count is not 100% until the end.
            # If we are stuck, we might want to pick from unsat list.
            # Let's try rejection first, it's O(1) avg.
            
            # Optimization: If we are very close to solution, rejection is fine.
            # If we are far, rejection is fine.
            # If we are stuck in a local optimum where many clauses are unsat, 
            # rejection still works.
            
            # However, if m is large, 1<<m is huge. 
            # Python handles big ints fine.
            
            c_idx = rng.randint(0, m - 1)
            
            # Check if c_idx is unsatisfied
            if current_sat_mask & (1 << c_idx):
                continue
            
            clause = clause_lits[c_idx]
            
            # Evaluate each literal in this clause as a candidate flip
            best_damage = float('inf')
            best_var = -1
            best_new_sat_mask = 0
            candidates = []
            
            for lit in clause:
                var = abs(lit) - 1
                
                # Determine current true literal mask for this var
                if assignment[var]:
                    # Current true literal is positive
                    # Clauses that might break are those in pos_mask[var]
                    # These clauses are currently satisfied by this literal.
                    # If we flip, they lose this satisfaction.
                    # They break if they don't have other true literals.
                    current_true_mask = pos_mask[var]
                else:
                    current_true_mask = neg_mask[var]
                
                # Calculate damage: number of clauses in current_true_mask that are ONLY satisfied by this literal
                damage = 0
                # Iterate through clauses in current_true_mask
                # We can do this by iterating bits
                temp_mask = current_true_mask
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    c_idx2 = lsb.bit_length() - 1
                    temp_mask ^= lsb
                    
                    # Check if clause c_idx2 is satisfied by any OTHER literal
                    # It is satisfied by current literal (by definition of being in current_true_mask)
                    # We need to check if it has another true literal.
                    # If it does, it won't break.
                    # If it doesn't, it breaks.
                    
                    # Optimization: If c_idx2 is the same as the clause we picked (c_idx), 
                    # it's unsatisfied, so it can't be in current_true_mask (which contains satisfied clauses).
                    # So we don't need to worry about c_idx being in damage.
                    
                    is_only_satisfied = True
                    for l2 in clause_lits[c_idx2]:
                        v2 = abs(l2) - 1
                        if v2 == var:
                            # This is the literal we are flipping. 
                            # We know it's true currently.
                            # We need to check if there are OTHER true literals.
                            continue
                        # Check if l2 is true
                        if l2 > 0:
                            if assignment[v2]:
                                is_only_satisfied = False
                                break
                        else:
                            if not assignment[v2]:
                                is_only_satisfied = False
                                break
                    
                    if is_only_satisfied:
                        damage += 1
                
                # Gain is always +1 (the chosen clause becomes satisfied)
                # Net change = 1 - damage
                # We want to maximize net change, so minimize damage.
                
                if damage < best_damage:
                    best_damage = damage
                    best_var = var
                    # We don't strictly need to compute best_new_sat_mask here if we just apply the flip
                    # and recompute, but recomputing is O(n).
                    # Let's just apply the flip and update mask incrementally?
                    # Incremental update:
                    # Remove clauses that break, add clauses that gain.
                    # Gains: c_idx is added.
                    # Are there other gains?
                    # Clauses in the *new* true mask that were previously unsat.
                    # New true mask:
                    if assignment[var]:
                        new_true_mask = neg_mask[var]
                    else:
                        new_true_mask = pos_mask[var]
                        
                    # Gains are clauses in new_true_mask that are NOT in current_sat_mask
                    # And are not in the "breaking" set (which is subset of current_true_mask)
                    # Actually, gains are simply: new_true_mask & ~current_sat_mask
                    # But we must be careful: a clause might be in both? No, disjoint vars?
                    # A clause can contain var. If it contains var, it's in both pos and neg? No.
                    # So new_true_mask and current_true_mask are disjoint sets of clauses?
                    # No, a clause contains only one literal of a variable.
                    # So pos_mask[var] and neg_mask[var] are disjoint.
                    # So gains = new_true_mask & ~current_sat_mask
                    # Breaks = (current_true_mask & current_sat_mask) that are not satisfied by others.
                    
                    # For now, let's just store the var and apply later.
                    pass
                elif damage == best_damage:
                    candidates.append(var)
            
            if best_var == -1:
                # Should not happen
                var = rng.choice([abs(l) - 1 for l in clause])
            elif candidates:
                var = rng.choice(candidates)
            else:
                var = best_var
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update sat_mask
            # Determine old and new true masks
            if assignment[var]:
                # Now True, so pos_mask is active, neg_mask was active before?
                # Wait, assignment[var] was flipped.
                # If it is now True, it was False.
                # Old active: neg_mask[var]
                # New active: pos_mask[var]
                old_active = neg_mask[var]
                new_active = pos_mask[var]
            else:
                # Now False, was True.
                # Old active: pos_mask[var]
                # New active: neg_mask[var]
                old_active = pos_mask[var]
                new_active = neg_mask[var]
            
            # Clauses to remove: Those in old_active that are not satisfied by new_active OR other vars.
            # This is complex to do incrementally without re-checking.
            # Given n=50, recomputing sat_mask is O(n) = 50 ops.
            # This is very fast.
            current_sat_mask = compute_sat_mask(assignment)
            current_sat_count = count_bits(current_sat_mask)
        
        # If we finish steps without solving, we restart (loop continues)
    
    # If all restarts fail, return last assignment
    return assignment
# EVOLVE-BLOCK-END
