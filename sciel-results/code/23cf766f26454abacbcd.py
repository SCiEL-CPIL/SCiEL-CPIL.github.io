# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Use bitmasks for O(1) clause satisfaction counting
    # For each variable i, precompute:
    #   pos_mask[i]: bitmask of clauses where literal +i appears
    #   neg_mask[i]: bitmask of clauses where literal -i appears
    #   lit_count[i]: number of literals in clause (for 3-SAT, all are 3)
    
    pos_mask = [0] * n
    neg_mask = [0] * n
    
    # Build bitmasks using Python integers (arbitrary precision)
    # Each clause c (0-indexed) corresponds to bit c
    for c, clause in enumerate(clauses):
        bit = 1 << c
        for lit in clause:
            var = abs(lit) - 1
            if lit > 0:
                pos_mask[var] |= bit
            else:
                neg_mask[var] |= bit
    
    # Initial random assignment
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Compute initial satisfied clause count using bitmasks
    # A clause is satisfied if any of its literals is true
    # We can compute the set of satisfied clauses as the OR of all true literal masks
    
    def compute_satisfied_mask(assign):
        """Compute bitmask of satisfied clauses given assignment."""
        sat_mask = 0
        for i in range(n):
            if assign[i]:
                sat_mask |= pos_mask[i]
            else:
                sat_mask |= neg_mask[i]
        return sat_mask
    
    def count_satisfied(sat_mask):
        """Count number of set bits in sat_mask (satisfied clauses)."""
        return bin(sat_mask).count('1')
    
    # Precompute for each variable and each clause, whether flipping helps
    # But we'll do greedy: for each unsatisfied clause, try flipping each variable in it
    # and pick the flip that maximizes net gain
    
    # For efficiency, we track satisfied clauses as a bitmask
    current_sat_mask = compute_satisfied_mask(assignment)
    current_sat_count = count_satisfied(current_sat_mask)
    
    if current_sat_count == m:
        return assignment
    
    # Greedy local search with random tie-breaking
    max_iterations = 50000
    # Track best solution found
    best_assignment = assignment[:]
    best_count = current_sat_count
    
    for iteration in range(max_iterations):
        if current_sat_count == m:
            break
        
        # Find unsatisfied clauses
        all_mask = (1 << m) - 1
        unsat_mask = all_mask & ~current_sat_mask
        
        if unsat_mask == 0:
            break
        
        # Pick a random unsatisfied clause
        # Get list of unsatisfied clause indices
        unsat_indices = []
        temp_mask = unsat_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & (-temp_mask)
            idx = lsb.bit_length() - 1
            unsat_indices.append(idx)
            temp_mask ^= lsb
        
        if not unsat_indices:
            break
        
        clause_idx = rng.choice(unsat_indices)
        clause = clauses[clause_idx]
        
        # For each variable in this clause, compute the net gain if we flip it
        best_gain = -1
        best_var = -1
        candidates = []
        
        for lit in clause:
            var = abs(lit) - 1
            # If we flip var, which clauses change?
            # Currently satisfied clauses that contain lit and will become unsatisfied:
            #   These are clauses in (pos_mask[var] if lit>0 else neg_mask[var]) that are currently satisfied
            #   AND do not contain any other true literal (i.e., they would become unsat)
            # Actually, simpler: 
            #   Let current_true_mask_for_var = pos_mask[var] if assignment[var] else neg_mask[var]
            #   Let new_true_mask_for_var = neg_mask[var] if assignment[var] else pos_mask[var]
            #   Clauses that become unsatisfied: (current_true_mask_for_var & current_sat_mask) 
            #       but only those that don't have other true literals... 
            #   This is getting complex. Let's use a different approach.
            
            # Alternative: compute the effect directly
            # Clauses currently satisfied due to var's current value:
            if assignment[var]:
                current_contrib = pos_mask[var]
                new_contrib = neg_mask[var]
            else:
                current_contrib = neg_mask[var]
                new_contrib = pos_mask[var]
            
            # Clauses that are currently satisfied AND only satisfied by this variable's current literal:
            # A clause c is "only satisfied by var" if:
            #   - c is in current_contrib
            #   - c is in current_sat_mask
            #   - No other literal in c is true
            # 
            # To check "no other literal in c is true", we need to know the other literals in c.
            # This is expensive per clause. 
            #
            # Let's use a different strategy: for each candidate flip, compute the new sat mask
            # and count the difference. Since clauses are small (3 literals), we can afford
            # to check a few candidates.
            
            # Compute new_sat_mask if we flip var
            new_sat_mask = current_sat_mask
            # Remove clauses that were only satisfied by current_contrib
            # and add clauses satisfied by new_contrib
            
            # Clauses that will lose satisfaction:
            #   Clauses in current_contrib that are in current_sat_mask
            #   but NOT in new_contrib (if they have no other true literal)
            #   AND not in any other variable's true mask
            #
            # This is hard to compute without iterating. Let's just compute new_sat_mask
            # from scratch for the flipped assignment, but only for the affected clauses.
            #
            # Actually, for a 3-SAT with 50 vars and reasonable clause count, let's just
            # recompute the sat mask for the flipped variable's effect.
            
            # Clauses affected by flipping var:
            affected = current_contrib | new_contrib
            
            # For each clause in affected, determine if it's satisfied under new assignment
            new_sat_mask = current_sat_mask
            
            # First, mark all affected clauses as potentially changed
            # We'll recompute satisfaction for each affected clause
            
            # Get list of affected clause indices
            affected_indices = []
            temp = affected
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                affected_indices.append(idx)
                temp ^= lsb
            
            # For each affected clause, check if it's satisfied under new assignment
            for c_idx in affected_indices:
                c = clauses[c_idx]
                # Check if clause c is satisfied under new assignment
                # (var is flipped, others same)
                is_sat = False
                for l in c:
                    v = abs(l) - 1
                    if v == var:
                        # var is flipped
                        new_val = not assignment[var]
                        if (l > 0 and new_val) or (l < 0 and not new_val):
                            is_sat = True
                            break
                    else:
                        if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                            is_sat = True
                            break
                
                bit = 1 << c_idx
                if is_sat:
                    new_sat_mask |= bit
                else:
                    new_sat_mask &= ~bit
            
            new_count = count_satisfied(new_sat_mask)
            gain = new_count - current_sat_count
            
            if gain > best_gain:
                best_gain = gain
                best_var = var
                best_new_sat_mask = new_sat_mask
            elif gain == best_gain:
                candidates.append((var, new_sat_mask))
        
        if best_var == -1:
            # No flip improves; pick random flip from clause
            var = rng.choice([abs(l) - 1 for l in clause])
            assignment[var] = not assignment[var]
            current_sat_mask = compute_satisfied_mask(assignment)
            current_sat_count = count_satisfied(current_sat_mask)
        elif candidates:
            # Random tie-breaking among best gains
            var, new_sat_mask = rng.choice(candidates)
            assignment[var] = not assignment[var]
            current_sat_mask = new_sat_mask
            current_sat_count = count_satisfied(current_sat_mask)
        else:
            # Apply the best flip
            var = best_var
            assignment[var] = not assignment[var]
            current_sat_mask = best_new_sat_mask
            current_sat_count = count_satisfied(current_sat_mask)
        
        if current_sat_count > best_count:
            best_count = current_sat_count
            best_assignment = assignment[:]
        
        # If we've reached all clauses satisfied, break
        if current_sat_count == m:
            break
    
    # Return the best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
