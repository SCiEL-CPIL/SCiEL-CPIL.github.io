import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 325.0

    # Pre-compute clause structure
    # clause_vars: (m, 3) int32, 0-based var indices
    # clause_polarity: (m, 3) int8, 1 or -1
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_polarity = np.zeros((m, 3), dtype=np.int8)
    
    # Variable to clause index lists
    var_clauses = [[] for _ in range(n)]
    
    for ci, c in enumerate(clauses):
        for li, l in enumerate(c):
            vi = abs(l) - 1
            clause_vars[ci, li] = vi
            clause_polarity[ci, li] = 1 if l > 0 else -1
            var_clauses[vi].append(ci)
    
    # Convert var_clauses to numpy arrays for fast access
    var_clause_indices = []
    for vi in range(n):
        if var_clauses[vi]:
            var_clause_indices.append(np.array(var_clauses[vi], dtype=np.int32))
        else:
            var_clause_indices.append(np.array([], dtype=np.int32))

    def init_state(rng):
        # Random assignment: 0 or 1
        assignment = rng.integers(2, size=n, dtype=np.int8)
        
        # Compute lit values: 1 if (assignment == 1 and polarity == 1) or (assignment == 0 and polarity == -1)
        # lit_val = 1 if satisfied, 0 if unsatisfied?
        # Let's use: sat = 1 if (2*assignment - 1) * polarity > 0 else 0
        lit_vals = (2 * assignment.astype(np.int8) - 1) # -1 or 1
        clause_lit_vals = lit_vals[clause_vars] # (m, 3)
        clause_products = clause_lit_vals * clause_polarity # (m, 3)
        
        # unsat_count: number of literals in clause that are FALSE
        # A literal is false if product < 0
        clause_unsat_count = (clause_products < 0).sum(axis=1).astype(np.int8)
        
        # Frustration: initialized to 0
        frustration = np.zeros(m, dtype=np.int32)
        
        return assignment, clause_unsat_count, frustration

    def flip_var(vi, assignment, clause_unsat_count, frustration, rng):
        """
        Flip variable vi.
        Returns the list of clause indices affected.
        """
        old_val = assignment[vi]
        assignment[vi] = 1 - old_val
        
        cidx = var_clause_indices[vi]
        if len(cidx) == 0:
            return []
        
        # Determine which clauses change status
        # old_lit_val = 2*old_val - 1
        # new_lit_val = -old_lit_val
        # For each clause in cidx, check polarity
        cpol = clause_polarity[cidx] # shape (k, 3) -- wait, clause_polarity is (m,3), cidx is (k,), so cpol is (k,3)
        
        # We need to know the position of vi in each clause to get the correct polarity.
        # This is inefficient. Let's store polarity per variable in the clause.
        
        # Better approach for flip:
        # We know the old value. We know the new value.
        # For each clause containing vi, we can recalculate the contribution of vi.
        # However, we don't have the specific polarity of vi in each clause readily available in a vectorized way without searching.
        
        # Let's pre-compute: var_clause_polarities[vi] = array of polarities of vi in each clause containing vi.
        # And var_clause_positions? No, we just need the polarity.
        
        # Let's re-preprocess to store polarities per variable.
        # This requires a second pass or storing during the first pass.
        
        # Since we can't easily restructure the pre-processing inside solve without slowing it down significantly, 
        # let's just compute the gain/impact by checking the specific literal.
        
        # Actually, the current code in the prompt does this:
        # var_clause_polarities[vi] = array of polarities.
        # Let's add that to the pre-processing step in this new version.
        
        pass # Placeholder, will implement correctly below

    # Re-do pre-processing to include polarities per variable
    var_clause_polarities = []
    for vi in range(n):
        if var_clauses[vi]:
            pols = []
            for ci in var_clauses[vi]:
                # Find polarity of vi in clause ci
                # clause_vars[ci] contains vi
                # Find index li where clause_vars[ci, li] == vi
                # This is O(3) per clause, so O(3) total per variable. Fast.
                for li in range(3):
                    if clause_vars[ci, li] == vi:
                        pols.append(clause_polarity[ci, li])
                        break
            var_clause_polarities.append(np.array(pols, dtype=np.int8))
        else:
            var_clause_polarities.append(np.array([], dtype=np.int8))

    def flip_var(vi, assignment, clause_unsat_count, frustration):
        old_val = assignment[vi]
        new_val = 1 - old_val
        assignment[vi] = new_val
        
        cidx = var_clause_indices[vi]
        cpol = var_clause_polarities[vi]
        
        if len(cidx) == 0:
            return
        
        old_lit_val = 2 * old_val - 1
        new_lit_val = -old_lit_val
        
        # Was the literal satisfied?
        # sat = (lit_val * polarity) > 0
        was_sat = (old_lit_val * cpol) > 0
        is_sat = (new_lit_val * cpol) > 0
        
        # Clauses where the literal changed from False to True (satisfied)
        # These clauses decrease their unsat count by 1
        # Clauses where the literal changed from True to False (unsatisfied)
        # These clauses increase their unsat count by 1
        
        # Update unsat counts
        # was_sat is True if old was satisfied. 
        # If was_sat and not is_sat: unsat count increases by 1
        # If not was_sat and is_sat: unsat count decreases by 1
        
        # Since we flip, was_sat and is_sat are always different.
        # So:
        # If was_sat: it became unsat. unsat_count += 1
        # If not was_sat: it became sat. unsat_count -= 1
        
        inc_mask = was_sat
        dec_mask = ~was_sat
        
        clause_unsat_count[cidx[inc_mask]] += 1
        clause_unsat_count[cidx[dec_mask]] -= 1
        
        # Update frustration
        # If a clause is now unsat (unsat_count == 3), frustration increases?
        # Or frustration is a timer: if unsat, +=1, else 0.
        # We only update frustration for affected clauses.
        
        # For affected clauses, check new unsat_count.
        # If unsat_count == 3, frustration += 1 (or set to high value?)
        # Standard WalkSAT frustration: if clause is unsat, weight/frustration increases.
        # Let's do: frustration[c] = frustration[c] + 1 if unsat_count[c] == 3 else 0
        
        new_counts = clause_unsat_count[cidx]
        unsat_mask = (new_counts == 3)
        
        # Reset frustration for all affected clauses to 0 first?
        # If a clause becomes sat, frustration should reset.
        # If a clause remains unsat, frustration should increase.
        
        # We can't easily "increase" only the ones that were already unsat without knowing their previous state.
        # But we know:
        # 1. Clauses that became sat (dec_mask): frustration should be 0.
        # 2. Clauses that became unsat (inc_mask): 
        #    - If they were already unsat (count was 3 before, now 4? No, max is 3 literals. If count was 2, now 3. If count was 3, impossible to become 4 in 3-SAT unless we count differently. 
        #    Wait, unsat_count is number of FALSE literals. Max is 3.
        #    If a literal flips from False to True: count decreases.
        #    If a literal flips from True to False: count increases.
        
        # Let's just reset frustration for ALL affected clauses to 0, and then add 1 to those that are NOW unsat?
        # This loses the history of how long it's been unsat.
        
        # Better:
        # For dec_mask (became sat): frustration = 0
        # For inc_mask (became unsat):
        #    We need to know if it was unsat before.
        #    old_count = clause_unsat_count[cidx[inc_mask]] - 1 (since we just incremented)
        #    If old_count == 3, it was already unsat. frustration += 1
        #    If old_count < 3, it was not unsat. frustration = 1
        
        # This is getting complex to vectorize efficiently.
        # Alternative: Use a simpler weight scheme.
        # weight[c] = weight[c] * 0.99 (decay)
        # weight[c_selected] += 1.0
        # This is easy to vectorize:
        # weights *= 0.99
        # weights[unsat_indices] += 1.0  # Or just selected clause?
        
        # Let's stick to the decay + boost scheme. It's robust.
        pass

    # Let's implement the decay + boost scheme.
    # It requires maintaining a `clause_weights` array.
    
    def update_weights(clause_weights, unsat_indices, selected_idx, rng):
        # Decay all weights
        clause_weights *= 0.95
        # Boost unsatisfied clauses? Or just the selected one?
        # Boosting all unsatisfied clauses makes the search focus on the "hardest" part of the formula.
        clause_weights[unsat_indices] += 1.0
        # Extra boost to selected clause?
        # clause_weights[selected_idx] += 2.0
        pass

    # Main Search
    rng = np.random.default_rng(42)
    best_assignment = None
    best_unsat = m + 1
    
    max_restarts = 1000
    base_steps = 10000
    max_steps = 100000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment, clause_unsat_count, _ = init_state(rng)
        clause_weights = np.ones(m, dtype=np.float64)
        
        current_unsat = int((clause_unsat_count == 3).sum())
        
        if current_unsat == 0:
            return assignment.tolist()
            
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        steps_in_restart = base_steps + restart * 1000
        steps_in_restart = min(steps_in_restart, max_steps)
        
        local_best = current_unsat
        steps_since_improve = 0
        walk_prob = 0.2
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
                
            unsat_mask = clause_unsat_count == 3
            if not np.any(unsat_mask):
                return assignment.tolist()
                
            unsat_indices = np.where(unsat_mask)[0]
            
            # Update weights
            clause_weights *= 0.95
            clause_weights[unsat_indices] += 1.0
            
            # Select clause
            # Normalize weights for selection
            w = clause_weights[unsat_indices]
            total_w = w.sum()
            if total_w == 0:
                probs = np.ones(len(unsat_indices)) / len(unsat_indices)
            else:
                probs = w / total_w
                
            sel_idx = int(rng.choice(len(unsat_indices), p=probs))
            c_idx = int(unsat_indices[sel_idx])
            
            # Select variable
            cv = clause_vars[c_idx]
            
            if rng.random() < walk_prob:
                vi = int(rng.choice(cv))
            else:
                # Greedy: find variable with max gain
                best_gain = -10
                best_vi = -1
                for li in range(3):
                    vi_cand = int(cv[li])
                    # Compute gain
                    cidx_v = var_clause_indices[vi_cand]
                    cpol_v = var_clause_polarities[vi_cand]
                    
                    if len(cidx_v) == 0:
                        g = 0
                    else:
                        old_val = assignment[vi_cand]
                        old_lit_val = 2 * old_val - 1
                        was_sat = (old_lit_val * cpol_v) > 0
                        
                        # Gain = (unsat clauses satisfied) - (sat clauses unsatisfied)
                        # Clauses that were unsat (count==3) and become sat (if this literal was false and becomes true)
                        # Clauses that were sat (count<3) and become unsat (if this literal was true and becomes false)
                        
                        # If literal was false (was_sat is False), flipping makes it true.
                        # Clauses where this literal is false: count decreases.
                        # If count was 3, becomes 2 (Sat). Gain +1.
                        # Clauses where this literal is true: count increases.
                        # If count was 2, becomes 3 (Unsat). Gain -1.
                        
                        unsat_clauses = cidx_v[~was_sat]
                        sat_clauses = cidx_v[was_sat]
                        
                        g = 0
                        if len(unsat_clauses) > 0:
                            counts = clause_unsat_count[unsat_clauses]
                            g += int((counts == 3).sum())
                        if len(sat_clauses) > 0:
                            counts = clause_unsat_count[sat_clauses]
                            g -= int((counts == 2).sum())
                    
                    g += rng.random() * 0.1
                    if g > best_gain:
                        best_gain = g
                        best_vi = vi_cand
                vi = best_vi
                
            # Flip
            flip_var(vi, assignment, clause_unsat_count, clause_weights)
            
            current_unsat = int((clause_unsat_count == 3).sum())
            
            if current_unsat == 0:
                return assignment.tolist()
                
            if current_unsat < local_best:
                local_best = current_unsat
                steps_since_improve = 0
                walk_prob = max(0.1, walk_prob * 0.9)
            else:
                steps_since_improve += 1
                walk_prob = min(0.8, walk_prob * 1.05)
                
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                
            if steps_since_improve > 5000:
                walk_prob = 0.2 # Reset walk prob

    if best_assignment is not None:
        return best_assignment.tolist()
        
    rng2 = np.random.default_rng(42)
    return (rng2.random(n) < 0.5).tolist()
