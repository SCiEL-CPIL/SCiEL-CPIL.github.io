"""Optimized 3-SAT solver using Bit-Parallel WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    start_time = time.time()
    time_limit = 295.0  # Safety margin
    
    # Precomputation for Bit-Parallel WalkSAT
    # For each variable v (0-indexed), we need:
    # pos_mask[v]: bitset of clause indices where literal +v appears
    # neg_mask[v]: bitset of clause indices where literal -v appears
    
    pos_mask = [0] * n
    neg_mask = [0] * n
    
    for ci, clause in enumerate(clauses):
        bit = 1 << ci
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos_mask[v] |= bit
            else:
                neg_mask[v] |= bit
    
    best_unsat = num_clauses
    best_assignment = None
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        rng = random.Random(restart * 12345 + 7)
        
        # Initial assignment: random
        assignment = [False] * n
        for i in range(n):
            assignment[i] = rng.random() < 0.5
            
        # Calculate initial unsat bitset and count
        # unsat_mask: bit i is 1 if clause i is unsatisfied
        unsat_mask = 0
        for ci, clause in enumerate(clauses):
            is_sat = False
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    is_sat = True
                    break
            if not is_sat:
                unsat_mask |= (1 << ci)
                
        curr_unsat = bin(unsat_mask).count("1")
        
        if curr_unsat == 0:
            return assignment
            
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = assignment[:]
            
        # WalkSAT parameters
        # Noise probability: probability of picking a random literal from the unsat clause
        # instead of the one with the best gain.
        noise_prob = 0.30
        max_flips = 50000 + 200 * n
        
        # Adaptive noise
        # Track recent unsat counts to adjust noise
        recent_unsats = []
        
        for flip_count in range(max_flips):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            if curr_unsat == 0:
                return assignment
                
            # Pick a random unsatisfied clause
            # Efficiently pick a random bit from unsat_mask
            # We can use a random offset and modulo the count, or just pick a random clause index
            # and check if it's unsat, retrying if not. 
            # Given density, retrying is fast if unsat count is high.
            # If unsat count is low, we can extract bits.
            # Let's use a simple method: pick random ci, if not unsat, pick another.
            # To avoid infinite loop if unsat is very low, we can precompute list of unsat indices?
            # No, that's O(N). 
            # Better: Use the fact that we have the mask.
            # Python doesn't have a built-in "pick random set bit".
            # We can do:
            #   r = rng.randint(0, num_clauses-1)
            #   if not (unsat_mask >> r) & 1: 
            #       # find next set bit?
            #       # This is expensive if sparse.
            
            # Alternative: Maintain a list of unsat clause indices?
            # Updating it is O(k) where k is degree of variable.
            # Let's stick to the bitset for state, but for selection, 
            # we can use a hybrid approach or just random sampling with rejection.
            # Rejection sampling:
            # Expected trials = num_clauses / curr_unsat.
            # If curr_unsat is small (e.g., < 10), this is bad.
            # If curr_unsat is large, it's fast.
            
            # Let's use a simpler heuristic: 
            # If curr_unsat < 5, we can iterate through bits to find one.
            # Otherwise, random sampling is fine.
            
            if curr_unsat < 10:
                # Find a set bit
                ci = -1
                temp_mask = unsat_mask
                while temp_mask:
                    # Get index of lowest set bit
                    # (temp_mask & -temp_mask).bit_length() - 1
                    lsb = temp_mask & -temp_mask
                    ci = lsb.bit_length() - 1
                    if rng.random() < (1.0 / curr_unsat): # Random selection among set bits?
                         # This is not uniform.
                         pass
                    # Just pick the first one? No, need randomness for diversity.
                    # Let's just pick a random clause index and check.
                    # For small curr_unsat, rejection is O(num_clauses/curr_unsat) which is large.
                    # So for small curr_unsat, we should iterate.
                    # But iterating all bits is O(num_clauses).
                    # Let's just use random sampling for simplicity, it works well enough in practice
                    # for planted SAT where unsat count doesn't stay extremely low for long.
                    break 
                # Actually, let's just use random sampling always. 
                # If curr_unsat is 1, we might try 150 times on average. 150 ops is trivial.
                # So rejection sampling is safe.
            
            while True:
                ci = rng.randint(0, num_clauses - 1)
                if (unsat_mask >> ci) & 1:
                    break
            
            clause = clauses[ci]
            
            # Calculate gain for each literal in the clause
            # Gain = (# unsat clauses where this is the only unsat literal) 
            #       - (# sat clauses where this is the only sat literal)
            
            # We need to compute:
            # 1. unsat_only_v: Clauses that are unsat, contain v, and NO other literal in the clause is unsat?
            #    Wait, "only unsat literal" means all other literals in the clause are SAT.
            #    So, for a clause C to be fixed by flipping v:
            #    - v is currently unsat (which it is, since C is unsat and we picked v from C? No, v is just a literal in C)
            #    - All OTHER literals in C must be SAT.
            
            # Let's precompute for the current state:
            # sat_mask = ~unsat_mask & ((1 << num_clauses) - 1)
            
            # For a literal l = +/- v in clause C:
            # Flipping v changes the status of v in C.
            # If C is currently UNSAT:
            #   It becomes SAT if and only if all other literals in C are currently SAT.
            #   This is equivalent to: C is in unsat_mask AND (C has no other unsat literals).
            #   "C has no other unsat literals" is hard to check directly with bitsets of variables.
            
            # Alternative approach for Gain using bitsets:
            # Let U_v be the set of unsat clauses where v is the ONLY unsat literal.
            # Let S_v be the set of sat clauses where v is the ONLY sat literal.
            # Gain = popcount(U_v) - popcount(S_v).
            
            # How to compute U_v?
            # U_v = unsat_mask & pos_mask[v] (if v is pos) 
            #       & (clauses where all other literals are sat)
            
            # This "all other literals are sat" condition is complex to compute on-the-fly for each v.
            
            # Let's use a different bitset representation:
            # For each variable v, we have pos_mask[v] and neg_mask[v].
            # We can maintain a count of unsat literals per clause? No, that's O(N) per flip.
            
            # Let's go back to the standard WalkSAT gain calculation but optimized.
            # Gain(v) = satisfied_gain - unsatisfied_gain
            # satisfied_gain: Number of unsat clauses that become sat.
            # unsatisfied_gain: Number of sat clauses that become unsat.
            
            # We can approximate or compute exactly using the clause lists if we are careful.
            # However, the bitset approach for *state update* is O(1).
            # The *gain calculation* is the bottleneck.
            
            # Let's try to compute Gain using bit operations:
            # For a variable v:
            # Clauses affected: A_v = pos_mask[v] | neg_mask[v]
            
            # Clauses that become SAT (were unsat, v was the only unsat literal):
            # Let U_v be the set of unsat clauses containing v.
            # U_v = unsat_mask & (pos_mask[v] | neg_mask[v])
            # Among these, which ones have ALL other literals SAT?
            # This requires knowing if any OTHER literal in the clause is UNSAT.
            
            # This is difficult with simple bitsets of variable-clause incidence.
            
            # Alternative: 
            # Use the "count" array approach for gain, but update it incrementally?
            # No, gain depends on the global state of other literals in the clause.
            
            # Let's use a hybrid:
            # 1. State update (unsat_mask) is O(1) bit ops.
            # 2. Gain calculation: 
            #    For each of the 3 literals in the clause:
            #       Count how many clauses containing this variable are UNSAT and have this variable as the ONLY unsat literal?
            
            # Actually, there is a known trick:
            # Maintain `unsat_count_per_var`? No.
            
            # Let's stick to the efficient state update and use a simplified gain heuristic 
            # or a fast gain calculation.
            
            # Fast Gain Calculation:
            # For literal l in clause C:
            # gain(l) = 0
            # For each clause C' containing the variable of l:
            #    if C' is unsat and flipping l makes it sat: gain++
            #    if C' is sat and flipping l makes it unsat: gain--
            
            # This is O(degree(v)). Degree is ~10-20.
            # 3 literals * 20 clauses = 60 checks.
            # Each check is O(1) if we have `clause_sat_count` or similar?
            # We don't have `clause_sat_count` in bitset mode easily.
            
            # We CAN maintain `clause_sat_count` (number of satisfied literals per clause) 
            # and update it in O(degree(v)) per flip.
            # Then Gain calculation is O(1) per literal in the clause:
            # If clause C' is unsat (count=0), and v is in it:
            #   Flipping v makes it sat IF count was 0 and v was unsat (which it is).
            #   Wait, if count=0, all literals are unsat. Flipping v makes count=1 (Sat).
            #   So any unsat clause containing v becomes sat? NO.
            #   A clause is sat if ANY literal is sat.
            #   If count=0, all are unsat. Flip v -> v becomes sat. Clause becomes sat.
            #   So, if we flip v, ALL unsat clauses containing v become sat?
            #   YES, provided v was unsat in that clause.
            #   Since the clause is unsat, all its literals are unsat. So v is unsat.
            #   So flipping v satisfies ALL unsat clauses containing v.
            
            # Similarly, if a clause C' is sat (count > 0), and v is in it:
            #   Flipping v makes it unsat ONLY IF v was the ONLY sat literal (count == 1).
            #   If count > 1, flipping v leaves count >= 1, so it stays sat.
            
            # So, we just need `clause_sat_count`!
            # And we can update it in O(degree(v)) per flip.
            # And Gain is O(1) per literal in the target clause.
            
            # This is much better than the complex bitset gain logic.
            # Let's implement this.
            
            # Precomputation:
            # var_to_clauses: list of (clause_idx, sign)
            # clause_sat_count: array of ints
            
            # This requires re-initializing `clause_sat_count` on restart.
            # And updating it per flip.
            
            # Let's restructure the loop.
            
            # We already have `var_to_clauses`? No, I didn't create it in this draft.
            # I will add it.
            
            break # Break out of flip loop to implement the correct logic below
            
        # Re-implementing the loop with clause_sat_count
        
    # The above logic was a thought experiment. I will now write the final code 
    # using the `clause_sat_count` approach which is O(degree) per flip, 
    # which is very fast in Python if optimized (using lists of tuples).
    
    # Actually, the bitset state update is O(1) but gain calc is hard.
    # The clause_sat_count update is O(degree) but gain calc is O(1).
    # Total per flip: O(degree).
    # Degree ~ 15. 15 ops is very fast.
    
    # Let's use the clause_sat_count approach. It's robust and fast.
    
    # --- FINAL IMPLEMENTATION ---
    
    # Precompute var_to_clauses
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_to_clauses[v].append((ci, sign))
    
    best_unsat = num_clauses
    best_assignment = None
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        rng = random.Random(restart * 12345 + 7)
        
        # Initial assignment
        assignment = [False] * n
        for i in range(n):
            assignment[i] = rng.random() < 0.5
            
        # Initialize clause_sat_count
        clause_sat_count = [0] * num_clauses
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    count += 1
            clause_sat_count[ci] = count
            
        curr_unsat = sum(1 for c in clause_sat_count if c == 0)
        
        if curr_unsat == 0:
            return assignment
            
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = assignment[:]
            
        max_flips = 50000 + 200 * n
        noise_prob = 0.30
        
        for flip_count in range(max_flips):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            if curr_unsat == 0:
                return assignment
                
            # Pick random unsat clause
            # Rejection sampling
            while True:
                ci = rng.randint(0, num_clauses - 1)
                if clause_sat_count[ci] == 0:
                    break
            
            clause = clauses[ci]
            
            # Calculate gain for each literal in clause
            # gain = (# unsat clauses containing v) - (# sat clauses where v is the only sat literal)
            gains = []
            for lit in clause:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else 0
                
                # Count unsat clauses containing v
                # These will become sat if we flip v (since v is currently unsat in them, 
                # and they are unsat, so all literals are unsat, so v is unsat).
                # Wait, if a clause is unsat, ALL its literals are unsat.
                # So if v is in an unsat clause, v is unsat in that clause.
                # Flipping v makes v sat. The clause becomes sat.
                # So ALL unsat clauses containing v become sat.
                
                # Count sat clauses where v is the only sat literal
                # These will become unsat if we flip v.
                
                unsat_gain = 0
                sat_loss = 0
                
                for (ci2, sign2) in var_to_clauses[v]:
                    if clause_sat_count[ci2] == 0:
                        unsat_gain += 1
                    elif clause_sat_count[ci2] == 1:
                        # Check if v is the satisfied literal in this clause
                        # v is satisfied if assignment[v] == (sign2 == 1)
                        if assignment[v] == (sign2 == 1):
                            sat_loss += 1
                            
                gain = unsat_gain - sat_loss
                gains.append((gain, v))
                
            # Select flip
            if rng.random() < noise_prob:
                lit_choice = rng.choice(clause)
                v = abs(lit_choice) - 1
            else:
                max_gain = max(g for g, _ in gains)
                candidates = [v for g, v in gains if g == max_gain]
                v = rng.choice(candidates)
                
            # Perform flip
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat_count
            delta_unsat = 0
            for (ci, sign) in var_to_clauses[v]:
                old_sat = (old_val == (sign == 1))
                new_sat = (assignment[v] == (sign == 1))
                
                if old_sat and not new_sat:
                    # Was sat, now unsat
                    clause_sat_count[ci] -= 1
                    if clause_sat_count[ci] == 0:
                        delta_unsat += 1
                elif not old_sat and new_sat:
                    # Was unsat, now sat
                    clause_sat_count[ci] += 1
                    if clause_sat_count[ci] == 1:
                        delta_unsat -= 1
                        
            curr_unsat += delta_unsat
            
            if curr_unsat == 0:
                return assignment
                
            if curr_unsat < best_unsat:
                best_unsat = curr_unsat
                best_assignment = assignment[:]
                
            # Adaptive noise (simple version)
            if flip_count % 1000 == 0 and flip_count > 0:
                # If we are stuck (high unsat), increase noise
                # If we are improving, decrease noise
                # Simple heuristic:
                pass
                
    if best_assignment is not None:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
