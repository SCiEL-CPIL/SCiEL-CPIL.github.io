<<<<<<< SEARCH
def solve(instance):
    # Robust WalkSAT with time guard, restarts, and a final polish pass.
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute clause literals as (var_index, polarity) for speed
    # var_index is 0-based, polarity is True if positive literal
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            vi = abs(l) - 1
            pol = (l > 0)
            cd.append((vi, pol))
        clause_data.append(cd)
    
    # Deadline guard: 300s limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Try multiple random restarts
    max_restarts = 50
    flips_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Count unsatisfied clauses
        def count_unsat(assignment, clause_data):
            cnt = 0
            for cd in clause_data:
                sat = False
                for vi, pol in cd:
                    if assignment[vi] == pol:
                        sat = True
                        break
                if not sat:
                    cnt += 1
            return cnt
        
        unsat_count = count_unsat(assignment, clause_data)
        if unsat_count == 0:
            return assignment
        
        # WalkSAT loop
        for flip in range(flips_per_restart):
            if time.time() > deadline - 2.0:
                break
                
            if unsat_count == 0:
                if best_unsat_count > unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
                return assignment
            
            # Find an unsatisfied clause
            unsat_clause = None
            for cd in clause_data:
                sat = False
                for vi, pol in cd:
                    if assignment[vi] == pol:
                        sat = True
                        break
                if not sat:
                    unsat_clause = cd
                    break
            
            if unsat_clause is None:
                # Should not happen if unsat_count > 0, but safety check
                break
            
            # Choose a literal to flip
            # Prefer flipping a literal that satisfies other clauses (min-conflicts heuristic)
            best_var = None
            best_delta = -1
            best_pol = None
            
            for vi, pol in unsat_clause:
                # If we flip this variable, does it satisfy this clause?
                # Currently assignment[vi] != pol (since clause is unsat)
                # After flip, assignment[vi] will be pol, so this clause becomes sat
                # Count how many other clauses would be broken
                delta = 0
                for cd in clause_data:
                    # Check if this clause is currently satisfied
                    sat = False
                    for v2, p2 in cd:
                        if assignment[v2] == p2:
                            sat = True
                            break
                    if not sat:
                        # Clause is currently unsat
                        # After flip, will it become sat?
                        # It becomes sat if the flipped variable's new value matches its polarity
                        # New value of vi is (not assignment[vi])
                        # We need to check if any literal in cd matches the new assignment
                        # But we only know the effect on vi
                        # If vi is in cd, and new assignment[vi] == pol_in_cd, then it helps
                        # If vi is not in cd, no change
                        new_vi_val = not assignment[vi]
                        helps = False
                        hurts = False
                        for v2, p2 in cd:
                            if v2 == vi:
                                if new_vi_val == p2:
                                    helps = True
                                break
                        # For simplicity, just count if flipping vi would satisfy this unsat clause
                        # This is a simplified heuristic
                        pass
                # Simplified: just pick randomly from the clause, but with slight preference
                # For speed, just pick the first literal that appears least often in other unsat clauses
                # Actually, let's just use a simple random choice for now, but try to be smart
                # Count occurrences of vi in other unsat clauses
                occ = 0
                for cd2 in clause_data:
                    sat2 = False
                    for v2, p2 in cd2:
                        if assignment[v2] == p2:
                            sat2 = True
                            break
                    if not sat2:
                        for v2, p2 in cd2:
                            if v2 == vi:
                                occ += 1
                                break
                if occ < best_delta or best_var is None:
                    best_delta = occ
                    best_var = vi
                    best_pol = pol
            
            if best_var is not None:
                assignment[best_var] = not assignment[best_var]
                # Recount unsat (expensive, but necessary for accuracy)
                # Optimization: only recount if we want to be precise
                # For now, recount every 100 flips or at the end
                if flip % 100 == 0 or flip == flips_per_restart - 1:
                    unsat_count = count_unsat(assignment, clause_data)
            else:
                break
        
        # After restart, check if we found a solution
        final_unsat = count_unsat(assignment, clause_data)
        if final_unsat == 0:
            return assignment
        
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = assignment[:]
    
    # Final polish: if we have a near-solution, try to fix the remaining unsat clauses
    if best_assignment is not None and best_unsat_count > 0 and time.time() < deadline - 1.0:
        assignment = best_assignment[:]
        for _ in range(1000):
            if time.time() > deadline - 0.5:
                break
            unsat_count = count_unsat(assignment, clause_data)
            if unsat_count == 0:
                return assignment
            # Find an unsat clause and flip a variable in it
            for cd in clause_data:
                sat = False
                for vi, pol in cd:
                    if assignment[vi] == pol:
                        sat = True
                        break
                if not sat:
                    vi, pol = cd[0]  # Flip first variable
                    assignment[vi] = not assignment[vi]
                    break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return the last assignment
    return assignment
=======
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute clause literals
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            vi = abs(l) - 1
            pol = (l > 0)
            cd.append((vi, pol))
        clause_data.append(cd)
    
    # Map variable to list of (clause_index, polarity)
    var_map = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for vi, pol in cd:
            var_map[vi].append((i, pol))

    start_time = time.time()
    deadline = start_time + 295.0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 5000
    steps_per_restart = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize unsat state
        is_unsat = [False] * len(clause_data)
        unsat_list = []
        
        for i, cd in enumerate(clause_data):
            sat = False
            for vi, pol in cd:
                if assignment[vi] == pol:
                    sat = True
                    break
            if not sat:
                is_unsat[i] = True
                unsat_list.append(i)
        
        unsat_count = len(unsat_list)
        if unsat_count == 0:
            return assignment
        
        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break
                
            if unsat_count == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            idx = rng.choice(unsat_list)
            cd = clause_data[idx]
            
            # Choose variable to flip using Min-Conflicts heuristic
            # Count how many unsat clauses each variable in cd is part of
            occ_counts = {vi: 0 for vi, _ in cd}
            
            for u_idx in unsat_list:
                for vi, _ in clause_data[u_idx]:
                    if vi in occ_counts:
                        occ_counts[vi] += 1
                        
            min_occ = min(occ_counts.values())
            candidates = [vi for vi, cnt in occ_counts.items() if cnt == min_occ]
            vi_to_flip = rng.choice(candidates)
            
            # Flip the variable
            assignment[vi_to_flip] = not assignment[vi_to_flip]
            
            # Update unsat status for clauses containing vi_to_flip
            for c_idx, pol in var_map[vi_to_flip]:
                # Check if clause c_idx is satisfied
                sat = False
                for v, p in clause_data[c_idx]:
                    if assignment[v] == p:
                        sat = True
                        break
                
                if sat:
                    if is_unsat[c_idx]:
                        is_unsat[c_idx] = False
                        unsat_count -= 1
                        # Remove from unsat_list (lazy deletion or swap with last)
                        # For efficiency, we can just leave it in the list and filter later, 
                        # or swap with last and pop.
                        # Swap with last is O(1) if we know the index.
                        # But finding the index in unsat_list is O(U).
                        # Let's just mark it and clean up periodically or use a set.
                        # Actually, for WalkSAT, maintaining an exact list is tricky without a set.
                        # Let's use a set for unsat_indices for O(1) add/remove.
                        pass 
                else:
                    if not is_unsat[c_idx]:
                        is_unsat[c_idx] = True
                        unsat_count += 1
                        # Add to unsat_list
                        pass
                        
            # The above update logic is flawed because `unsat_list` is a list and removing items is O(N).
            # Let's switch to a set for `unsat_indices` for O(1) updates.
            
            # Restarting the inner loop logic with Set-based unsat tracking
            # This requires changing the initialization and the loop.
            # Since I can't easily refactor the whole function in a small edit, 
            # and the previous attempt was messy, I will provide a fully correct 
            # implementation in the next block.
            break
            
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
    if best_assignment is not None:
        return best_assignment
    return assignment
>>>>>>> REPLACE

Wait, the `break` and `pass` comments indicate I'm still struggling with the in-place edit logic. The best approach is to replace the *entire* function body with a clean, correct, and efficient implementation using a set for unsatisfied clauses.

