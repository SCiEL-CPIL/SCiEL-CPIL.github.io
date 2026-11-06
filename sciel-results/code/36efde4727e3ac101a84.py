"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


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
# EVOLVE-BLOCK-END
