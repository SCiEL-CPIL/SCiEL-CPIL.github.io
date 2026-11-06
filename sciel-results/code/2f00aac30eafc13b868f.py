# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random
import heapq

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert clauses to 0-based
    # We will maintain a list of clauses where each clause is a list of (var, sign)
    # sign: 1 for positive, 0 for negative
    # We also track which clauses are active (not satisfied)
    
    # 1. Preprocessing: Pure Literal Elimination & Unit Propagation
    
    # Build index: var -> list of (clause_idx, sign)
    var_occ = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_occ[var].append((ci, sign))
    
    assignment = [None] * n
    
    # Helper to check if a clause is satisfied
    def is_clause_sat(ci):
        for lit in clauses[ci]:
            var = abs(lit) - 1
            val = assignment[var]
            if val is not None:
                if (lit > 0 and val) or (lit < 0 and not val):
                    return True
        return False
    
    # Helper to simplify clause (remove assigned literals, return None if sat, [] if empty, else list of unassigned lits)
    def simplify_clause(ci):
        sat = False
        remaining = []
        for lit in clauses[ci]:
            var = abs(lit) - 1
            val = assignment[var]
            if val is not None:
                if (lit > 0 and val) or (lit < 0 and not val):
                    sat = True
                    break
            else:
                remaining.append(lit)
        if sat:
            return None
        if len(remaining) == 0:
            return []
        return remaining

    # Iterative Pure Literal and Unit Propagation
    changed = True
    while changed:
        changed = False
        
        # Pass 1: Pure Literal Elimination
        for var in range(n):
            if assignment[var] is not None:
                continue
            pos = False
            neg = False
            for ci, sign in var_occ[var]:
                # Check if this clause is already satisfied by other literals
                # Optimization: Just check if var appears in active clauses
                # For simplicity and correctness in this loop, we check occurrence in original clauses
                # but we must be careful if the clause is already satisfied.
                # Actually, pure literal elimination is safe if we just look at *active* occurrences.
                # Let's refine: A variable is pure if it only appears in one polarity in the *current* set of unsatisfied clauses.
                pass # We will do a more robust pass below or just rely on unit prop first?
                # Standard approach: 
                # 1. Find pure literals.
                # 2. Assign them.
                # 3. Propagate units.
                
        # Let's do a cleaner loop:
        # 1. Identify pure literals in the set of currently unsatisfied clauses.
        # We need a list of active clause indices.
        active_clauses = [ci for ci in range(len(clauses)) if not is_clause_sat(ci)]
        
        # Determine polarity counts for active clauses
        var_pos_active = defaultdict(int)
        var_neg_active = defaultdict(int)
        for ci in active_clauses:
            for lit in clauses[ci]:
                var = abs(lit) - 1
                if lit > 0:
                    var_pos_active[var] += 1
                else:
                    var_neg_active[var] += 1
        
        # Assign pure literals
        pure_found = False
        for var in range(n):
            if assignment[var] is not None:
                continue
            if var_pos_active[var] == 0 and var_neg_active[var] > 0:
                assignment[var] = True
                pure_found = True
                changed = True
            elif var_neg_active[var] == 0 and var_pos_active[var] > 0:
                assignment[var] = False
                pure_found = True
                changed = True
        
        # If we assigned pure literals, we should re-evaluate active clauses and unit clauses.
        # But unit propagation is more critical. Let's do unit propagation next.
        
        # Pass 2: Unit Propagation
        # We need to find unit clauses among the active ones.
        # We'll loop until no new units are found or no changes.
        unit_changed = True
        while unit_changed:
            unit_changed = False
            active_clauses = [ci for ci in range(len(clauses)) if not is_clause_sat(ci)]
            
            for ci in active_clauses:
                rem = simplify_clause(ci)
                if rem is None:
                    continue # Satisfied
                if len(rem) == 0:
                    # Empty clause: UNSAT. But problem is planted SAT. 
                    # If this happens, we failed preprocessing. Return random or best effort.
                    # For now, break and let WalkSAT handle it (it will struggle).
                    pass
                elif len(rem) == 1:
                    lit = rem[0]
                    var = abs(lit) - 1
                    if assignment[var] is None:
                        assignment[var] = (lit > 0)
                        unit_changed = True
                        changed = True
                        # Restart unit propagation to catch cascading effects
                        break
        
        # If we made changes, the while loop repeats to find more pure literals/units.
        # If no changes, we are done with preprocessing.

    # Check if all satisfied
    all_sat = True
    for ci in range(len(clauses)):
        if not is_clause_sat(ci):
            all_sat = False
            break
            
    if all_sat:
        final_assignment = []
        rng = random.Random(0)
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment

    # 2. WalkSAT on the reduced instance
    # Identify unsatisfied clauses
    unsat_indices = [ci for ci in range(len(clauses)) if not is_clause_sat(ci)]
    
    if not unsat_indices:
        # Should have been caught above
        final_assignment = []
        rng = random.Random(0)
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment

    return _walksat_optimized(n, clauses, assignment, unsat_indices, rng_seed=42)

def _walksat_optimized(n, clauses, assignment, initial_unsat_indices, rng_seed=42):
    """
    Optimized WalkSAT.
    """
    rng = random.Random(rng_seed)
    
    # Initialize full assignment for unassigned vars
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = rng.random() < 0.5
    
    m = len(clauses)
    
    # Precompute clause structure for fast access
    # clause_vars[ci] = list of (var, sign)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            cv.append((var, sign))
        clause_vars.append(cv)
    
    # Precompute for each variable, which clauses it appears in and with what sign
    # var_clauses[var] = list of (ci, sign)
    var_clauses = defaultdict(list)
    for ci in range(m):
        for var, sign in clause_vars[ci]:
            var_clauses[var].append((ci, sign))
    
    # Track unsatisfied clauses
    # We can use a set or a list. For fast removal/addition, a set is good for membership,
    # but we need to pick randomly.
    # Let's maintain a list of unsat clauses and a set for O(1) check.
    unsat_set = set(initial_unsat_indices)
    unsat_list = list(unsat_set)
    
    # Function to check if a clause is satisfied given current assignment
    def is_sat(ci):
        for var, sign in clause_vars[ci]:
            if assignment[var] == (sign == 1):
                return True
        return False
    
    # Initialize unsat_list based on current assignment (in case preprocessing left some inconsistencies or we just want to be sure)
    unsat_list = []
    unsat_set = set()
    for ci in range(m):
        if not is_sat(ci):
            unsat_list.append(ci)
            unsat_set.add(ci)
            
    if not unsat_list:
        return assignment

    max_flips = 100000
    noise_start = 0.2
    noise_end = 0.02
    
    for flip in range(max_flips):
        if not unsat_list:
            break
        
        # Dynamic noise
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Pick a random unsatisfied clause
        # To avoid bias towards early indices, we can pick randomly from the list
        idx = rng.randrange(len(unsat_list))
        ci = unsat_list[idx]
        
        # Get the variables in this clause
        c_vars = clause_vars[ci]
        
        # Choose a variable to flip
        if rng.random() < noise:
            # Random choice
            var, sign = c_vars[rng.randrange(len(c_vars))]
        else:
            # Greedy choice: Maximize gain (clauses satisfied) - Minimize loss (clauses unsatisfied)
            best_var = None
            best_score = -1
            
            for var, sign in c_vars:
                new_val = not assignment[var]
                gain = 0
                loss = 0
                
                # Check all clauses containing this variable
                for ci2, sign2 in var_clauses[var]:
                    if ci2 == ci:
                        continue # We are already flipping to satisfy this one (if it becomes sat)
                    
                    # Determine if flipping var to new_val satisfies ci2
                    # ci2 is satisfied if any literal is true.
                    # We only care about the effect of this variable.
                    # If new_val satisfies the literal in ci2, then:
                    #   - If ci2 was unsat, it becomes sat (Gain +1)
                    #   - If ci2 was sat, it might become unsat if this was the *only* satisfied literal (Loss +1)
                    
                    # Efficient check:
                    # Does new_val satisfy the literal (var, sign2) in ci2?
                    if new_val == (sign2 == 1):
                        # This literal would be True.
                        # If ci2 is currently unsat, it becomes sat.
                        if ci2 in unsat_set:
                            gain += 1
                        else:
                            # ci2 is currently sat.
                            # It becomes unsat ONLY if no other literal in ci2 is satisfied.
                            # Check if any other literal in ci2 is satisfied.
                            is_still_sat = False
                            for v2, s2 in clause_vars[ci2]:
                                if v2 == var:
                                    continue
                                if assignment[v2] == (s2 == 1):
                                    is_still_sat = True
                                    break
                            if not is_still_sat:
                                loss += 1
                    else:
                        # This literal would be False.
                        # If ci2 is currently sat, and this was the *only* satisfied literal, it becomes unsat.
                        if ci2 not in unsat_set:
                            # Check if any other literal is satisfied
                            is_still_sat = False
                            for v2, s2 in clause_vars[ci2]:
                                if v2 == var:
                                    continue
                                if assignment[v2] == (s2 == 1):
                                    is_still_sat = True
                                    break
                            if not is_still_sat:
                                loss += 1
                        # If ci2 is unsat, it remains unsat (since this literal was false and remains false)
                    
                    score = gain - loss
                    if score > best_score:
                        best_score = score
                        best_var = var
                        
            if best_var is None:
                best_var = c_vars[0][0]
            var = best_var
            
        # Flip the variable
        old_val = assignment[var]
        assignment[var] = not old_val
        
        # Update unsat set/list
        # We need to update the status of all clauses containing var
        # And specifically the clause we picked (ci)
        
        # Helper to update a clause's status
        def update_clause_status(c_idx):
            nonlocal unsat_list, unsat_set
            # Check if satisfied
            sat = False
            for v, s in clause_vars[c_idx]:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            
            if sat:
                if c_idx in unsat_set:
                    unsat_set.remove(c_idx)
                    # Remove from list (lazy deletion or swap)
                    # For efficiency, we can just filter the list periodically or use a set for membership
                    # and a separate list for random selection.
                    # Simple approach: Mark as removed, clean up list occasionally.
                    # Or, since n is small, we can just rebuild the list if it gets too dirty.
                    # Let's use a "dirty" flag approach or just rebuild if len(unsat_list) > 2 * len(unsat_set)
                    pass # We will handle removal below
            else:
                if c_idx not in unsat_set:
                    unsat_set.add(c_idx)
                    unsat_list.append(c_idx)
            
        # This is getting complex with list management.
        # Given n=150, m is likely around 400-600.
        # Rebuilding the unsat list every flip is O(m) which is ~500.
        # 100000 flips * 500 = 50M operations. This might be slow in Python.
        
        # Alternative: Just recalculate unsat_list from scratch every K flips or if it gets too large.
        # Or, simpler: Since we only change one variable, we only need to update clauses containing that variable.
        
        affected_clauses = set([c[0] for c in var_clauses[var]])
        
        # Update statuses for affected clauses
        for c_idx in affected_clauses:
            # Check old status
            was_unsat = c_idx in unsat_set
            
            # Check new status
            is_now_sat = False
            for v, s in clause_vars[c_idx]:
                if assignment[v] == (s == 1):
                    is_now_sat = True
                    break
            
            if is_now_sat:
                if was_unsat:
                    unsat_set.remove(c_idx)
            else:
                if not was_unsat:
                    unsat_set.add(c_idx)
                    unsat_list.append(c_idx)
        
        # Clean up unsat_list: remove indices that are no longer in unsat_set
        # To keep it simple and fast enough for 150 vars:
        # If len(unsat_list) is much larger than len(unsat_set), rebuild.
        if len(unsat_list) > 2 * len(unsat_set) + 10:
            unsat_list = list(unsat_set)
            
    return assignment

# EVOLVE-BLOCK-END
