# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # 1. Preprocessing: Pure Literal Elimination and Unit Propagation
    # We will simplify the instance.
    # state: 
    # assignment: list of (var, value) or dict
    # active_clauses: list of lists of (var, sign)
    
    # Convert to 0-based
    raw_clauses = []
    for clause in clauses:
        c = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            c.append((var, sign))
        raw_clauses.append(c)
        
    assignment = {} # var -> bool
    active_clauses = raw_clauses
    
    # Helper to check if clause is satisfied given assignment
    def is_clause_sat(clause, assignment):
        for var, sign in clause:
            if var in assignment:
                if assignment[var] == (sign == 1):
                    return True
        return False

    # Helper to simplify clause given assignment
    def simplify_clause(clause, assignment):
        new_clause = []
        for var, sign in clause:
            if var in assignment:
                if assignment[var] == (sign == 1):
                    return None # Satisfied
            else:
                new_clause.append((var, sign))
        return new_clause

    # Iterative preprocessing
    changed = True
    while changed:
        changed = False
        
        # 1. Pure Literal Elimination
        # Build occurrence map for unassigned vars
        var_occurrences = defaultdict(lambda: [0, 0]) # var -> [pos_count, neg_count]
        for clause in active_clauses:
            for var, sign in clause:
                if var not in assignment:
                    if sign == 1:
                        var_occurrences[var][0] += 1
                    else:
                        var_occurrences[var][1] += 1
        
        for var, (pos, neg) in var_occurrences.items():
            if var in assignment:
                continue
            if pos > 0 and neg == 0:
                assignment[var] = False # Set to False to satisfy positive literals? 
                # Wait: literal is +var. To satisfy +var, var must be True.
                # If only positive occurrences, setting var=True satisfies all.
                assignment[var] = True
                changed = True
            elif neg > 0 and pos == 0:
                # Only negative occurrences (-var). To satisfy -var, var must be False.
                assignment[var] = False
                changed = True
        
        # 2. Simplify clauses based on new assignments
        new_active_clauses = []
        unit_clauses = []
        for clause in active_clauses:
            simplified = simplify_clause(clause, assignment)
            if simplified is None:
                continue # Satisfied
            if len(simplified) == 0:
                # Unsat, but problem is planted SAT
                return [False] * n # Fallback
            if len(simplified) == 1:
                unit_clauses.append(simplified[0])
            else:
                new_active_clauses.append(simplified)
        
        active_clauses = new_active_clauses
        
        # 3. Unit Propagation
        if unit_clauses:
            changed = True
            for var, sign in unit_clauses:
                if var not in assignment:
                    assignment[var] = (sign == 1)
                    # Loop will restart and simplify again
                    break # Restart loop to simplify all clauses with this new assignment
            
    # 2. Solve reduced instance with WalkSAT
    # Get remaining unassigned variables
    unassigned_vars = [i for i in range(n) if i not in assignment]
    
    if not active_clauses:
        # All satisfied
        final_assignment = [False] * n
        for var, val in assignment.items():
            final_assignment[var] = val
        # Fill rest randomly
        rng = random.Random(0)
        for var in unassigned_vars:
            final_assignment[var] = rng.random() < 0.5
        return final_assignment

    # Prepare data for WalkSAT
    # We only care about unassigned variables and active clauses
    # Map original var index to local index? No, just use original indices but filter.
    
    # Build index: var -> list of clause indices in active_clauses
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(active_clauses):
        for var, sign in clause:
            var_to_clauses[var].append(ci)
            
    m = len(active_clauses)
    
    # Initialize local assignment for unassigned vars
    # We will work with a dictionary or array for speed.
    # Since n is small (150), an array is fine.
    local_assignment = {}
    rng = random.Random(42)
    for var in unassigned_vars:
        local_assignment[var] = rng.random() < 0.5
        
    # Initialize clause satisfaction counts
    # clause_sat_count[ci] = number of satisfied literals in clause ci
    clause_sat_count = [0] * m
    for ci, clause in enumerate(active_clauses):
        count = 0
        for var, sign in clause:
            if local_assignment[var] == (sign == 1):
                count += 1
        clause_sat_count[ci] = count
        
    # List of unsatisfied clause indices
    unsat_clauses = [ci for ci in range(m) if clause_sat_count[ci] == 0]
    
    max_flips = 100000
    noise_start = 0.25
    noise_end = 0.05
    
    for flip in range(max_flips):
        if not unsat_clauses:
            break
            
        # Select clause
        # Weight by length? Shorter clauses are easier to satisfy? 
        # Usually random is fine, or weight by 1/len.
        # Let's stick to uniform random for simplicity and speed, or slight preference for shorter.
        # Given 150 vars, m is likely not huge.
        ci = rng.choice(unsat_clauses)
        clause = active_clauses[ci]
        
        # Select variable to flip
        # Calculate gain for each variable in clause
        # Gain = (# clauses that become satisfied) - (# clauses that become unsatisfied)
        # Since clause ci is currently unsatisfied, flipping any var in it *might* satisfy it.
        # Actually, if we flip var, and the literal in ci becomes true, ci becomes satisfied.
        
        best_var = None
        best_gain = -float('inf')
        
        # Dynamic noise
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        if rng.random() < noise:
            # Random flip from clause
            var, sign = rng.choice(clause)
            best_var = var
        else:
            for var, sign in clause:
                gain = 0
                # Check all clauses containing var
                for ci2 in var_to_clauses[var]:
                    # Determine if flipping var changes satisfaction of ci2
                    # Current state of var in ci2:
                    # If local_assignment[var] == (sign2 == 1), literal is True
                    # If we flip, it becomes False.
                    
                    # We need to know if ci2 is currently satisfied and if it depends on this literal being true.
                    # Simpler: Just calculate the change in sat_count for ci2.
                    
                    # Current sat count of ci2
                    # We don't have sat count for individual literals, just total.
                    # We can check:
                    # 1. Does ci2 contain var? Yes.
                    # 2. What is the sign of var in ci2?
                    # We need to find the sign.
                    
                    # Optimization: Precompute for each clause, the signs?
                    # Or just iterate clause ci2.
                    
                    # Since degree is low, iterating is okay.
                    # But we can be smarter.
                    
                    # Let's just compute the delta.
                    # If var is currently True:
                    #   If sign2 is +, literal is True. Flip -> False.
                    #   If sign2 is -, literal is False. Flip -> True.
                    
                    # We need to know if this literal is the *only* reason ci2 is satisfied.
                    
                    # Let's just re-evaluate satisfaction of ci2 before and after flip?
                    # That's O(len(clause)).
                    
                    # Alternative:
                    # For each ci2 in var_to_clauses[var]:
                    #   Find sign of var in ci2.
                    #   current_val = local_assignment[var]
                    #   new_val = not current_val
                    #   current_lit_val = (current_val == (sign2 == 1))
                    #   new_lit_val = (new_val == (sign2 == 1))
                    
                    #   if current_lit_val and not new_lit_val:
                    #       # Literal was True, becomes False.
                    #       # ci2 loses a satisfied literal.
                    #       # If ci2 was satisfied ONLY by this literal, it becomes unsatisfied.
                    #       # We need to know if any other literal in ci2 is satisfied.
                    #       # This requires checking other literals in ci2.
                    #       pass
                    #   elif not current_lit_val and new_lit_val:
                    #       # Literal was False, becomes True.
                    #       # ci2 gains a satisfied literal.
                    #       # If ci2 was unsatisfied, it becomes satisfied.
                    #       pass
                    
                    # This "check other literals" part is the expensive part if done naively.
                    
                    # However, for 150 vars, the average clause length is 3.
                    # The number of clauses per variable is likely small (e.g., < 20).
                    # So checking other literals is cheap.
                    
                    # Let's implement the check.
                    
                    # To make it faster, we can pre-store the list of (var, sign) for each clause.
                    # We already have active_clauses.
                    
                    # Find sign2
                    # Linear search in ci2 clause for var.
                    sign2 = None
                    for v2, s2 in active_clauses[ci2]:
                        if v2 == var:
                            sign2 = s2
                            break
                    if sign2 is None: continue # Should not happen
                    
                    current_val = local_assignment[var]
                    new_val = not current_val
                    
                    current_lit_sat = (current_val == (sign2 == 1))
                    new_lit_sat = (new_val == (sign2 == 1))
                    
                    if current_lit_sat and not new_lit_sat:
                        # We are turning a True literal to False.
                        # If ci2 is currently satisfied, and this was the only true literal, it becomes unsat.
                        if clause_sat_count[ci2] > 0:
                            # Check if any other literal in ci2 is satisfied
                            other_sat = False
                            for v2, s2 in active_clauses[ci2]:
                                if v2 == var:
                                    continue
                                if local_assignment[v2] == (s2 == 1):
                                    other_sat = True
                                    break
                            if not other_sat:
                                gain -= 1 # Lose satisfaction
                                # Note: if ci2 is the chosen clause ci, and it becomes unsat?
                                # ci is currently unsat (count 0). So current_lit_sat must be False.
                                # So this branch (current_lit_sat True) won't happen for ci.
                    elif not current_lit_sat and new_lit_sat:
                        # We are turning a False literal to True.
                        # If ci2 is currently unsatisfied, it becomes satisfied.
                        if clause_sat_count[ci2] == 0:
                            gain += 1 # Gain satisfaction
                            
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
                    
        # Flip best_var
        var = best_var
        local_assignment[var] = not local_assignment[var]
        
        # Update clause_sat_count and unsat_clauses
        # Only clauses containing var are affected
        # We need to update the list `unsat_clauses`
        # This is tricky to do in-place efficiently.
        # Strategy: Mark clauses to be added/removed, then update list.
        
        clauses_to_add = []
        clauses_to_remove = []
        
        for ci2 in var_to_clauses[var]:
            # Determine change in sat count
            # Find sign of var in ci2
            sign2 = None
            for v2, s2 in active_clauses[ci2]:
                if v2 == var:
                    sign2 = s2
                    break
            
            new_val = local_assignment[var]
            new_lit_sat = (new_val == (sign2 == 1))
            
            # We need to know old_lit_sat.
            # We can infer it: 
            # If we just flipped, old_val = not new_val.
            old_val = not new_val
            old_lit_sat = (old_val == (sign2 == 1))
            
            if new_lit_sat and not old_lit_sat:
                # Count increased
                clause_sat_count[ci2] += 1
                if clause_sat_count[ci2] == 1:
                    # Became satisfied
                    clauses_to_remove.append(ci2)
            elif old_lit_sat and not new_lit_sat:
                # Count decreased
                clause_sat_count[ci2] -= 1
                if clause_sat_count[ci2] == 0:
                    # Became unsatisfied
                    clauses_to_add.append(ci2)
                    
        # Update unsat_clauses list
        # Remove those in clauses_to_remove
        # Add those in clauses_to_add
        
        # To avoid O(m) scan for removal, we can use a set or just filter.
        # Since m is small, filtering is fine.
        remove_set = set(clauses_to_remove)
        unsat_clauses = [ci for ci in unsat_clauses if ci not in remove_set]
        unsat_clauses.extend(clauses_to_add)
        
    # Construct final assignment
    final_assignment = [False] * n
    for var, val in assignment.items():
        final_assignment[var] = val
    for var in unassigned_vars:
        final_assignment[var] = local_assignment[var]
        
    return final_assignment

# EVOLVE-BLOCK-END
