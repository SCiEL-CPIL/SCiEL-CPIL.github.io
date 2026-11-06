# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 295.0  # safety margin before 300s kill

    # Convert to 0-based signed literals
    # literal l: variable = abs(l)-1, sign = l > 0
    # We'll store as (var, sign) where sign is +1 or -1
    # Actually let's keep as signed integers but 0-based: 
    # original literal l -> new literal: if l > 0: l-1, else: -(abs(l)-1)
    # Wait, that's confusing. Let me use a different representation.
    # Let's use: literal value = var_index if positive, -(var_index+1) if negative
    # So original l: if l > 0, new = l-1; if l < 0, new = -(abs(l)) = -abs(l)
    # Hmm, let me just use (var, is_positive) tuples internally.
    
    # Actually for speed, let's use a compact representation:
    # literal l (1-based signed) -> 0-based: 
    #   if l > 0: var = l-1, positive
    #   if l < 0: var = -l-1, negative
    # Store as: positive literal = var, negative literal = ~var (bitwise not)
    # So literal l -> if l>0: l-1, else: ~(-l-1) = l  (wait that's the same)
    # Let me just use: lit = abs(l)-1 for var, and track sign separately.
    # For speed in the inner loop, I'll use: 
    #   pos_lit = var (0-based)
    #   neg_lit = ~var (bitwise complement, so -1, -2, ... for var 0, 1, ...)
    # This way: var_of(lit) = abs(lit), is_pos(lit) = lit >= 0
    
    converted_clauses = []
    for clause in clauses:
        new_clause = []
        for l in clause:
            if l > 0:
                new_clause.append(l - 1)  # positive literal
            else:
                new_clause.append(~(-l - 1))  # negative literal: ~var
        converted_clauses.append(new_clause)
    
    num_clauses = len(converted_clauses)
    
    # Build clause index for each variable
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(converted_clauses):
        for lit in clause:
            var = abs(lit)
            var_to_clauses[var].append(ci)
    
    # Preprocessing: pure literal elimination
    def preprocess():
        """Apply pure literal elimination. Returns (assignment_dict, remaining_clauses)"""
        assignment = {}  # var -> bool (True = positive)
        
        while True:
            # Build literal occurrence counts
            # For each variable, count how many clauses contain it positively and negatively
            pos_count = [0] * n
            neg_count = [0] * n
            
            for clause in converted_clauses:
                for lit in clause:
                    if lit >= 0:
                        pos_count[lit] += 1
                    else:
                        neg_count[~lit] += 1  # ~lit gives back the var
            
            # Find pure literals
            pure_found = False
            for var in range(n):
                if var in assignment:
                    continue
                if pos_count[var] == 0 and neg_count[var] > 0:
                    # Only appears negatively -> set to False (negative)
                    assignment[var] = False
                    pure_found = True
                elif neg_count[var] == 0 and pos_count[var] > 0:
                    # Only appears positively -> set to True (positive)
                    assignment[var] = True
                    pure_found = True
            
            if not pure_found:
                break
            
            # Remove clauses satisfied by the new assignment
            new_clauses = []
            for clause in converted_clauses:
                sat = False
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        is_pos = lit >= 0
                        if assignment[var] == is_pos:
                            sat = True
                            break
                if not sat:
                    new_clauses.append(clause)
            
            converted_clauses = new_clauses
            # Rebuild var_to_clauses
            var_to_clauses.clear()
            for ci, clause in enumerate(converted_clauses):
                for lit in clause:
                    var = abs(lit)
                    var_to_clauses[var].append(ci)
        
        return assignment, converted_clauses
    
    fixed_assignment, remaining_clauses = preprocess()
    
    # If no clauses remain, we're done
    if not remaining_clauses:
        result = [fixed_assignment.get(i, False) for i in range(n)]
        return result
    
    # Now apply WalkSAT on remaining clauses
    # Variables not fixed get random initial assignment
    rng = random.Random(42)
    
    # Track which variables are free
    free_vars = [i for i in range(n) if i not in fixed_assignment]
    
    # Initial random assignment for free variables
    current_assignment = {}
    for var in free_vars:
        current_assignment[var] = rng.random() < 0.5
    
    # Combine: full assignment for checking
    def get_assignment(var):
        if var in fixed_assignment:
            return fixed_assignment[var]
        return current_assignment[var]
    
    def is_satisfied(clause):
        for lit in clause:
            var = abs(lit)
            is_pos = lit >= 0
            if get_assignment(var) == is_pos:
                return True
        return False
    
    # Check if all satisfied
    def all_satisfied():
        for clause in remaining_clauses:
            if not is_satisfied(clause):
                return False
        return True
    
    if all_satisfied():
        result = [get_assignment(i) for i in range(n)]
        return result
    
    # WalkSAT with adaptive noise
    # Track unsatisfied clause count for efficiency
    # For each clause, track number of falsified literals (0 = satisfied)
    
    # Build clause state: for each clause, how many literals are false
    clause_false_count = [0] * len(remaining_clauses)
    for ci, clause in enumerate(remaining_clauses):
        fc = 0
        for lit in clause:
            var = abs(lit)
            is_pos = lit >= 0
            if get_assignment(var) != is_pos:
                fc += 1
        clause_false_count[ci] = fc
    
    unsat_clauses = [ci for ci in range(len(remaining_clauses)) if clause_false_count[ci] > 0]
    
    if not unsat_clauses:
        result = [get_assignment(i) for i in range(n)]
        return result
    
    # WalkSAT loop
    max_steps = 500000
    noise_start = 0.25
    noise_end = 0.05
    restart_threshold = 5000
    consecutive_no_improve = 0
    best_unsat = len(unsat_clauses)
    best_assignment_snapshot = None
    
    step = 0
    while step < max_steps and time.time() < deadline:
        if not unsat_clauses:
            break
        
        # Adaptive noise
        progress = min(step / max_steps, 1.0)
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Select an unsatisfied clause
        ci = rng.choice(unsat_clauses)
        clause = remaining_clauses[ci]
        
        # Score each literal in the clause
        # For a literal in an unsat clause, all literals are false
        # Flipping a variable will:
        #   - satisfy this clause (if the literal becomes true)
        #   - may falsify other clauses containing that variable
        best_var = None
        best_score = -float('inf')
        
        for lit in clause:
            var = abs(lit)
            if var in fixed_assignment:
                continue  # can't flip fixed variables
            is_pos = lit >= 0
            # Currently this literal is false (since clause is unsat)
            # If we flip var, this literal becomes true -> satisfies this clause
            # Count how many other clauses would be falsified
            gain = 1  # this clause becomes satisfied
            loss = 0
            for other_ci in var_to_clauses[var]:
                if other_ci == ci:
                    continue
                other_clause = remaining_clauses[other_ci]
                # Check if flipping var would falsify this other clause
                # The other clause is currently satisfied if clause_false_count[other_ci] == 0
                # We need to check if the literal of var in other_clause is currently true
                # and would become false after flip
                for olit in other_clause:
                    ovar = abs(olit)
                    if ovar == var:
                        ois_pos = olit >= 0
                        # Currently: get_assignment(var) == ois_pos? 
                        # If the literal is currently true, flipping makes it false
                        if get_assignment(var) == ois_pos:
                            loss += 1
                        break
            
            score = gain - loss + rng.random() * noise
            if score > best_score:
                best_score = score
                best_var = var
        
        if best_var is None:
            # All variables in clause are fixed, pick any free var
            break
        
        # Flip the variable
        old_val = current_assignment[best_var]
        current_assignment[best_var] = not old_val
        
        # Update clause false counts
        for ci2 in var_to_clauses[best_var]:
            clause2 = remaining_clauses[ci2]
            # Find the literal for best_var in this clause
            for lit2 in clause2:
                if abs(lit2) == best_var:
                    is_pos2 = lit2 >= 0
                    # Before flip: was this literal true?
                    was_true = old_val == is_pos2
                    # After flip: is this literal true?
                    is_true_now = (not old_val) == is_pos2
                    if was_true and not is_true_now:
                        clause_false_count[ci2] += 1
                        if clause_false_count[ci2] == 1:
                            # Newly unsatisfied
                            if ci2 not in unsat_clauses:
                                unsat_clauses.append(ci2)
                    elif not was_true and is_true_now:
                        clause_false_count[ci2] -= 1
                        if clause_false_count[ci2] == 0:
                            # Newly satisfied
                            unsat_clauses.remove(ci2)
                    break
        
        step += 1
        
        # Track best
        if len(unsat_clauses) < best_unsat:
            best_unsat = len(unsat_clauses)
            best_assignment_snapshot = dict(current_assignment)
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # Restart if stuck
        if consecutive_no_improve > restart_threshold:
            # Re-randomize
            for var in free_vars:
                current_assignment[var] = rng.random() < 0.5
            # Rebuild clause states
            for ci in range(len(remaining_clauses)):
                fc = 0
                for lit in remaining_clauses[ci]:
                    var = abs(lit)
                    is_pos = lit >= 0
                    if get_assignment(var) != is_pos:
                        fc += 1
                clause_false_count[ci] = fc
            unsat_clauses = [ci for ci in range(len(remaining_clauses)) if clause_false_count[ci] > 0]
            consecutive_no_improve = 0
    
    # Build result
    result = [get_assignment(i) for i in range(n)]
    return result
# EVOLVE-BLOCK-END
