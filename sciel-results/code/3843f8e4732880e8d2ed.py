# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based literals for internal use
    # Store as list of lists of (var, sign) where sign is 1 for positive, 0 for negative
    clause_data = []
    for clause in clauses:
        cd = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            cd.append((var, sign))
        clause_data.append(cd)
    
    m = len(clause_data)
    
    # Phase 1: Vectorized preprocessing
    # Build incidence structure for efficient updates
    # var_to_clauses[var] = list of (clause_idx, sign)
    var_to_clauses = defaultdict(list)
    for ci, cd in enumerate(clause_data):
        for var, sign in cd:
            var_to_clauses[var].append((ci, sign))
    
    assignment = [None] * n
    # Track clause satisfaction state: 0 = unsatisfied, 1 = satisfied, -1 = removed
    clause_state = [0] * m  # all unsatisfied initially
    
    # Unit propagation and pure literal elimination
    # Use a queue for unit clauses
    unit_queue = []
    
    # First pass: identify pure literals
    for var in range(n):
        if not var_to_clauses[var]:
            continue
        signs = set(sign for _, sign in var_to_clauses[var])
        if len(signs) == 1:
            # Pure literal
            sign = signs.pop()
            new_val = (sign == 1)
            if assignment[var] is None:
                assignment[var] = new_val
                # Propagate: mark clauses containing this var as satisfied
                for ci, s in var_to_clauses[var]:
                    if clause_state[ci] == 0:
                        clause_state[ci] = 1
    
    # Second pass: unit propagation
    changed = True
    while changed:
        changed = False
        for ci in range(m):
            if clause_state[ci] != 0:
                continue
            # Count unassigned literals in this clause
            unassigned = []
            for var, sign in clause_data[ci]:
                if assignment[var] is None:
                    unassigned.append((var, sign))
            if len(unassigned) == 0:
                # Empty clause - shouldn't happen for SAT instance
                pass
            elif len(unassigned) == 1:
                var, sign = unassigned[0]
                new_val = (sign == 1)
                if assignment[var] is None:
                    assignment[var] = new_val
                    changed = True
                    # Mark all clauses containing this var as satisfied
                    for ci2, s2 in var_to_clauses[var]:
                        if clause_state[ci2] == 0:
                            clause_state[ci2] = 1
    
    # Check if all clauses are satisfied
    all_sat = all(clause_state[ci] != 0 for ci in range(m))
    if all_sat:
        final_assignment = []
        rng = random.Random(0)
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment
    
    # Phase 2: WalkSAT on remaining unsatisfied clauses
    return _walksat_efficient(n, clause_data, assignment, clause_state, var_to_clauses, rng_seed=42)


def _walksat_efficient(n, clause_data, assignment, clause_state, var_to_clauses, rng_seed=42):
    """
    Efficient WalkSAT with delta tracking for O(1) unsat count updates.
    """
    rng = random.Random(rng_seed)
    
    # Initialize unassigned variables
    for var in range(n):
        if assignment[var] is None:
            assignment[var] = rng.random() < 0.5
    
    m = len(clause_data)
    
    # Build reverse index: for each clause, which variables it contains
    # And for each variable, which clauses it appears in (with sign)
    # var_to_clauses already exists, but let's also build clause_to_vars efficiently
    
    # Count unsatisfied clauses
    unsat_count = sum(1 for ci in range(m) if clause_state[ci] == 0)
    
    if unsat_count == 0:
        return assignment
    
    # Precompute for each clause: list of (var, sign)
    # Already in clause_data
    
    # For efficient gain calculation, we need to know for each variable in an unsat clause,
    # how many other unsat clauses would be satisfied by flipping it.
    
    max_flips = 100000
    noise_start = 0.3
    noise_end = 0.05
    
    # Track which clauses are unsatisfied for quick access
    unsat_clauses = set(ci for ci in range(m) if clause_state[ci] == 0)
    
    for flip in range(max_flips):
        if not unsat_clauses:
            break
        
        # Dynamic noise
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Choose an unsatisfied clause
        # Weight by inverse length to prefer shorter clauses
        unsat_list = list(unsat_clauses)
        weights = [1.0 / len(clause_data[ci]) for ci in unsat_list]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        chosen_ci = unsat_list[0]
        for idx, ci in enumerate(unsat_list):
            cumulative += weights[idx]
            if r <= cumulative:
                chosen_ci = ci
                break
        
        clause = clause_data[chosen_ci]
        
        # Choose variable to flip
        if rng.random() < noise:
            # Random flip
            var, sign = clause[rng.randrange(len(clause))]
        else:
            # Greedy: pick variable with max gain
            # Gain = number of unsat clauses that would become satisfied by flipping var
            best_gain = -1
            best_var = clause[0][0]
            for var, sign in clause:
                current_val = assignment[var]
                new_val = not current_val
                gain = 0
                # Count unsat clauses that contain var and would be satisfied
                for ci2, s2 in var_to_clauses[var]:
                    if clause_state[ci2] != 0:
                        continue
                    # If we set var to new_val, is this clause satisfied?
                    if new_val == (s2 == 1):
                        # Check if this clause would be satisfied (at least one literal true)
                        # Since it's currently unsat, all literals are false
                        # If we make this literal true, clause is satisfied
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            var = best_var
        
        # Flip the variable
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        # Update clause states: for each clause containing var, check if state changes
        for ci, sign in var_to_clauses[var]:
            if clause_state[ci] == 1:
                # Was satisfied, might become unsatisfied
                # Check if any other literal in this clause is now true
                still_sat = False
                for v, s in clause_data[ci]:
                    if v == var:
                        continue
                    if assignment[v] == (s == 1):
                        still_sat = True
                        break
                if not still_sat:
                    clause_state[ci] = 0
                    unsat_clauses.add(ci)
            elif clause_state[ci] == 0:
                # Was unsatisfied, might become satisfied
                # Check if the flipped literal is now true
                if new_val == (sign == 1):
                    clause_state[ci] = 1
                    unsat_clauses.discard(ci)
    
    return assignment


# EVOLVE-BLOCK-END
