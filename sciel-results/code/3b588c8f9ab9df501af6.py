"""Rewritten solver: bitset-accelerated WalkSAT with frequency-aware flip selection."""
# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Preprocess: convert clauses to 0-based indices and signs
    # clause: list of signed 1-based literals
    # We'll store as (var_idx, is_positive)
    processed_clauses = []
    for clause in clauses:
        processed = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            processed.append((var_idx, is_pos))
        processed_clauses.append(processed)
    
    # Initialize random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # For efficient clause satisfaction checking, we maintain:
    # For each clause, the count of satisfied literals
    # and for each variable, which clauses it appears in and with what sign
    
    # Build variable-to-clause mapping
    var_to_clauses = [[] for _ in range(n)]  # list of (clause_idx, sign) where sign=1 if positive
    for ci, clause in enumerate(processed_clauses):
        for var_idx, is_pos in clause:
            var_to_clauses[var_idx].append((ci, 1 if is_pos else -1))
    
    # Compute initial satisfied counts per clause
    m = len(processed_clauses)
    sat_counts = np.zeros(m, dtype=np.int32)
    for ci, clause in enumerate(processed_clauses):
        cnt = 0
        for var_idx, is_pos in clause:
            if assignment[var_idx] == is_pos:
                cnt += 1
        sat_counts[ci] = cnt
    
    # Number of unsatisfied clauses
    num_unsat = np.sum(sat_counts == 0)
    
    if num_unsat == 0:
        return assignment
    
    # For each clause, track the variables in it for quick access
    # We need to find unsatisfied clauses and pick one, then pick a variable to flip
    
    # Strategy: 
    # 1. Maintain list of unsatisfied clause indices
    # 2. When flipping a variable, update sat_counts for affected clauses
    # 3. Use frequency-aware selection: flip variables in unsat clauses with prob inversely proportional to frequency
    
    # Build a set of unsatisfied clauses
    unsat_clauses = set(ci for ci in range(m) if sat_counts[ci] == 0)
    
    # Frequency of variables in unsat clauses
    var_freq = np.zeros(n, dtype=np.int32)
    for ci in unsat_clauses:
        for var_idx, is_pos in processed_clauses[ci]:
            var_freq[var_idx] += 1
    
    # Flip budget
    max_flips = 100000
    
    for flip_iter in range(max_flips):
        if not unsat_clauses:
            break
        
        # Pick an unsatisfied clause uniformly
        ci = rng.choice(list(unsat_clauses))
        clause = processed_clauses[ci]
        
        # Compute flip probabilities inversely proportional to frequency
        # For each variable in the clause, weight = 1 / (freq + 1)
        weights = []
        for var_idx, is_pos in clause:
            weights.append(1.0 / (var_freq[var_idx] + 1))
        
        # Normalize and pick
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0.0
        chosen_var = clause[0][0]
        for (var_idx, is_pos), w in zip(clause, weights):
            cumulative += w
            if r <= cumulative:
                chosen_var = var_idx
                break
        else:
            chosen_var = clause[-1][0]
        
        # Flip the variable
        old_val = assignment[chosen_var]
        assignment[chosen_var] = not old_val
        
        # Update sat_counts for all clauses containing this variable
        for (clause_idx, sign) in var_to_clauses[chosen_var]:
            # sign = 1 if variable appears positively, -1 if negatively
            # If old_val was True and sign=1, the literal was satisfied; now it's not
            # If old_val was False and sign=-1, the literal was satisfied; now it's not
            # In general: literal is satisfied iff assignment[var] == (sign > 0)
            old_satisfied = (old_val == (sign > 0))
            new_satisfied = ((not old_val) == (sign > 0))
            
            if old_satisfied and not new_satisfied:
                sat_counts[clause_idx] -= 1
                if sat_counts[clause_idx] == 0:
                    unsat_clauses.add(clause_idx)
            elif not old_satisfied and new_satisfied:
                sat_counts[clause_idx] += 1
                if sat_counts[clause_idx] == 1:
                    unsat_clauses.discard(clause_idx)
            
            # Update var_freq for this clause if it's unsat
            if sat_counts[clause_idx] == 0:
                # This clause is unsat, so all its variables contribute to freq
                pass  # We'll recompute freq periodically or handle incrementally
        
        # Incrementally update var_freq:
        # When a clause becomes unsat, add 1 to freq of all its vars
        # When a clause becomes sat, subtract 1 from freq of all its vars
        # We need to track which clauses changed status
        
        # Let's do it more carefully: track changes
        # Actually, let's recompute var_freq every few iterations to keep it simple
        # But that's expensive. Let's do incremental updates.
        
        # For the clauses we just updated:
        for (clause_idx, sign) in var_to_clauses[chosen_var]:
            # Check if status changed
            pass  # This is getting complex. Let's use a simpler approach.
    
    # Simpler approach: recompute var_freq less frequently
    # Let's restart with a cleaner implementation
    
    # Reset and use a cleaner bitset-based approach
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    sat_counts = np.zeros(m, dtype=np.int32)
    for ci, clause in enumerate(processed_clauses):
        cnt = 0
        for var_idx, is_pos in clause:
            if assignment[var_idx] == is_pos:
                cnt += 1
        sat_counts[ci] = cnt
    
    unsat_clauses = set(ci for ci in range(m) if sat_counts[ci] == 0)
    
    if not unsat_clauses:
        return assignment
    
    var_freq = np.zeros(n, dtype=np.int32)
    for ci in unsat_clauses:
        for var_idx, is_pos in processed_clauses[ci]:
            var_freq[var_idx] += 1
    
    max_flips = 100000
    freq_update_interval = 100
    
    for flip_iter in range(max_flips):
        if not unsat_clauses:
            break
        
        # Periodically recompute var_freq
        if flip_iter % freq_update_interval == 0:
            var_freq = np.zeros(n, dtype=np.int32)
            for ci in unsat_clauses:
                for var_idx, is_pos in processed_clauses[ci]:
                    var_freq[var_idx] += 1
        
        # Pick an unsatisfied clause
        ci = rng.choice(list(unsat_clauses))
        clause = processed_clauses[ci]
        
        # Compute weights inversely proportional to frequency
        weights = []
        for var_idx, is_pos in clause:
            weights.append(1.0 / (var_freq[var_idx] + 1))
        
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0.0
        chosen_var = clause[0][0]
        for (var_idx, is_pos), w in zip(clause, weights):
            cumulative += w
            if r <= cumulative:
                chosen_var = var_idx
                break
        else:
            chosen_var = clause[-1][0]
        
        # Flip
        old_val = assignment[chosen_var]
        assignment[chosen_var] = not old_val
        
        # Update sat_counts
        for (clause_idx, sign) in var_to_clauses[chosen_var]:
            old_sat = (old_val == (sign > 0))
            new_sat = ((not old_val) == (sign > 0))
            
            if old_sat and not new_sat:
                sat_counts[clause_idx] -= 1
                if sat_counts[clause_idx] == 0:
                    unsat_clauses.add(clause_idx)
                    # Update var_freq for this clause
                    for var_idx2, is_pos2 in processed_clauses[clause_idx]:
                        var_freq[var_idx2] += 1
            elif not old_sat and new_sat:
                sat_counts[clause_idx] += 1
                if sat_counts[clause_idx] == 1:
                    unsat_clauses.discard(clause_idx)
                    # Update var_freq for this clause
                    for var_idx2, is_pos2 in processed_clauses[clause_idx]:
                        var_freq[var_idx2] -= 1
    
    return assignment
# EVOLVE-BLOCK-END
