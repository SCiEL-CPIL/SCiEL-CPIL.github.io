"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import sys

def solve(instance):
    start_time = time.time()
    time_limit = 290.0  # Safety margin (300s kill, leave buffer)
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Build clause-variable matrix A (m x n)
    # A[i, j] is the literal value for variable j in clause i:
    # 1 if +j, -1 if -j, 0 if not present
    A = np.zeros((m, n), dtype=np.int8)
    for i, c in enumerate(clauses):
        for l in c:
            var_idx = abs(l) - 1
            if l > 0:
                A[i, var_idx] = 1
            else:
                A[i, var_idx] = -1
    
    # Precompute clause lengths (number of literals in each clause)
    # For 3-SAT, this is always 3, but good to be general
    clause_lengths = np.sum(A != 0, axis=1)
    
    # Variable to clause indices (for efficient update, though vectorized approach might make this less critical for damage calc)
    # But we need to know which clauses contain a variable for the "random pick from clause" step
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                var_to_clauses[j].append(i)
    
    # Helper: check if assignment satisfies all clauses
    def is_satisfied(assignment):
        # assignment is numpy array of 1s and -1s
        # S = A @ assignment
        # Satisfies if all S > 0
        S = A.dot(assignment)
        return np.all(S > 0)
    
    def count_unsat(assignment):
        S = A.dot(assignment)
        return np.sum(S <= 0)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random()
    
    def walk_sat():
        nonlocal best_assignment, best_unsat_count
        
        # Random initial assignment: 1 or -1
        assignment = np.random.choice([1, -1], size=n)
        
        # Initial satisfaction scores
        S = A.dot(assignment)
        unsat_mask = S <= 0
        current_unsat = np.sum(unsat_mask)
        
        if current_unsat == 0:
            return (assignment == 1).astype(bool)
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = (assignment == 1).astype(bool)
        
        max_steps = 100000
        noise = 0.45
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            if current_unsat == 0:
                return (assignment == 1).astype(bool)
            
            # Pick a random unsat clause
            # Get indices of unsat clauses
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return (assignment == 1).astype(bool)
            
            clause_idx = rng.choice(unsat_indices)
            
            # Get variables in this clause
            # Find non-zero elements in A[clause_idx]
            clause_vars = np.where(A[clause_idx] != 0)[0]
            
            if len(clause_vars) == 0:
                # Should not happen for valid 3-SAT
                var_idx = rng.randint(0, n-1)
            else:
                # Calculate damage and gain for each variable in the clause
                # We want to flip a variable in this clause.
                # For each candidate var v in clause_vars:
                #   delta_S_c = -2 * A[c, v] * assignment[v]
                #   New S_c = S_c + delta_S_c
                #   Damage: clauses where S_old > 0 and S_new <= 0
                #   Gain: clauses where S_old <= 0 and S_new > 0
                #   Score = Gain - Damage (or similar)
                
                # Vectorized calculation for all vars in clause_vars
                # A_clause = A[clause_idx, clause_vars] (shape k)
                # x_clause = assignment[clause_vars] (shape k)
                
                A_clause = A[clause_idx, clause_vars]
                x_clause = assignment[clause_vars]
                
                # For each variable v, the change in S for ALL clauses is:
                # delta_S = -2 * A[:, v] * x_v
                # We only care about the clauses that contain v? No, we need global damage/gain.
                # However, flipping v only affects clauses containing v.
                # So for variable v at position j in clause_vars:
                #   affected_clauses = var_to_clauses[v]
                #   S_affected = S[affected_clauses]
                #   A_affected = A[affected_clauses, v]
                #   x_v = assignment[v]
                #   delta_S_affected = -2 * A_affected * x_v
                #   S_new_affected = S_affected + delta_S_affected
                #   damage_v = sum((S_affected > 0) & (S_new_affected <= 0))
                #   gain_v = sum((S_affected <= 0) & (S_new_affected > 0))
                #   score_v = gain_v - damage_v
                
                best_score = -float('inf')
                best_var_idx = -1
                
                for v in clause_vars:
                    affected = var_to_clauses[v]
                    if not affected:
                        continue
                    
                    S_aff = S[affected]
                    A_aff = A[affected, v]
                    x_v = assignment[v]
                    
                    delta_S = -2 * A_aff * x_v
                    S_new = S_aff + delta_S
                    
                    damage = np.sum((S_aff > 0) & (S_new <= 0))
                    gain = np.sum((S_aff <= 0) & (S_new > 0))
                    
                    score = gain - damage
                    # Add small random noise to break ties
                    score += rng.random() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_var_idx = v
                
                if best_var_idx == -1:
                    best_var_idx = rng.choice(clause_vars)
                
                # Decide to flip best_var or random var based on noise
                if rng.random() < noise:
                    var_idx = rng.choice(clause_vars)
                else:
                    var_idx = best_var_idx
            
            # Perform flip
            assignment[var_idx] = -assignment[var_idx]
            
            # Update S and unsat_mask incrementally
            # S changes only for clauses containing var_idx
            affected = var_to_clauses[var_idx]
            A_aff = A[affected, var_idx]
            delta_S = -2 * A_aff * assignment[var_idx] # Note: assignment is already flipped
            # Wait, delta is based on OLD assignment.
            # Let's recalculate properly.
            # Old assignment was x. New is -x.
            # Delta = A * (New - Old) = A * (-x - x) = -2 A x.
            # So we need the OLD x. But we already flipped.
            # So Old x = -assignment[var_idx].
            old_x = -assignment[var_idx]
            delta_S = -2 * A_aff * old_x
            
            S[affected] += delta_S
            
            # Update unsat_mask
            old_unsat_aff = unsat_mask[affected]
            new_unsat_aff = S[affected] <= 0
            
            # Update count
            current_unsat = current_unsat - np.sum(old_unsat_aff & ~new_unsat_aff) + np.sum(~old_unsat_aff & new_unsat_aff)
            
            unsat_mask[affected] = new_unsat_aff
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = (assignment == 1).astype(bool)
            
            # Adaptive noise
            if step % 500 == 0 and step > 0:
                if current_unsat >= best_unsat_count:
                    noise = min(0.6, noise + 0.02)
                else:
                    noise = max(0.3, noise - 0.01)
        
        return None
    
    # Main loop with restarts
    restarts = 0
    max_restarts = 1000
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        result = walk_sat()
        if result is not None:
            return result
        restarts += 1
    
    if best_assignment:
        return best_assignment
    
    # Final fallback
    return np.random.choice([True, False], size=n)

# EVOLVE-BLOCK-END
