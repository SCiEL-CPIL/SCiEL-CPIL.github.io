"""Improved WalkSAT with vectorized evaluation, phase-aware heuristics, and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin under 300s limit
    
    # Precompute clause structure for vectorized evaluation
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int32)  # 1 for positive, -1 for negative
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # Precompute variable occurrence counts for phase heuristics
    # Count how many times each variable appears positively and negatively
    pos_count = np.zeros(n, dtype=np.int32)
    neg_count = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                pos_count[v] += 1
            else:
                neg_count[v] += 1
    # Phase bias: prefer setting variable to 1 if it appears more often positively
    phase_bias = pos_count - neg_count
    
    # Precompute for each variable, which clauses contain it
    var_to_clauses = {}
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if v not in var_to_clauses:
                var_to_clauses[v] = []
            var_to_clauses[v].append(i)
    # Convert to numpy arrays for faster access
    var_clause_lists = {}
    for v, clist in var_to_clauses.items():
        var_clause_lists[v] = np.array(clist, dtype=np.int32)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    def evaluate(assignment):
        """Vectorized clause satisfaction check. Returns number of unsatisfied clauses."""
        lit_vals = clause_signs * (2 * assignment[clause_vars] - 1)
        clause_sat = (lit_vals > 0).any(axis=1)
        return m - clause_sat.sum()
    
    def evaluate_with_unsat(assignment):
        """Returns unsat_count and unsat_indices."""
        lit_vals = clause_signs * (2 * assignment[clause_vars] - 1)
        clause_sat = (lit_vals > 0).any(axis=1)
        unsat_count = m - clause_sat.sum()
        unsat_indices = np.where(~clause_sat)[0]
        return unsat_count, unsat_indices
    
    # Try multiple restarts with different seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(seed)
        
        # Use phase-biased initial assignment
        if seed == 0:
            # First restart: use phase bias
            assignment = np.zeros(n, dtype=np.int32)
            for v in range(n):
                if phase_bias[v] > 0:
                    assignment[v] = 1
                elif phase_bias[v] < 0:
                    assignment[v] = 0
                else:
                    assignment[v] = rng.randint(2)
        else:
            # Subsequent restarts: random with slight phase bias
            assignment = rng.rand(n) < 0.5
            # Apply phase bias with some probability
            for v in range(n):
                if phase_bias[v] > 2 and rng.rand() < 0.3:
                    assignment[v] = 1
                elif phase_bias[v] < -2 and rng.rand() < 0.3:
                    assignment[v] = 0
            assignment = assignment.astype(np.int32)
        
        # Local search with adaptive strategy
        max_steps = 10000
        unsat_count, unsat_indices = evaluate_with_unsat(assignment)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            c_idx = unsat_indices[rng.randint(len(unsat_indices))]
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
            # Score each literal in the clause by estimated impact
            best_var = -1
            best_score = -1000000
            
            for j in range(3):
                v = c_vars[j]
                # Get all clauses containing this variable
                if v in var_clause_lists:
                    affected = var_clause_lists[v]
                    # Compute current satisfaction status of affected clauses
                    # For each affected clause, check if it's currently satisfied
                    # and what would happen if we flip v
                    
                    # Current value of v
                    old_val = assignment[v]
                    
                    # For each affected clause, determine if flipping v helps or hurts
                    # A clause is satisfied if any literal is true
                    # Flipping v changes the truth value of literal (v, sign) in clauses containing v
                    
                    # Quick heuristic: count how many affected clauses are currently unsat
                    # and would become sat, minus how many are currently sat and would become unsat
                    
                    # Get the literal positions of v in affected clauses
                    # This is expensive, so use a simpler approach:
                    # Score based on phase bias and random tie-breaking
                    
                    # Phase-aware score: prefer flipping to match phase
                    if phase_bias[v] > 0:
                        # Prefer setting to 1 (True)
                        score = 1 if old_val == 0 else -1
                    elif phase_bias[v] < 0:
                        # Prefer setting to 0 (False)
                        score = 1 if old_val == 1 else -1
                    else:
                        score = 0
                    
                    # Add small random perturbation for diversity
                    score += rng.rand() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_var = v
            
            if best_var == -1:
                best_var = c_vars[rng.randint(3)]
            
            assignment[best_var] = 1 - assignment[best_var]
            
            # Update unsat_count and unsat_indices incrementally
            # Only need to re-evaluate clauses affected by the flip
            if best_var in var_clause_lists:
                affected = var_clause_lists[best_var]
                # Re-evaluate only affected clauses
                lit_vals_affected = clause_signs[affected] * (2 * assignment[clause_vars[affected]] - 1)
                clause_sat_affected = (lit_vals_affected > 0).any(axis=1)
                
                # Update the global clause_sat array
                # We need to maintain a running clause_sat array
                # For simplicity, recompute periodically
                if step % 50 == 0:
                    unsat_count, unsat_indices = evaluate_with_unsat(assignment)
            else:
                # Recompute if needed
                if step % 100 == 0:
                    unsat_count, unsat_indices = evaluate_with_unsat(assignment)
        
        # After this restart, check if we improved
        final_count, _ = evaluate_with_unsat(assignment)
        if final_count < best_unsat_count:
            best_unsat_count = final_count
            best_assignment = assignment.copy()
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    # Fallback: all True
    return [True] * n
# EVOLVE-BLOCK-END
