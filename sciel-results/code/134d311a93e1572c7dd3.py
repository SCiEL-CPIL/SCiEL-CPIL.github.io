"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock limit, use 325s to be safe
    deadline = time.time() + 325.0
    
    # Preprocess clauses into arrays for fast access
    # literals: array of shape (m, 3) with 1-based signed literals
    # vars_idx: array of shape (m, 3) with 0-based variable indices
    # signs: array of shape (m, 3) with +1 or -1
    lits = np.array(clauses, dtype=np.int32)
    vars_idx = np.abs(lits) - 1
    signs = np.sign(lits).astype(np.int32)
    
    # Helper to check if a clause is satisfied given assignment array
    # assignment is a numpy array of 0s and 1s
    def is_satisfied(assignment, clause_idx):
        # For each literal in clause: assignment[var] == 1 if sign>0 else 0
        # satisfied if any literal is true
        vals = assignment[vars_idx[clause_idx]]
        needed = (signs[clause_idx] > 0).astype(np.int32)
        return np.any(vals == needed)
    
    def count_unsat(assignment):
        # Returns list of unsatisfied clause indices
        # Vectorized: for each clause, check if any literal is satisfied
        # assignment shape (n,), vars_idx shape (m,3), signs shape (m,3)
        # lit_val = assignment[vars_idx] -> shape (m,3)
        lit_vals = assignment[vars_idx]  # (m, 3)
        needed = (signs > 0).astype(np.int32)  # (m, 3)
        # A literal is true if lit_val == needed
        lit_true = (lit_vals == needed)  # (m, 3) boolean
        # Clause is satisfied if any literal is true
        clause_sat = np.any(lit_true, axis=1)  # (m,)
        return np.where(~clause_sat)[0]
    
    def flip_var(assignment, var_idx, new_val):
        assignment[var_idx] = new_val
    
    def walk_sat(assignment, max_steps, rng):
        """WalkSAT with vectorized unsat tracking."""
        unsat_idx = count_unsat(assignment)
        steps = 0
        while len(unsat_idx) > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            # Pick a random unsatisfied clause
            clause_i = rng.choice(unsat_idx)
            # Pick a literal in that clause to flip
            # With some probability, pick the literal that would satisfy the clause
            # (i.e., the one that is currently false)
            # Actually in WalkSAT, we pick a random literal from the clause and flip it
            lit_pos = rng.randint(0, 2)
            var = vars_idx[clause_i, lit_pos]
            sign = signs[clause_i, lit_pos]
            # Flip: if sign > 0, we want assignment[var] to become 1 (to satisfy this lit)
            # but WalkSAT just flips
            assignment[var] = 1 - assignment[var]
            steps += 1
            
            # Update unsat set incrementally is complex; do it every few steps or use efficient check
            # For 50 vars, m is likely ~100-300, so full check is fast
            if steps % 10 == 0 or steps < 100:
                unsat_idx = count_unsat(assignment)
            else:
                # Just check if we might have fixed or broken things
                # For efficiency, do full check less often but we need accuracy
                # Actually for 50 vars, full check is very fast, do it every time
                unsat_idx = count_unsat(assignment)
        return assignment, len(unsat_idx) == 0
    
    def local_search(assignment, rng, max_steps):
        """Enhanced local search with clause weighting (MiniSAT-like)."""
        unsat_idx = count_unsat(assignment)
        if len(unsat_idx) == 0:
            return assignment, True
        
        # Track clause weights for biasing
        clause_weights = np.ones(m, dtype=np.float64)
        
        steps = 0
        while len(unsat_idx) > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            
            # Pick unsatisfied clause with probability proportional to weight
            total_weight = clause_weights[unsat_idx].sum()
            probs = clause_weights[unsat_idx] / total_weight
            clause_i = unsat_idx[rng.choice(len(unsat_idx), p=probs)]
            
            # Pick literal to flip with probability proportional to variable occurrence weight
            # For simplicity, pick random literal
            lit_pos = rng.randint(0, 2)
            var = vars_idx[clause_i, lit_pos]
            
            # Compute delta: how many clauses would be satisfied vs unsatisfied
            old_val = assignment[var]
            new_val = 1 - old_val
            
            # Find all clauses containing this variable
            # Precompute var_to_clauses for efficiency
            # For now, just flip and recount
            assignment[var] = new_val
            steps += 1
            unsat_idx = count_unsat(assignment)
            
            # Update weights: increase weight of clauses that became unsatisfied
            # This is a simplification; proper MiniSAT tracks this more carefully
            if len(unsat_idx) > 0:
                for ci in unsat_idx:
                    clause_weights[ci] *= 1.1
                # Decay others slightly
                clause_weights[~np.isin(np.arange(m), unsat_idx)] *= 0.99
        
        return assignment, len(unsat_idx) == 0
    
    # Precompute variable-to-clause mapping for potential use
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            v = vars_idx[ci, li]
            var_to_clauses[v].append(ci)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    rng = random.Random(42)
    
    # Multiple restarts with different initializations
    num_restarts = 50
    steps_per_restart = 10000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int32)
        
        # Check if already satisfied
        unsat_idx = count_unsat(assignment)
        if len(unsat_idx) == 0:
            return assignment.tolist()
        
        # Try WalkSAT
        result, success = walk_sat(assignment.copy(), steps_per_restart, rng)
        if success:
            return result.tolist()
        
        cur_unsat = len(count_unsat(result))
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = result.copy()
        
        # Try enhanced local search from the best point
        if time.time() < deadline - 10.0:
            result2, success2 = local_search(result.copy(), rng, steps_per_restart // 2)
            if success2:
                return result2.tolist()
            cur_unsat2 = len(count_unsat(result2))
            if cur_unsat2 < best_unsat_count:
                best_unsat_count = cur_unsat2
                best_assignment = result2.copy()
        
        # Try from random again with different params
        if time.time() < deadline - 10.0 and restart % 2 == 1:
            assignment2 = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int32)
            result3, success3 = walk_sat(assignment2, steps_per_restart, rng)
            if success3:
                return result3.tolist()
            cur_unsat3 = len(count_unsat(result3))
            if cur_unsat3 < best_unsat_count:
                best_unsat_count = cur_unsat3
                best_assignment = result3.copy()
    
    # If we have a best assignment, return it
    if best_assignment is not None:
        return best_assignment.tolist()
    
    # Fallback: return random
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
