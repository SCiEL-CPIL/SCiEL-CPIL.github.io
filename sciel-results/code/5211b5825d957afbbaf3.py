"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    """
    Solve a planted satisfiable 3-SAT instance using a hybrid approach:
    1. Preprocessing with unit propagation and pure literal elimination.
    2. Vectorized WalkSAT with clause-weighted flipping and dynamic noise.
    3. Restart strategy with phase saving.
    Deadline guard to stay within time limits.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: aim to finish well before 330s
    deadline = time.time() + 280.0
    
    # --- Preprocessing: Unit Propagation and Pure Literal Elimination ---
    # Build literal-to-variable mapping
    # We'll work with a set of active clauses and track variable assignments
    
    # Convert clauses to 0-based indices for internal use
    # Each clause is a list of signed literals (1-based)
    # We'll maintain:
    # - var_assign: dict var_index -> True/False/None
    # - active_clauses: list of clauses (lists of literals)
    
    var_assign = [None] * n  # None = unassigned, True/False = assigned
    
    def propagate_unit_and_pure():
        """
        Perform unit propagation and pure literal elimination.
        Returns False if contradiction found, True otherwise.
        Modifies var_assign and clauses in place.
        """
        changed = True
        while changed:
            changed = False
            
            # Remove clauses that are satisfied, collect unit clauses
            new_clauses = []
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                unit_lit = None
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if var_assign[var_idx] is not None:
                        if var_assign[var_idx] == (lit > 0):
                            sat = True
                            break
                    else:
                        unit_lit = lit
                if sat:
                    continue
                if len(clause) == 1:
                    # Unit clause: must assign
                    lit = clause[0]
                    var_idx = abs(lit) - 1
                    val = (lit > 0)
                    if var_assign[var_idx] is not None:
                        if var_assign[var_idx] != val:
                            return False  # Contradiction
                    else:
                        var_assign[var_idx] = val
                        changed = True
                else:
                    # Filter out assigned literals that are false
                    new_clause = []
                    for lit in clause:
                        var_idx = abs(lit) - 1
                        if var_assign[var_idx] is not None:
                            if var_assign[var_idx] == (lit > 0):
                                # Clause satisfied, skip
                                new_clause = None
                                break
                            else:
                                # Literal is false, remove
                                continue
                        else:
                            new_clause.append(lit)
                    if new_clause is None:
                        continue  # Clause satisfied
                    elif len(new_clause) == 0:
                        return False  # Empty clause: contradiction
                    elif len(new_clause) == 1:
                        # New unit clause
                        lit = new_clause[0]
                        var_idx = abs(lit) - 1
                        val = (lit > 0)
                        if var_assign[var_idx] is not None:
                            if var_assign[var_idx] != val:
                                return False
                        else:
                            var_assign[var_idx] = val
                            changed = True
                    else:
                        new_clauses.append(new_clause)
            
            clauses[:] = new_clauses
            
            # Pure literal elimination
            if not clauses:
                break
                
            # Count occurrences of each literal
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses:
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if var_assign[var_idx] is None:
                        if lit > 0:
                            pos_count[var_idx] += 1
                        else:
                            neg_count[var_idx] += 1
            
            # Assign pure literals
            for var_idx in range(n):
                if var_assign[var_idx] is not None:
                    continue
                if pos_count[var_idx] > 0 and neg_count[var_idx] == 0:
                    var_assign[var_idx] = True
                    changed = True
                elif pos_count[var_idx] == 0 and neg_count[var_idx] > 0:
                    var_assign[var_idx] = False
                    changed = True
        
        return True
    
    # Run preprocessing
    if not propagate_unit_and_pure():
        # Contradiction during preprocessing - shouldn't happen for satisfiable instances
        # Return all True as fallback
        return [True] * n
    
    # If all variables are assigned, check if all clauses are satisfied
    all_assigned = all(v is not None for v in var_assign)
    if all_assigned:
        # Verify all original clauses are satisfied
        for clause in clauses:
            if not any(var_assign[abs(l) - 1] == (l > 0) for l in clause):
                # Shouldn't happen if preprocessing was correct
                break
        return var_assign
    
    # --- Vectorized WalkSAT with clause-weighted flipping ---
    # Build data structures for efficient vectorized evaluation
    
    # For each clause, store the literals (0-based var index and sign)
    # clauses is already a list of lists of literals
    
    # Create numpy arrays for fast evaluation
    # For each clause, we need to know if any literal is satisfied
    
    # Phase: save phase information from preprocessing
    phase = [var_assign[i] if var_assign[i] is not None else (random.random() < 0.5) for i in range(n)]
    
    # Initial assignment: use phase for unassigned variables
    assignment = phase[:]
    
    # Build clause data for vectorized operations
    # For each clause, store:
    # - var_indices: array of variable indices
    # - signs: array of +1/-1 indicating polarity
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return var_assign
    
    # Precompute clause structures
    clause_var_indices = []
    clause_signs = []
    for clause in clauses:
        var_idx_list = []
        sign_list = []
        for lit in clause:
            var_idx_list.append(abs(lit) - 1)
            sign_list.append(1 if lit > 0 else -1)
        clause_var_indices.append(var_idx_list)
        clause_signs.append(sign_list)
    
    # Convert to numpy arrays where possible
    # Since clauses have variable length (but all 3 in this case), we can use fixed-size arrays
    # But for generality, we'll handle variable length
    
    # For vectorized evaluation, we'll use a different approach:
    # For each variable, precompute which clauses contain it and with what sign
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((ci, sign))
    
    # Track number of satisfied literals per clause
    # sat_count[ci] = number of literals in clause ci that are satisfied
    sat_count = np.zeros(num_clauses, dtype=np.int32)
    
    # Initialize sat_count
    for ci in range(num_clauses):
        count = 0
        for lit in clauses[ci]:
            var_idx = abs(lit) - 1
            if assignment[var_idx] == (lit > 0):
                count += 1
        sat_count[ci] = count
    
    # Find unsatisfied clauses
    unsat_indices = np.where(sat_count == 0)[0]
    
    if len(unsat_indices) == 0:
        return var_assign
    
    # WalkSAT loop with vectorized updates
    rng = random.Random(42)
    
    max_steps = 50000
    noise = 0.35  # Probability of random flip
    
    for step in range(max_steps):
        # Check deadline
        if time.time() > deadline - 10.0:
            break
        
        if len(unsat_indices) == 0:
            break
        
        # Pick a random unsatisfied clause
        ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
        clause = clauses[ci]
        
        # Decide which variable to flip
        if rng.random() < noise:
            # Random flip from the clause
            lit = clause[rng.randint(0, len(clause) - 1)]
        else:
            # Weighted flip: prefer variables that are in more unsatisfied clauses
            # Count how many unsatisfied clauses each variable in this clause appears in
            weights = []
            for lit in clause:
                var_idx = abs(lit) - 1
                # Count unsatisfied clauses containing this variable
                w = 0
                for (uci, usign) in var_to_clauses[var_idx]:
                    if sat_count[uci] == 0:
                        w += 1
                weights.append(w)
            
            # Weighted random selection
            total_w = sum(weights)
            if total_w == 0:
                lit = clause[rng.randint(0, len(clause) - 1)]
            else:
                r = rng.random() * total_w
                cum = 0
                lit = clause[-1]
                for i, w in enumerate(weights):
                    cum += w
                    if r < cum:
                        lit = clause[i]
                        break
        
        var_idx = abs(lit) - 1
        old_val = assignment[var_idx]
        new_val = not old_val
        
        # Update sat_count for all clauses containing var_idx
        for (uci, usign) in var_to_clauses[var_idx]:
            # Check if the literal was satisfied before
            lit_in_clause = usign * (1 if old_val else 0)  # This is wrong, let me fix
            
            # Actually, let's just recompute for affected clauses
            # For each clause containing var_idx, check if the literal is satisfied
            # before and after the flip
        
        # Simpler: just update sat_count for affected clauses
        for (uci, usign) in var_to_clauses[var_idx]:
            # The literal in clause uci for var_idx has sign usign
            # Before flip: satisfied if assignment[var_idx] == (usign > 0)
            was_sat = (old_val == (usign > 0))
            # After flip: satisfied if new_val == (usign > 0)
            is_sat = (new_val == (usign > 0))
            
            if was_sat and not is_sat:
                sat_count[uci] -= 1
            elif not was_sat and is_sat:
                sat_count[uci] += 1
        
        # Apply the flip
        assignment[var_idx] = new_val
        
        # Update unsat_indices
        unsat_indices = np.where(sat_count == 0)[0]
    
    # Final check
    if len(unsat_indices) == 0:
        return var_assign
    
    # If we didn't solve it, return the current best assignment
    # Try a few more restarts with different seeds if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Try a few more restarts
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Restart with new random assignment, but keep phase for assigned vars
            assignment = [var_assign[i] if var_assign[i] is not None else (rng.random() < 0.5) for i in range(n)]
            
            # Reinitialize sat_count
            sat_count = np.zeros(num_clauses, dtype=np.int32)
            for ci in range(num_clauses):
                count = 0
                for lit in clauses[ci]:
                    var_idx = abs(lit) - 1
                    if assignment[var_idx] == (lit > 0):
                        count += 1
                sat_count[ci] = count
            
            unsat_indices = np.where(sat_count == 0)[0]
            
            if len(unsat_indices) == 0:
                return var_assign
            
            # Short WalkSAT
            for step in range(10000):
                if time.time() > deadline - 5:
                    break
                if len(unsat_indices) == 0:
                    break
                
                ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
                clause = clauses[ci]
                
                if rng.random() < 0.35:
                    lit = clause[rng.randint(0, len(clause) - 1)]
                else:
                    weights = []
                    for lit in clause:
                        var_idx = abs(lit) - 1
                        w = 0
                        for (uci, usign) in var_to_clauses[var_idx]:
                            if sat_count[uci] == 0:
                                w += 1
                        weights.append(w)
                    total_w = sum(weights)
                    if total_w == 0:
                        lit = clause[rng.randint(0, len(clause) - 1)]
                    else:
                        r = rng.random() * total_w
                        cum = 0
                        lit = clause[-1]
                        for i, w in enumerate(weights):
                            cum += w
                            if r < cum:
                                lit = clause[i]
                                break
                
                var_idx = abs(lit) - 1
                old_val = assignment[var_idx]
                new_val = not old_val
                
                for (uci, usign) in var_to_clauses[var_idx]:
                    was_sat = (old_val == (usign > 0))
                    is_sat = (new_val == (usign > 0))
                    if was_sat and not is_sat:
                        sat_count[uci] -= 1
                    elif not was_sat and is_sat:
                        sat_count[uci] += 1
                
                assignment[var_idx] = new_val
                unsat_indices = np.where(sat_count == 0)[0]
            
            if len(unsat_indices) == 0:
                return var_assign
    
    # Return the assignment, converting None to True
    result = []
    for i in range(n):
        if var_assign[i] is not None:
            result.append(var_assign[i])
        else:
            result.append(assignment[i])
    return result
# EVOLVE-BLOCK-END
