# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    # Convert clauses to numpy array for vectorized operations
    # literals are 1-based signed
    m = len(clauses)
    if m == 0:
        return [False] * n
        
    # clauses_np: shape (m, 3)
    # If a clause has fewer than 3 literals, pad with 0 (dummy)
    clauses_np = np.zeros((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            if j < 3:
                clauses_np[i, j] = lit
    
    # Helper to get variable index (0-based) and sign from literal
    # We'll work with 0-based variables internally for numpy indexing
    
    def get_unsat_clauses(assign):
        """
        assign: np.array of bools (size n)
        Returns: indices of unsatisfied clauses
        """
        # Expand assign to match literal signs
        # For each clause, check if any literal is true
        # lit > 0: var is True
        # lit < 0: var is False
        # lit == 0: dummy (always false)
        
        # Create a boolean array for all literals in all clauses
        # var_idx = abs(lit) - 1
        # is_true = (lit > 0 and assign[var_idx]) or (lit < 0 and not assign[var_idx])
        
        # Vectorized check
        # vars_0based = np.abs(clauses_np) - 1
        # signs = np.sign(clauses_np)
        
        # We need to index assign with vars_0based
        # But assign is 1D. We can use advanced indexing.
        
        # Handle 0 literals (dummy)
        vars_0based = np.abs(clauses_np) - 1
        # Replace -1 (from 0) with 0, we'll mask them later
        vars_0based[clauses_np == 0] = 0
        
        # Get current values of variables for each literal position
        # val = assign[vars_0based] -> shape (m, 3)
        vals = assign[vars_0based]
        
        # Determine if literal is satisfied
        # lit > 0: val must be True
        # lit < 0: val must be False
        # lit == 0: False
        
        pos_sat = (clauses_np > 0) & vals
        neg_sat = (clauses_np < 0) & (~vals)
        dummy_sat = (clauses_np == 0) # Always False
        
        sat_mask = pos_sat | neg_sat | dummy_sat
        
        # Clause is satisfied if ANY literal is satisfied
        clause_sat = sat_mask.any(axis=1)
        
        return np.where(~clause_sat)[0]

    def walk_sat_step(assign, unsat_indices, noise=0.2):
        """
        Perform one WalkSAT step.
        Returns new assign and new unsat_indices
        """
        if len(unsat_indices) == 0:
            return assign, unsat_indices
            
        # Pick a random unsatisfied clause
        ci = unsat_indices[random.randint(0, len(unsat_indices) - 1)]
        clause_lits = clauses_np[ci]
        
        # Get the 3 literals
        l1, l2, l3 = clause_lits
        vars_to_consider = []
        for lit in clause_lits:
            if lit != 0:
                vars_to_consider.append(abs(lit) - 1)
        
        if not vars_to_consider:
             return assign, unsat_indices

        # Calculate flip score for each variable in the clause
        # Score = (# of clauses that become satisfied) - (# of clauses that become unsatisfied)
        # We only look at clauses containing these variables.
        
        # For simplicity and speed, we just pick a random variable from the clause
        # with a probability proportional to its "potential" to fix things.
        # A simpler heuristic: Randomly flip one of the variables in the clause.
        # WalkSAT usually uses a noise parameter to sometimes pick the variable that minimizes damage.
        
        # To keep it fast, we'll just pick a random variable from the clause.
        # This is "Random Walk" which is a variant of WalkSAT.
        var_idx = vars_to_consider[random.randint(0, len(vars_to_considers) - 1)]
        
        # Flip the variable
        assign[var_idx] = not assign[var_idx]
        
        # Re-evaluate unsat clauses
        # This is expensive if we do it fully. 
        # Optimization: Only update clauses affected by var_idx.
        # But finding them requires a map.
        
        # For 150 vars and ~500-1000 clauses, full re-evaluation is ~150*1000 = 150k ops.
        # In Python/NumPy, this is fast enough for many iterations.
        
        new_unsat = get_unsat_clauses(assign)
        return assign, new_unsat

    # Main Loop
    best_assign = None
    best_unsat_count = m + 1
    
    rng = np.random.default_rng(42)
    
    # Pre-compute variable to clauses mapping for potential optimization?
    # Let's stick to full vectorized evaluation first.
    
    while time.time() < deadline - 5.0:
        # Restart with random assignment
        assign = rng.random(n) < 0.5
        unsat_indices = get_unsat_clauses(assign)
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assign.tolist()
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assign = assign.copy()
            
        # Local search phase
        steps = 0
        max_steps = 50000 # Limit steps per restart to avoid getting stuck too long
        noise = 0.5
        
        while unsat_count > 0 and steps < max_steps and time.time() < deadline - 5.0:
            # Pick random unsat clause
            ci = unsat_indices[rng.integers(0, len(unsat_indices))]
            clause_lits = clauses_np[ci]
            
            # Get valid variables in this clause
            valid_vars = []
            for lit in clause_lits:
                if lit != 0:
                    valid_vars.append(abs(lit) - 1)
            
            if not valid_vars:
                break
                
            # Heuristic: Pick variable that minimizes new unsat clauses
            # We approximate this by checking the impact on a few random clauses or just random pick.
            # For speed, let's use a simple random pick with occasional "greedy" choice.
            
            # Simple WalkSAT: Randomly flip one variable in the clause
            var_idx = valid_vars[rng.integers(0, len(valid_vars))]
            
            # Flip
            old_val = assign[var_idx]
            assign[var_idx] = not old_val
            
            # Update unsat clauses
            # Optimization: We can update only affected clauses, but for now, full check.
            # To speed up, we can check if the flipped variable was the *only* satisfying literal
            # in any clause, or if it *became* the only satisfying literal.
            
            # Let's just re-evaluate. It's O(m*3).
            new_unsat_indices = get_unsat_clauses(assign)
            new_unsat_count = len(new_unsat_indices)
            
            if new_unsat_count < unsat_count:
                # Accept move
                unsat_indices = new_unsat_indices
                unsat_count = new_unsat_count
                # Reduce noise or keep it?
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assign = assign.copy()
            else:
                # Accept with probability noise (WalkSAT behavior)
                if rng.random() < noise:
                    unsat_indices = new_unsat_indices
                    unsat_count = new_unsat_count
                else:
                    # Revert
                    assign[var_idx] = old_val
                    # unsat_indices remains same as before the flip attempt?
                    # Actually, if we revert, the state is identical to before.
                    # So we don't need to re-evaluate if we revert.
            
            steps += 1
            # Adapt noise
            if steps % 1000 == 0:
                noise = max(0.1, noise * 0.9)
        
        # If we found a solution during local search, return it
        if unsat_count == 0:
            return assign.tolist()

    # If loop finishes, return best found
    if best_assign is not None:
        return best_assign.tolist()
    
    # Fallback: Random
    return [False] * n

# EVOLVE-BLOCK-END
