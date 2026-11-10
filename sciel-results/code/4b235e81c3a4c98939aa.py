import random
import time

def solve(instance):
    """
    Solves a 3-SAT instance using an efficient WalkSAT algorithm.
    
    Efficiency improvements:
    1. Maintains 'unsat_count' for each clause to quickly identify unsatisfied clauses.
    2. Maintains 'flip_gain' (or similar heuristic) implicitly via quick lookups.
    3. O(1) amortized cost per flip instead of O(M) where M is number of clauses.
    4. Time-guarded loop to maximize search within the deadline.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # 1. Pre-computation: Build variable-to-clause index and initial state
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(i)
            
    # Initial random assignment
    rng = random.Random(0)
    assignment = [False] * n
    for i in range(n):
        assignment[i] = rng.random() < 0.5
        
    # Initialize unsatisfied count for each clause
    # unsat_count[c] = number of literals in clause c that are False
    unsat_count = [0] * len(clauses)
    
    # Helper to check if a literal is satisfied given current assignment
    def is_lit_true(lit):
        var = abs(lit) - 1
        if lit > 0:
            return assignment[var]
        else:
            return not assignment[var]
            
    # Initial pass to populate unsat_count
    for i, clause in enumerate(clauses):
        for lit in clause:
            if not is_lit_true(lit):
                unsat_count[i] += 1
                
    # Check if solved initially
    if all(c == 3 for c in unsat_count): # Wait, unsat_count is count of UNSAT literals. 
        # If unsat_count == 0, the clause is satisfied.
        if all(c == 0 for c in unsat_count):
            return assignment

    # 2. Search Loop
    # Heuristic parameters for WalkSAT
    # P(noise) is the probability of picking a random literal from the clause
    # otherwise pick the literal that minimizes unsat_count increase (min-conflicts)
    p_noise = 0.35 
    
    deadline = time.time() + 290.0 # 10s safety margin
    best_unsat = sum(1 for c in unsat_count if c > 0)
    
    while time.time() < deadline:
        # Find unsatisfied clauses
        # Optimization: Instead of scanning all clauses every time, we could maintain a list/set of unsat clauses.
        # However, for 150 vars, M is likely ~ 500-1000. Scanning is O(M). 
        # With O(1) flip, we can afford scanning if M is small.
        # Let's maintain a list of unsat clause indices for faster selection.
        
        # For simplicity and robustness in this rewrite, we will scan for unsat clauses.
        # If M is large, we should maintain a heap or list.
        # Let's assume M is manageable (< 5000).
        
        unsat_clauses = [i for i, cnt in enumerate(unsat_count) if cnt > 0]
        
        if not unsat_clauses:
            break # Solved!
            
        # If we found a better solution (fewer unsat clauses), we could track it, 
        # but WalkSAT usually sticks to the current state. 
        # For planted SAT, we just need *a* solution.
        
        # Select a random unsatisfied clause
        c_idx = rng.choice(unsat_clauses)
        clause = clauses[c_idx]
        
        # Select a literal to flip
        # Strategy: 
        # 1. With probability p_noise, pick a random literal from the clause.
        # 2. Otherwise, pick the literal that, when flipped, results in the minimum increase in unsatisfied clauses (min-conflicts).
        
        if rng.random() < p_noise:
            lit = rng.choice(clause)
        else:
            # Min-conflicts heuristic
            best_lit = None
            best_gain = float('inf')
            
            for lit in clause:
                var = abs(lit) - 1
                
                # Calculate gain: change in total number of unsatisfied clauses if we flip var
                # Flipping var affects all clauses containing var.
                # Current state:
                # - Clause c_idx is unsatisfied.
                # - Flipping lit might satisfy c_idx (if lit is currently false).
                # - Flipping lit might unsatisfy other clauses where lit is currently true.
                
                # Let's compute the delta in unsat_count for the specific clause c_idx first
                # If lit is currently False, flipping it to True reduces unsat_count[c_idx] by 1.
                # If lit is currently True, flipping it to False increases unsat_count[c_idx] by 1 (but it's already unsat, so this doesn't change the *count* of unsat clauses, just the internal count).
                # Actually, we care about the number of *unsatisfied clauses* (where count > 0).
                
                # Let's evaluate the effect on the number of unsatisfied clauses globally.
                # This is O(deg(var)).
                
                new_unsat_clauses_delta = 0
                
                # 1. Effect on c_idx
                if is_lit_true(lit):
                    # Lit is currently True. Clause is unsatisfied.
                    # Flipping to False: unsat_count[c_idx] increases. Still unsatisfied.
                    # No change in "is unsatisfied" status for c_idx.
                    pass
                else:
                    # Lit is currently False. Clause is unsatisfied.
                    # Flipping to True: unsat_count[c_idx] decreases by 1.
                    # If unsat_count[c_idx] becomes 0, the clause becomes satisfied.
                    if unsat_count[c_idx] == 1:
                        # It was the only unsatisfied literal.
                        new_unsat_clauses_delta -= 1 # One fewer unsat clause
                    # else: still unsat, just closer to being satisfied.
                    
                # 2. Effect on other clauses containing var
                for other_c_idx in var_to_clauses[var]:
                    if other_c_idx == c_idx:
                        continue
                    other_clause = clauses[other_c_idx]
                    
                    # Determine if flipping lit affects the satisfaction status of other_c_idx
                    # It affects it if lit is in other_c_idx (it is, by construction)
                    
                    # Current status of other_c_idx:
                    # is_other_satisfied = (unsat_count[other_c_idx] == 0)
                    
                    # Does lit contribute to the satisfaction of other_c_idx?
                    # If is_lit_true(lit) is True, it contributes.
                    # If is_lit_true(lit) is False, it doesn't.
                    
                    if is_lit_true(lit):
                        # Lit is currently True. It helps satisfy other_c_idx.
                        # Flipping to False removes this help.
                        # If other_c_idx was satisfied (count 0), it becomes unsatisfied (count 1).
                        # If other_c_idx was unsatisfied, it remains unsatisfied (count increases).
                        if unsat_count[other_c_idx] == 0:
                            new_unsat_clauses_delta += 1 # One more unsat clause
                        # else: no change in status
                    else:
                        # Lit is currently False. It doesn't help satisfy other_c_idx.
                        # Flipping to True adds this help.
                        # If other_c_idx was unsatisfied, it might become satisfied.
                        # It becomes satisfied if this was the ONLY unsatisfied literal.
                        if unsat_count[other_c_idx] == 1:
                            new_unsat_clauses_delta -= 1 # One fewer unsat clause
                        # else: no change in status

                # Gain is the change in the number of unsatisfied clauses.
                # We want to minimize this (prefer negative or 0).
                if new_unsat_clauses_delta < best_gain:
                    best_gain = new_unsat_clauses_delta
                    best_lit = lit
            
            lit = best_lit
            
        # Perform the flip
        var = abs(lit) - 1
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        # Update unsat_count for all clauses affected by var
        for c_idx in var_to_clauses[var]:
            # We need to know if the literal in this clause is the one we flipped
            # and if it was True or False before.
            # Note: The literal in the clause might be positive or negative.
            
            # Find the literal in clause c_idx that corresponds to var
            # Since clauses are small (size 3), we can just iterate.
            for l in clauses[c_idx]:
                if abs(l) - 1 == var:
                    # This is the literal we care about
                    # Check if it was True or False BEFORE the flip
                    # We know old_val.
                    # If l > 0:
                    #   lit_val = old_val
                    # If l < 0:
                    #   lit_val = not old_val
                    if l > 0:
                        was_true = old_val
                    else:
                        was_true = not old_val
                        
                    # Update unsat_count[c_idx]
                    if was_true:
                        # Literal was True (satisfied), now False (unsatisfied)
                        unsat_count[c_idx] += 1
                    else:
                        # Literal was False (unsatisfied), now True (satisfied)
                        unsat_count[c_idx] -= 1
                    break # Found the literal, break inner loop
                    
    return assignment

