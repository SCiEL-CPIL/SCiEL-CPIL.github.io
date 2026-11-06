"""Improved WalkSAT with bit-vector acceleration and adaptive strategies."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: keep a safety margin before 330s
    deadline = time.time() + 320.0
    
    # Precompute bitmask representations for fast clause evaluation
    # For each variable i, store two bitmasks:
    #   pos_mask[i]: bitmask of clauses where variable i appears positively
    #   neg_mask[i]: bitmask of clauses where variable i appears negatively
    # Also for each clause, we need to know which literals are in it.
    
    # For efficient satisfied-clause checking, we'll use a different approach:
    # Maintain an array of unsatisfied clause indices, and for each variable,
    # precompute which unsatisfied clauses it can help satisfy.
    
    # Actually, let's use a hybrid approach:
    # 1. Precompute for each variable the list of clauses it appears in, with sign
    # 2. Use a clause counter to track how many literals in each clause are satisfied
    # 3. When flipping a variable, update the counter for affected clauses
    
    # Precomputation
    var_clauses = [[] for _ in range(n)]  # var_clauses[i] = list of (clause_idx, sign) where sign=1 for positive, -1 for negative
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[vi].append((ci, sign))
    
    # clause_sat_count[ci] = number of satisfied literals in clause ci
    # Initialize with random assignment
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def init_sat_counts():
        counts = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                vi = abs(lit) - 1
                if assignment[vi] == (lit > 0):
                    counts[ci] += 1
        return counts
    
    clause_sat_count = init_sat_counts()
    
    def is_satisfied(ci):
        return clause_sat_count[ci] > 0
    
    def count_unsat():
        return sum(1 for c in clause_sat_count if c == 0)
    
    def flip_var(vi):
        """Flip variable vi and update clause_sat_count. Returns net change in satisfied clause count."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        
        delta_sat = 0
        for ci, sign in var_clauses[vi]:
            # If the literal (vi with sign) was satisfied before, it's now unsatisfied
            # If it was unsatisfied before, it's now satisfied
            was_satisfied = (old_val == (sign == 1))
            if was_satisfied:
                clause_sat_count[ci] -= 1
                if clause_sat_count[ci] == 0:
                    delta_sat -= 1
            else:
                clause_sat_count[ci] += 1
                if clause_sat_count[ci] == 1:
                    delta_sat += 1
        return delta_sat
    
    # WalkSAT with adaptive strategy
    # Try multiple restarts with increasing effort
    
    best_unsat = count_unsat()
    best_assignment = assignment[:]
    
    # Strategy: WalkSAT with noise, periodic restarts
    # Use a scoring function: minimize unsatisfied clauses, with random tie-breaking
    
    max_restarts = 200
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Reset to best assignment found so far, or random if first restart
        if restart > 0 and best_unsat < count_unsat():
            assignment = best_assignment[:]
            clause_sat_count = init_sat_counts()
        
        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            unsat_count = count_unsat()
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Find an unsatisfied clause
            # To avoid O(m) scan every step, maintain a list of unsatisfied clause indices
            # But for simplicity with 50 vars and moderate clause count, let's sample
            # Actually, let's be smarter: pick a random clause, check if unsat, retry a few times
            
            clause_idx = None
            for _ in range(10):
                ci = rng.randrange(m)
                if clause_sat_count[ci] == 0:
                    clause_idx = ci
                    break
            
            if clause_idx is None:
                # All clauses might be satisfied, or we missed them
                # Do a full check
                unsat_clauses = [ci for ci in range(m) if clause_sat_count[ci] == 0]
                if not unsat_clauses:
                    return assignment
                clause_idx = rng.choice(unsat_clauses)
            
            clause = clauses[clause_idx]
            
            # Choose a literal to flip: prefer those that maximize gain
            # Compute gain for each literal in the clause
            best_gain = -float('inf')
            candidates = []
            for lit in clause:
                vi = abs(lit) - 1
                # If we flip vi, what's the net change in satisfied clauses?
                # We can estimate: for each clause containing vi, check if flip helps
                gain = 0
                for ci2, sign2 in var_clauses[vi]:
                    if sign2 == (1 if lit > 0 else -1):
                        # This is the same literal, flipping makes it unsatisfied
                        if clause_sat_count[ci2] == 1:
                            gain -= 1
                        # If it was the only satisfied literal, we lose a clause
                    else:
                        # Opposite literal, flipping makes it satisfied
                        if clause_sat_count[ci2] == 0:
                            gain += 1
                        # If clause was unsatisfied, we gain a clause
                
                # Add randomness for tie-breaking
                gain += rng.random() * 0.5
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [vi]
                elif gain == best_gain:
                    candidates.append(vi)
            
            # Pick randomly from best candidates
            vi_to_flip = rng.choice(candidates)
            flip_var(vi_to_flip)
    
    # Return the best assignment found
    return best_assignment

# EVOLVE-BLOCK-END
