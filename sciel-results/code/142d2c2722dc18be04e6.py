"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    """
    Solve a planted 3-SAT instance using a combination of:
    1. Phase saving heuristic initialization
    2. WalkSAT with adaptive noise and clause weighting
    3. Multiple restarts with increasing effort
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Build literal occurrence counts for phase saving
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    
    # Initial assignment using phase saving heuristic
    assignment = [pos_count[i] > neg_count[i] for i in range(n)]
    
    # Pre-compute clause indices for faster access
    # clause_vars[i] = list of (var_index, sign) for clause i
    # where sign is True if positive literal, False if negative
    clause_data = []
    for clause in clauses:
        cdata = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = lit > 0
            cdata.append((var_idx, sign))
        clause_data.append(cdata)
    
    m = len(clauses)
    
    def is_satisfied(ci):
        cdata = clause_data[ci]
        for var_idx, sign in cdata:
            if assignment[var_idx] == sign:
                return True
        return False
    
    def count_unsat():
        return sum(1 for i in range(m) if not is_satisfied(i))
    
    # Check if already satisfied
    if count_unsat() == 0:
        return assignment
    
    # WalkSAT with adaptive parameters
    best_score = count_unsat()
    
    # Try multiple restarts with different strategies
    total_budget = 500000  # Total flip operations across all restarts
    num_restarts = 10
    flips_per_restart = total_budget // num_restarts
    
    for restart in range(num_restarts):
        # For first restart, use phase-saving init; for others, random init
        if restart == 0:
            # Keep phase-saving assignment
            pass
        else:
            rng = random.Random(42 + restart)
            assignment = [rng.random() < 0.5 for i in range(n)]
            # Blend with phase saving: 70% random, 30% phase saving
            ps = [pos_count[i] > neg_count[i] for i in range(n)]
            for i in range(n):
                if rng.random() < 0.3:
                    assignment[i] = ps[i]
        
        current_unsat = count_unsat()
        if current_unsat == 0:
            return assignment
        
        # Adaptive WalkSAT
        rng = random.Random(100 + restart)
        
        # Weighted clause selection: weight unsatisfied clauses by how many literals are false
        # (i.e., prefer clauses that are "close" to being satisfied)
        
        for step in range(flips_per_restart):
            # Find unsatisfied clauses
            unsat_indices = [i for i in range(m) if not is_satisfied(i)]
            if not unsat_indices:
                return assignment
            
            # Weighted selection: weight = number of false literals in the clause
            # For 3-CNF, this is 1, 2, or 3
            weights = []
            for ci in unsat_indices:
                cdata = clause_data[ci]
                false_count = 0
                for var_idx, sign in cdata:
                    if assignment[var_idx] != sign:
                        false_count += 1
                weights.append(false_count)
            
            # Select clause with probability proportional to weight
            total_weight = sum(weights)
            r = rng.random() * total_weight
            cumulative = 0
            selected_ci = unsat_indices[0]
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    selected_ci = unsat_indices[idx]
                    break
            
            cdata = clause_data[selected_ci]
            
            # Compute flip scores for each literal in the selected clause
            # Score = (number of satisfied clauses that would become unsatisfied) 
            #        - (number of unsatisfied clauses that would become satisfied)
            # We want to minimize this score (or maximize the negative)
            
            best_var = -1
            best_score_val = float('inf')
            
            for var_idx, sign in cdata:
                # Simulate flipping var_idx
                # Count how many currently-satisfied clauses would become unsatisfied
                # and how many currently-unsatisfied clauses would become satisfied
                
                # We only need to check clauses containing var_idx
                # But for efficiency with small n, let's just compute directly
                # Actually, let's precompute which clauses contain each variable
                
                pass
            
            # For efficiency, let's use a simpler heuristic:
            # Flip the literal that is false in the most other unsatisfied clauses
            # (i.e., the variable that appears in the most other unsatisfied clauses as the same polarity)
            
            # Count for each variable in the clause: how many other unsatisfied clauses 
            # would be satisfied if we flip this variable to the correct value
            # And how many satisfied clauses would be broken
            
            # Let's do a proper score computation
            # For each literal in the selected clause:
            #   score = - (number of unsat clauses that contain this literal with same sign)
            #            + (number of sat clauses that contain this literal with opposite sign)
            # We want to minimize score (most negative is best)
            
            # Precompute: for each variable, list of (clause_idx, sign) 
            var_clauses = [[] for _ in range(n)]
            for ci in range(m):
                for var_idx, sign in clause_data[ci]:
                    var_clauses[var_idx].append((ci, sign))
            
            best_var_idx = cdata[0][0]
            best_flip_score = float('inf')
            
            for var_idx, sign in cdata:
                # If we flip var_idx, it becomes (not assignment[var_idx])
                # Current value: assignment[var_idx]
                # New value: not assignment[var_idx]
                
                # Clauses containing var_idx:
                # - If clause currently satisfied only by var_idx (with current value), flipping breaks it
                # - If clause currently unsatisfied and contains var_idx with sign matching new value, flipping satisfies it
                
                # Simpler: compute delta
                # For each clause containing var_idx:
                #   old_contrib = 1 if clause is satisfied by var_idx's current value, else 0
                #   new_contrib = 1 if clause is satisfied by var_idx's new value, else 0
                #   But we need to check if the clause is satisfied by OTHER literals too
                
                # Let's compute:
                #   num_broken = number of currently-satisfied clauses that would become unsatisfied
                #   num_fixed = number of currently-unsatisfied clauses that would become satisfied
                
                num_broken = 0
                num_fixed = 0
                new_val = not assignment[var_idx]
                
                for ci, csign in var_clauses[var_idx]:
                    cdata_i = clause_data[ci]
                    # Check if clause is currently satisfied
                    currently_sat = False
                    for v, s in cdata_i:
                        if assignment[v] == s:
                            currently_sat = True
                            break
                    
                    # Check if clause would be satisfied after flipping var_idx
                    would_be_sat = False
                    for v, s in cdata_i:
                        if v == var_idx:
                            if new_val == s:
                                would_be_sat = True
                                break
                        else:
                            if assignment[v] == s:
                                would_be_sat = True
                                break
                    
                    if currently_sat and not would_be_sat:
                        num_broken += 1
                    elif not currently_sat and would_be_sat:
                        num_fixed += 1
                
                flip_score = num_broken - num_fixed
                if flip_score < best_flip_score:
                    best_flip_score = flip_score
                    best_var_idx = var_idx
            
            # Adaptive noise: with some probability, flip a random literal from the clause
            # instead of the best one
            noise_prob = 0.25  # Could be adaptive
            if rng.random() < noise_prob:
                var_idx, sign = rng.choice(cdata)
                assignment[var_idx] = not assignment[var_idx]
            else:
                assignment[best_var_idx] = not assignment[best_var_idx]
        
        # After this restart, check if solved
        if count_unsat() == 0:
            return assignment
    
    # Final check
    if count_unsat() == 0:
        return assignment
    
    return assignment
# EVOLVE-BLOCK-END
