"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute literal signs and variable indices for speed
    # clauses is a list of lists of signed literals (1-based)
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n

    # Build data structures for faster clause evaluation
    # For each clause, store (var_index, sign) where sign is +1 or -1
    clause_data = []
    for c in clauses:
        cd = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cd.append((var_idx, sign))
        clause_data.append(cd)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit

    def is_satisfied(clause_idx, assignment):
        cd = clause_data[clause_idx]
        for var_idx, sign in cd:
            if assignment[var_idx] == (sign == 1):
                return True
        return False

    def count_unsat_clauses(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    # WalkSAT with adaptive parameters and restarts
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(42)
    
    # Multiple restarts with increasing budget
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        # For planted SAT, random init is often good
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unsat count
        unsat_count = count_unsat_clauses(assignment)
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive WalkSAT
        # Parameters:
        # - flip_budget: number of flips per restart
        # - noise: probability of random flip vs greedy
        # - clause_weight: weight for choosing unsat clause with fewest satisfied literals
        
        # Start with moderate parameters and adapt
        base_flip_budget = 5000 + restart * 1000
        flip_budget = min(base_flip_budget, 50000)
        
        # Noise schedule: start high, decrease
        noise = max(0.1, 0.5 - restart * 0.02)
        
        # Track current unsat clauses for efficiency
        # For simplicity, we'll recompute unsat clauses when needed
        # But we can optimize by tracking which clauses are unsat
        
        for step in range(flip_budget):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for i in range(num_clauses):
                if not is_satisfied(i, assignment):
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment
            
            # Track best
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
            
            # Choose an unsatisfied clause
            # Use weighted selection: prefer clauses with fewer satisfied literals (closer to being satisfied)
            if len(unsat_clauses) > 1 and rng.random() < 0.7:
                # Weight by inverse of (num_satisfied_literals + 1)
                weights = []
                for ci in unsat_clauses:
                    cd = clause_data[ci]
                    num_sat = 0
                    for var_idx, sign in cd:
                        if assignment[var_idx] == (sign == 1):
                            num_sat += 1
                    weights.append(1.0 / (num_sat + 1))
                # Weighted choice
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen_idx = unsat_clauses[0]
                for j, ci in enumerate(unsat_clauses):
                    cumulative += weights[j]
                    if r <= cumulative:
                        chosen_idx = ci
                        break
            else:
                chosen_idx = rng.choice(unsat_clauses)
            
            # Choose a literal to flip
            cd = clause_data[chosen_idx]
            
            # Decide between greedy (flip a literal in this clause that makes it satisfied) 
            # and random (flip any literal in this clause)
            if rng.random() < noise:
                # Random flip among literals in the clause
                var_idx, sign = rng.choice(cd)
                assignment[var_idx] = not assignment[var_idx]
            else:
                # Greedy: count how many other clauses would become unsatisfied
                # For each literal in the chosen clause, compute the delta
                best_var = None
                best_delta = float('inf')
                
                for var_idx, sign in cd:
                    # If we flip this variable, how many clauses become unsatisfied/satisfied?
                    # A clause is affected if it contains this variable
                    # We need to check all clauses containing var_idx
                    # For efficiency, we could pre-build a variable-to-clauses map
                    # But for now, we do a simple approach: count changes
                    
                    # Count how many currently satisfied clauses would become unsatisfied
                    # and how many currently unsatisfied would become satisfied
                    delta = 0
                    for i in range(num_clauses):
                        cd_i = clause_data[i]
                        contains_var = False
                        var_sign_in_clause = None
                        for vi, si in cd_i:
                            if vi == var_idx:
                                contains_var = True
                                var_sign_in_clause = si
                                break
                        if not contains_var:
                            continue
                        
                        # Current state of this clause
                        currently_satisfied = is_satisfied(i, assignment)
                        
                        # After flipping var_idx
                        # The variable's value changes
                        new_val = not assignment[var_idx]
                        
                        # Check if this clause would be satisfied after flip
                        # A clause is satisfied if any literal is true
                        # After flip, the literal with var_idx has value: new_val == (var_sign_in_clause == 1)
                        # Other literals remain the same
                        would_be_satisfied = False
                        for vi, si in cd_i:
                            if vi == var_idx:
                                if new_val == (si == 1):
                                    would_be_satisfied = True
                                    break
                            else:
                                if assignment[vi] == (si == 1):
                                    would_be_satisfied = True
                                    break
                        
                        if currently_satisfied and not would_be_satisfied:
                            delta += 1
                        elif not currently_satisfied and would_be_satisfied:
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
        
        # After flip budget exhausted, check final state
        final_unsat = count_unsat_clauses(assignment)
        if final_unsat == 0:
            return assignment
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [True] * n
# EVOLVE-BLOCK-END
