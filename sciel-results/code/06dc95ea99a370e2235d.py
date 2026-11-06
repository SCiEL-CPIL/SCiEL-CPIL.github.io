import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # If n is small, brute force or simple DPLL might be faster, but let's stick to local search for consistency.
    # For 150 variables, local search is appropriate.
    
    rng = random.Random(42) # Fixed seed for reproducibility in evaluation if needed, but task says "locally generated", usually implies we can use any seed. Using 0 or 42 is fine.
    
    max_restarts = 50
    max_steps_per_restart = 50000
    
    # Weights for clauses
    clause_weights = [1.0] * len(clauses)
    # Activity for variables
    var_activity = [1.0] * n
    
    def get_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False
        
    def count_unsat(clauses, assignment):
        return sum(0 if is_satisfied(c, assignment) else 1 for c in clauses)

    for restart in range(max_restarts):
        # Initialize assignment
        # Bias towards random, or maybe all False? Random is usually best for planted.
        assignment = get_assignment()
        
        # Reset some weights? Or keep them to guide future restarts? 
        # Keeping clause weights can help if the same hard clauses appear.
        # Let's decay clause weights slightly on restart to avoid bias accumulation.
        for i in range(len(clauses)):
            clause_weights[i] *= 0.8
        for i in range(n):
            var_activity[i] *= 0.8
            
        unsat_count = count_unsat(clauses, assignment)
        
        if unsat_count == 0:
            return assignment
            
        # Local search loop
        for step in range(max_steps_per_restart):
            if unsat_count == 0:
                return assignment
                
            # Find an unsatisfied clause
            # To speed up, we could maintain a list of unsat clauses, but for 150 vars and ~300-400 clauses, 
            # scanning is fast enough. 
            unsat_clauses = []
            for i, c in enumerate(clauses):
                if not is_satisfied(c, assignment):
                    unsat_clauses.append(i)
                    
            if not unsat_clauses:
                return assignment
                
            # Pick a clause to focus on
            # Weighted random selection based on clause_weights
            # Sum of weights
            total_weight = sum(clause_weights[i] for i in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_clause_idx = unsat_clauses[0]
            for idx in unsat_clauses:
                cumulative += clause_weights[idx]
                if r <= cumulative:
                    chosen_clause_idx = idx
                    break
            
            clause = clauses[chosen_clause_idx]
            
            # Decide which literal to flip
            # Calculate greedy gain for each literal in the clause
            # Gain = number of other unsatisfied clauses that would become satisfied
            # Note: Flipping a variable might also unsatisfy other clauses. 
            # A simpler score: number of satisfied clauses that would be broken? 
            # Standard WalkSAT: 
            # 1. Pick random unsat clause.
            # 2. Pick random literal in it.
            # 3. Flip.
            # 
            # Improved: 
            # 1. Pick random unsat clause (weighted).
            # 2. For each literal in clause, calculate "score".
            #    Score can be: (number of other unsat clauses satisfied by this flip) - (number of sat clauses broken by this flip)
            #    This is expensive to calculate every step for 150 vars.
            #    
            # Let's use a simpler heuristic: 
            # With probability p (0.2), pick the literal that appears in the most unsatisfied clauses (activity-based).
            # With probability 1-p, pick a random literal.
            
            p_greedy = 0.2
            if rng.random() < p_greedy:
                # Pick literal with highest variable activity among those in the clause
                best_lit = clause[0]
                best_act = var_activity[abs(best_lit) - 1]
                for l in clause[1:]:
                    act = var_activity[abs(l) - 1]
                    if act > best_act:
                        best_act = act
                        best_lit = l
                lit_to_flip = best_lit
            else:
                lit_to_flip = rng.choice(clause)
                
            var_idx = abs(lit_to_flip) - 1
            assignment[var_idx] = not assignment[var_idx]
            
            # Update weights
            # Increase weight of the clause we just tried to satisfy
            clause_weights[chosen_clause_idx] *= 1.1
            # Increase activity of the variable we flipped
            var_activity[var_idx] *= 1.1
            
            # Recalculate unsat count
            # Optimization: Instead of full scan, we can update incrementally, but for simplicity and robustness:
            unsat_count = count_unsat(clauses, assignment)
            
            # Decay activities and weights slightly to prevent explosion
            if step % 100 == 0:
                for i in range(n):
                    var_activity[i] *= 0.95
                for i in range(len(clauses)):
                    clause_weights[i] *= 0.95

    # If we fail all restarts, return the last assignment (which is unsat)
    # But we should try one more time with a different strategy?
    # Or just return the best effort.
    return assignment
