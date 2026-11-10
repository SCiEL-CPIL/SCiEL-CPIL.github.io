import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocess: remove duplicates and empty clauses (if any, though task says satisfiable)
    # Convert to 0-based indices for faster access
    # clause: list of (var, is_pos) or just signed ints
    # We will keep signed ints but use 0-based var index internally
    
    # Optimized WalkSAT
    # 1. Map variables to their occurrences in clauses for fast flip evaluation
    # var_clauses[var] = list of (clause_index, literal_sign)
    
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            is_pos = lit > 0
            var_clauses[var].append((ci, is_pos))
            
    # Precompute clause lengths? Not strictly needed for WalkSAT, but useful.
    
    # State:
    # assignment: list of bools (True/False)
    # sat_count: list of int, number of satisfied literals in each clause
    # is_satisfied: list of bools, whether clause is satisfied
    
    def init_state(assignment):
        sat_count = [0] * len(clauses)
        is_satisfied = [False] * len(clauses)
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    count += 1
            sat_count[ci] = count
            if count > 0:
                is_satisfied[ci] = True
        return sat_count, is_satisfied

    def flip_var(var, assignment, sat_count, is_satisfied):
        """
        Flips variable 'var' and updates sat_count and is_satisfied.
        Returns the number of newly satisfied clauses (gain) and newly unsatisfied clauses (loss).
        """
        # Determine current value
        current_val = assignment[var]
        
        # Update all clauses containing this variable
        # We need to know if the literal in the clause was satisfied before flip.
        # If assignment[var] == (lit > 0), then it was satisfied.
        
        # For each clause containing var:
        for ci, is_pos in var_clauses[var]:
            lit = var + 1 if is_pos else -(var + 1)
            # Check if this literal was satisfied before flip
            was_sat = (assignment[var] == (lit > 0))
            
            if was_sat:
                # It was satisfied, now it will be unsatisfied
                sat_count[ci] -= 1
                if sat_count[ci] == 0:
                    is_satisfied[ci] = False
            else:
                # It was unsatisfied, now it will be satisfied
                sat_count[ci] += 1
                if sat_count[ci] == 1:
                    is_satisfied[ci] = True
            
        assignment[var] = not current_val

    # WalkSAT parameters
    # p: probability of choosing the best literal vs random
    # steps: max steps per restart
    # restarts: max restarts
    
    rng = random.Random(42)
    
    # Time limit check
    start_time = time.time()
    time_limit = 250.0  # Leave some buffer for Python overhead
    
    # Try multiple restarts
    for restart in range(100):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count, is_satisfied = init_state(assignment)
        
        # Check if already satisfied
        if all(is_satisfied):
            return assignment
        
        unsat_clauses = [ci for ci in range(len(clauses)) if not is_satisfied[ci]]
        
        if not unsat_clauses:
            return assignment
            
        # WalkSAT loop
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
                
            if not unsat_clauses:
                return assignment
                
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clauses[ci]
            
            # Calculate scores for each literal in the clause
            # Score = (number of unsat clauses it satisfies) - (number of sat clauses it breaks)
            best_var = -1
            best_score = -1
            scores = []
            
            for lit in clause:
                var = abs(lit) - 1
                is_pos = lit > 0
                
                # Calculate gain (unsat clauses satisfied) and loss (sat clauses broken)
                gain = 0
                loss = 0
                
                for cj, pos in var_clauses[var]:
                    # Check if this variable's literal in clause cj is satisfied
                    # lit_j = var+1 if pos else -(var+1)
                    # is_sat_j = assignment[var] == (lit_j > 0)
                    
                    # Actually, we can just look at sat_count and is_satisfied
                    # If the literal in clause cj is currently satisfied:
                    #   It will become unsatisfied after flip -> Loss
                    # Else:
                    #   It will become satisfied after flip -> Gain
                    
                    # How to know if it's satisfied?
                    # We have sat_count[cj]. But we need to know if THIS literal contributes.
                    # If assignment[var] == (lit_j > 0), then it contributes to sat_count[cj].
                    
                    lit_j = var + 1 if pos else -(var + 1)
                    is_sat_j = (assignment[var] == (lit_j > 0))
                    
                    if is_sat_j:
                        loss += 1
                    else:
                        gain += 1
                        
                score = gain - loss
                scores.append((score, var))
                
            # Sort by score descending
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Choose variable to flip
            # With probability p, pick the best one. Otherwise, pick randomly.
            # Or pick from top 3.
            # Standard WalkSAT: pick best with prob p, else random.
            p = 0.5
            if rng.random() < p:
                var_to_flip = scores[0][1]
            else:
                var_to_flip = scores[rng.randint(0, len(scores)-1)][1]
                
            # Flip the variable
            flip_var(var_to_flip, assignment, sat_count, is_satisfied)
            
            # Update unsat_clauses
            # This is expensive if done naively. 
            # Optimization: Keep a set of unsat clause indices.
            # When we flip a var, only clauses containing that var change status.
            # We can update the list by removing those that became sat and adding those that became unsat.
            
            # For simplicity and since n=50, m is likely small, we can just rebuild or update carefully.
            # Let's maintain a set of unsat clause indices for O(1) membership and easy removal/addition.
            
            # Actually, the current code uses a list `unsat_clauses`. 
            # Let's switch to a set for efficiency in the inner loop, or just rebuild if it's not too slow.
            # Given n=50, even O(m) rebuild per step is fine if m is small.
            # But let's try to be smart.
            
            # We don't have a set in the current scope. Let's just rebuild the list of unsat clauses.
            # It's O(m). m is usually < 1000 for n=50 3-SAT.
            unsat_clauses = [cj for cj in range(len(clauses)) if not is_satisfied[cj]]

    # If we reach here, we didn't find a solution in time/restarts.
    # Return the last assignment as a fallback (should be rare for planted instances)
    return assignment
