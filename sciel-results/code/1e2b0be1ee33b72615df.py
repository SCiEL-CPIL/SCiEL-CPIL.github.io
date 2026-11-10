"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
    
    def is_satisfied(clause_idx, assignment):
        fc = fast_clauses[clause_idx]
        for var_idx, required in fc:
            if assignment[var_idx] == required:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                count += 1
        return count
    
    # WalkSAT with restarts and improved heuristics
    best_assignment = None
    best_unsat_count = float('inf')
    
    num_restarts = 200  # More restarts with shorter walks
    max_steps = 10000   # Steps per restart
    
    # Walk probability: probability of picking a random literal instead of heuristic
    walk_prob = 0.25
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment randomly
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Track current unsat count and unsat clause set
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        # Pre-compute per-variable unsat clause count for fast heuristic
        var_unsat_count = [0] * n
        for c_idx in unsat_set:
            fc = fast_clauses[c_idx]
            for var_idx, _ in fc:
                var_unsat_count[var_idx] += 1
        
        current_unsat = len(unsat_set)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # WalkSAT loop with clause status tracking
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
            
            # Choose a clause from unsat_set
            clause_idx = random.choice(list(unsat_set))
            clause = fast_clauses[clause_idx]
            
            # With probability walk_prob, pick a random literal from the clause
            # Otherwise, use frequency-based heuristic
            if random.random() < walk_prob:
                var_idx, required = random.choice(clause)
            else:
                # Heuristic: prefer literals that appear in more unsat clauses
                # Use pre-computed var_unsat_count for O(1) per literal
                scores = []
                for var_idx, required in clause:
                    score = var_unsat_count[var_idx]
                    scores.append(score)
                
                total_score = sum(scores)
                if total_score == 0:
                    var_idx, required = random.choice(clause)
                else:
                    # Weighted selection
                    r = random.random() * total_score
                    cumulative = 0
                    var_idx, required = clause[0]
                    for i, score in enumerate(scores):
                        cumulative += score
                        if r <= cumulative:
                            var_idx, required = clause[i]
                            break
            
            # Flip the variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_set and var_unsat_count incrementally
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_sat = is_satisfied(c_idx, assignment)
                
                if was_unsat and is_sat:
                    # Clause became satisfied: remove from unsat_set, decrement counts
                    unsat_set.discard(c_idx)
                    fc = fast_clauses[c_idx]
                    for v_idx, _ in fc:
                        var_unsat_count[v_idx] -= 1
                elif not was_unsat and not is_sat:
                    # Clause became unsatisfied: add to unsat_set, increment counts
                    unsat_set.add(c_idx)
                    fc = fast_clauses[c_idx]
                    for v_idx, _ in fc:
                        var_unsat_count[v_idx] += 1
            
            # Early exit if solved
            if not unsat_set:
                return assignment
        
        # End of restart
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
