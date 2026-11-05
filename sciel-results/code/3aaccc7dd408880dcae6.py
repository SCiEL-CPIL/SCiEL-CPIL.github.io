"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure: for each clause, list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, list of (clause_idx, is_positive) it appears in
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    def solve_one_restart(rng):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # clause_sat_count[ci] = number of literals in clause ci that are true
        # clause_satisfied[ci] = whether clause ci is satisfied
        clause_sat_count = [0] * m
        clause_satisfied = [False] * m
        
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if assignment[vi] == pos:
                    count += 1
            clause_sat_count[ci] = count
            clause_satisfied[ci] = (count > 0)
        
        unsat_count = m - sum(clause_satisfied)
        
        if unsat_count == 0:
            return assignment
        
        # Parameters for weighted WalkSAT
        # We use a dynamic approach: bias toward flipping variables that
        # satisfy more unsatisfied clauses without breaking satisfied ones.
        
        max_steps = 50000
        # Decay factor for the random component
        p = 0.5  # probability of picking a random literal from the clause
        
        for step in range(max_steps):
            # Check deadline
            if time.time() > deadline - 5.0:
                return None
            
            # Pick a random unsatisfied clause
            # We maintain a list of unsatisfied clause indices for efficiency
            # But rebuilding it every step is O(m), so let's just pick randomly and check
            # For 150 vars and reasonable m, this should be fine
            # Actually, let's maintain a proper unsat list
            
            # Find an unsatisfied clause
            # To be efficient, we'll pick a random clause and check
            # If satisfied, try again (with a limit)
            found_unsat = False
            for _ in range(10):
                ci = rng.randint(0, m - 1)
                if not clause_satisfied[ci]:
                    found_unsat = True
                    break
            
            if not found_unsat:
                # All clauses satisfied (shouldn't happen if unsat_count > 0, but safety)
                return assignment
            
            cv = clause_vars[ci]
            
            # Score each literal in the clause
            # Score = (number of unsatisfied clauses that would be satisfied by flipping)
            #        - (number of satisfied clauses that would be broken by flipping)
            best_var = -1
            best_score = -10**9
            scores = []
            
            for vi, pos in cv:
                # If we flip variable vi:
                # - In the current unsatisfied clause ci, the literal (vi, pos) becomes true,
                #   so clause ci becomes satisfied (since it was unsat, all its literals were false,
                #   so flipping any one makes it satisfied)
                # - For other clauses containing vi:
                #   If the clause was satisfied, flipping vi might break it (if the literal in that
                #   clause was the only true one)
                #   If the clause was unsatisfied, flipping vi might satisfy it (if the literal
                #   in that clause becomes true)
                
                gain = 0  # clauses that go from unsat to sat
                loss = 0  # clauses that go from sat to unsat
                
                for cj, pos_j in var_clauses[vi]:
                    if cj == ci:
                        continue
                    # Determine if the literal (vi, pos_j) is currently true
                    literal_true = (assignment[vi] == pos_j)
                    if literal_true:
                        # The literal is true, so clause cj might be satisfied
                        if clause_satisfied[cj]:
                            # If this is the only true literal in clause cj, flipping breaks it
                            if clause_sat_count[cj] == 1:
                                loss += 1
                    else:
                        # The literal is false
                        if not clause_satisfied[cj]:
                            # Flipping will make this literal true, satisfying clause cj
                            gain += 1
                
                # The flip also affects clause ci itself:
                # Since ci is unsatisfied, all its literals are false.
                # Flipping vi makes (vi, pos) true in clause ci, so ci becomes satisfied.
                # This is a gain of 1.
                gain += 1
                
                score = gain - loss
                scores.append((score, vi))
            
            # Choose a literal to flip
            if rng.random() < p:
                # Random choice from the clause
                vi = rng.choice(cv)[0]
            else:
                # Choose the variable with the best score
                # Break ties randomly
                best_score = max(s for s, _ in scores)
                candidates = [vi for s, vi in scores if s == best_score]
                vi = rng.choice(candidates)
            
            # Flip variable vi
            assignment[vi] = not assignment[vi]
            
            # Update clause statuses for all clauses containing vi
            for cj, pos_j in var_clauses[vi]:
                # Determine if the literal (vi, pos_j) is now true
                literal_now_true = (assignment[vi] == pos_j)
                # Determine if it was true before
                # Since we just flipped, it was the opposite
                literal_was_true = not literal_now_true
                
                if literal_now_true and not literal_was_true:
                    # Literal went from false to true
                    clause_sat_count[cj] += 1
                    if clause_satisfied[cj]:
                        pass  # was already satisfied, still is
                    else:
                        # Now satisfied
                        clause_satisfied[cj] = True
                        unsat_count -= 1
                elif not literal_now_true and literal_was_true:
                    # Literal went from true to false
                    clause_sat_count[cj] -= 1
                    if clause_satisfied[cj]:
                        if clause_sat_count[cj] == 0:
                            # Now unsatisfied
                            clause_satisfied[cj] = False
                            unsat_count += 1
                    else:
                        pass  # was already unsatisfied, still is
            
            if unsat_count == 0:
                return assignment
        
        return None
    
    # Multiple restarts
    rng = random.Random(0)
    best_result = None
    
    while True:
        if time.time() > deadline - 5.0:
            break
        
        result = solve_one_restart(rng)
        if result is not None:
            # Verify the solution
            for c in clauses:
                if not any(result[abs(l) - 1] == (l > 0) for l in c):
                    break
            else:
                return result
        
        # If we haven't found a solution, continue with more restarts
    
    # Return the best we can do (shouldn't reach here in normal operation)
    # Return a random assignment as fallback
    return [False] * n
# EVOLVE-BLOCK-END
