import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 290 seconds to be safe
    deadline = time.time() + 290
    
    # Pre-process: map literal to variable index (0-based)
    # clauses are list of lists of signed literals (1-based)
    
    # Build incidence lists:
    # var_clauses[var] = list of clause indices where var appears
    var_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            v_idx = abs(lit) - 1
            var_clauses[v_idx].append(c_idx)

    m = len(clauses)
    
    rng = random.Random()
    
    # WalkSAT parameters
    max_steps_per_restart = 5000
    noise_prob = 0.05
    
    # Helper to check if a clause is satisfied given an assignment
    def is_clause_satisfied(clause, assignment):
        return any(assignment[abs(l)-1] == (l > 0) for l in clause)

    while time.time() < deadline:
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause satisfaction status
        clause_sat = [False] * m
        unsat_indices = []
        
        for c_idx, clause in enumerate(clauses):
            if is_clause_satisfied(clause, assignment):
                clause_sat[c_idx] = True
            else:
                unsat_indices.append(c_idx)
                
        if not unsat_indices:
            return assignment
            
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            if not unsat_indices:
                return assignment
                
            # Pick a random unsatisfied clause
            # We use a list of indices. Some might be stale (satisfied now).
            # We pick a random index, and if it's stale, we discard it.
            # To prevent infinite loops if all are stale, we clean up if we fail a few times.
            
            c_idx = None
            attempts = 0
            while attempts < 10:
                idx = rng.choice(unsat_indices)
                if not clause_sat[idx]:
                    c_idx = idx
                    break
                attempts += 1
                
            if c_idx is None:
                # Clean up the list
                unsat_indices = [i for i in unsat_indices if not clause_sat[i]]
                if not unsat_indices:
                    return assignment
                c_idx = rng.choice(unsat_indices)
            
            clause = clauses[c_idx]
            
            # Heuristic: Pick variable in clause to flip
            if rng.random() < noise_prob:
                chosen_lit = rng.choice(clause)
                v_idx = abs(chosen_lit) - 1
            else:
                best_var = None
                best_score = -1000
                
                for lit in clause:
                    v_idx = abs(lit) - 1
                    
                    # Calculate score: net change in number of satisfied clauses
                    # old_sat: number of clauses containing v_idx that are currently satisfied
                    # new_sat: number of clauses containing v_idx that would be satisfied if v_idx is flipped
                    
                    old_sat_count = 0
                    new_sat_count = 0
                    
                    for c in var_clauses[v_idx]:
                        if clause_sat[c]:
                            old_sat_count += 1
                            
                        # Check if flipping v_idx satisfies clause c
                        # We need to check the literal for v_idx in clause c
                        # Iterate through clause c to find the literal for v_idx
                        for l in clauses[c]:
                            if abs(l) - 1 == v_idx:
                                # If l is +v_idx, it is satisfied if v_idx is True.
                                # If l is -v_idx, it is satisfied if v_idx is False.
                                # We are flipping v_idx.
                                # Current value: assignment[v_idx]
                                # New value: not assignment[v_idx]
                                
                                # If l > 0 (positive literal):
                                #   Satisfied if new value is True -> not assignment[v_idx] is True -> assignment[v_idx] is False
                                # If l < 0 (negative literal):
                                #   Satisfied if new value is False -> not assignment[v_idx] is False -> assignment[v_idx] is True
                                
                                if (l > 0 and not assignment[v_idx]) or (l < 0 and assignment[v_idx]):
                                    new_sat_count += 1
                                break
                                
                    score = new_sat_count - old_sat_count
                    if score > best_score:
                        best_score = score
                        best_var = v_idx
                        
                if best_var is not None:
                    v_idx = best_var
                else:
                    v_idx = abs(rng.choice(clause)) - 1

            # Flip the variable
            assignment[v_idx] = not assignment[v_idx]
            
            # Update clause satisfaction status for all clauses containing v_idx
            for c in var_clauses[v_idx]:
                was_sat = clause_sat[c]
                is_now_sat = is_clause_satisfied(clauses[c], assignment)
                
                if is_now_sat and not was_sat:
                    clause_sat[c] = True
                    # It becomes satisfied, so it should be removed from unsat_indices.
                    # We don't remove it immediately, we just mark it.
                elif not is_now_sat and was_sat:
                    clause_sat[c] = False
                    # It becomes unsatisfied, so it should be added to unsat_indices.
                    unsat_indices.append(c)
                else:
                    # State didn't change (sat->sat or unsat->unsat)
                    # But we must ensure clause_sat is correct?
                    # It was correct before, and state didn't change, so it's still correct.
                    pass

            steps += 1

    # If deadline reached, return last assignment
    return assignment
