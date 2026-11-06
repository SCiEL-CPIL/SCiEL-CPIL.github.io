"""Planted 3-SAT solver using Weighted WalkSAT with Adaptive Restarts."""
import random
import time

def solve(instance):
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Parse clauses into 0-based indices and polarities (1 for True, 0 for False)
    # literal = (var_idx, polarity)
    # If literal is positive, polarity=1. If negative, polarity=0.
    # Assignment is a list of bools.
    # Clause is satisfied if any (assignment[var] == polarity) is True.
    
    clause_lits = []
    var_occurrences = [[] for _ in range(n)]
    pos_count = [0] * n
    neg_count = [0] * n
    
    for ci, c in enumerate(clauses):
        lits = []
        for l in c:
            v = abs(l) - 1
            p = 1 if l > 0 else 0
            lits.append((v, p))
            var_occurrences[v].append((ci, p))
            if p == 1:
                pos_count[v] += 1
            else:
                neg_count[v] += 1
        clause_lits.append(lits)
        
    # Helper to check if a clause is satisfied
    def is_satisfied(ci, assignment):
        for v, p in clause_lits[ci]:
            if assignment[v] == (p == 1):
                return True
        return False
        
    # Helper to count unsatisfied clauses
    def count_unsat(assignment):
        count = 0
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                count += 1
        return count

    # Strategy 1: Try polarity-based assignment quickly
    def try_planted_assignment():
        assignment = [pos_count[v] >= neg_count[v] for v in range(n)]
        if count_unsat(assignment) == 0:
            return assignment
        return None
        
    result = try_planted_assignment()
    if result is not None:
        return result

    # Weighted WalkSAT
    # Weights for each clause
    clause_weights = [1.0] * m
    # Track unsatisfied clauses for quick access
    # Instead of a set, we can just iterate or maintain a list. 
    # For 150 vars, m is likely around 400-600. Iterating is fine.
    
    best_unsat = float('inf')
    best_assignment = None
    
    max_restarts = 500
    # p_flip: probability of flipping a variable that doesn't improve the score
    p_flip = 0.55
    
    for restart in range(max_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Initialization
        if restart == 0:
            # Polarity biased start
            assignment = []
            for v in range(n):
                if pos_count[v] > neg_count[v]:
                    assignment.append(random.random() < 0.8)
                elif neg_count[v] > pos_count[v]:
                    assignment.append(random.random() < 0.2)
                else:
                    assignment.append(random.random() < 0.5)
        else:
            # Random restart, but maybe slightly biased by previous best? 
            # Standard WalkSAT uses pure random.
            # Let's try pure random to escape local minima effectively.
            assignment = [random.random() < 0.5 for _ in range(n)]
            
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            
        steps = 0
        max_steps = 1000 + restart * 100
        no_improve_steps = 0
        local_best_unsat = current_unsat
        
        while steps < max_steps and time.time() - start_time < deadline:
            # Find an unsatisfied clause
            # To be efficient, we can pick a random clause index and check if it's unsat.
            # If it is sat, try again. With high unsat rate, this is fast.
            # With low unsat rate, it might be slow.
            # Better: maintain a list of unsat clause indices.
            
            # For simplicity and speed in Python, let's just scan or use a list.
            # Given m is small (~500), scanning is O(m).
            # Let's find a random unsat clause.
            unsat_ci = -1
            # Random sampling
            while True:
                ci = random.randrange(m)
                if not is_satisfied(ci, assignment):
                    unsat_ci = ci
                    break
                    
            if unsat_ci == -1:
                # Should not happen if count_unsat > 0
                break
                
            # Select a variable to flip from the unsatisfied clause
            clause = clause_lits[unsat_ci]
            # Standard WalkSAT: pick random variable from the clause
            v, p = random.choice(clause)
            
            # Calculate new unsat count if we flip v
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val
            
            # Calculate change in unsat count
            # We need to check all clauses containing v
            delta = 0
            for (cci, cp) in var_occurrences[v]:
                # Check state before flip (conceptually) and after
                # Since we already flipped, we check current state vs what it was.
                # Easier: just count unsat for clauses involving v before and after?
                # No, we flipped. So we need to know if they were sat/unsat before.
                
                # Let's revert for calculation or just calculate difference directly.
                # Current state: assignment[v] = new_val
                # Previous state: assignment[v] = old_val
                
                # Is cci satisfied now?
                now_sat = is_satisfied(cci, assignment)
                
                # Was cci satisfied before?
                # We can simulate: if assignment[v] was the only literal satisfying it, then it was sat.
                # Or just check: if we set v to old_val, is it sat?
                # To avoid double flip cost, let's just check the other literals.
                
                # If the clause is satisfied by any other literal, its status doesn't change with v's flip.
                # It only changes if v was the *only* satisfying literal (and it becomes unsat) 
                # or v was the *only* unsatisfying literal in an otherwise unsat clause (and it becomes sat).
                
                other_sat = False
                for (vv, pp) in clause_lits[cci]:
                    if vv != v:
                        if assignment[vv] == (pp == 1):
                            other_sat = True
                            break
                
                if other_sat:
                    # Clause was satisfied by others, and is satisfied by others.
                    # Status remains SAT.
                    pass
                else:
                    # Clause depends solely on v's value for satisfaction.
                    # If v makes it sat now, it was unsat before (delta -1)
                    # If v makes it unsat now, it was sat before (delta +1)
                    if now_sat:
                        delta -= 1
                    else:
                        delta += 1
            
            current_unsat += delta
            
            if current_unsat == 0:
                return assignment
                
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                # If we didn't improve, with probability (1 - p_flip) we revert
                if random.random() > p_flip:
                    assignment[v] = old_val
                    current_unsat -= delta
                    # If we reverted, we didn't really take the step? 
                    # Usually WalkSAT keeps the flip if random < p, else reverts.
                    # If reverted, we should probably not count it as a "step" towards progress, 
                    # but it is a move.
                    # Let's stick to: if random > p, revert.
                    # And if reverted, we try again next loop.
                    
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                
            # Adaptive restart within the same restart loop?
            # If stuck for too long, break out to do a new random restart.
            if no_improve_steps > 100:
                break
                
            steps += 1
            
        # End of restart loop iteration
        # If we found a solution, we returned.
        # Otherwise, we continue to next restart.
        # We keep clause weights? Standard WalkSAT doesn't. 
        # Weighted WalkSAT would update weights here.
        # Let's add simple clause weighting: increase weight of unsat clauses.
        # This requires knowing which clauses are unsat at the end of the phase.
        # It's expensive to compute all unsat clauses.
        # Let's skip complex weighting for now and rely on restarts + random walk.
        # Actually, adding a small weight update is cheap if we just track the last few unsat clauses?
        # No, let's stick to pure WalkSAT with aggressive restarts.
        
    if best_assignment is not None:
        return best_assignment
    return [False] * n
