import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    
    for idx, c in enumerate(clauses):
        for l in c:
            var = abs(l) - 1
            if l > 0:
                pos_clauses[var].append(idx)
            else:
                neg_clauses[var].append(idx)
                
    start_time = time.time()
    deadline = start_time + 290.0
    
    MAX_STEPS_PER_RESTART = 5000
    MAX_RESTARTS = 100
    
    rng = random.Random()
    
    best_assignment = None
    min_unsat_seen = float('inf')
    
    for restart in range(MAX_RESTARTS):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause states
        # clause_sat[i] = True if clause i is satisfied
        # clause_true_count[i] = number of true literals in clause i
        clause_sat = [False] * len(clauses)
        clause_true_count = [0] * len(clauses)
        
        for idx, c in enumerate(clauses):
            count = 0
            for l in c:
                if assignment[abs(l)-1] == (l > 0):
                    count += 1
            clause_true_count[idx] = count
            if count > 0:
                clause_sat[idx] = True
            else:
                clause_sat[idx] = False
                
        unsat_count = sum(1 for s in clause_sat if not s)
        unsat_indices = [i for i, s in enumerate(clause_sat) if not s]
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < min_unsat_seen:
            min_unsat_seen = unsat_count
            best_assignment = assignment[:]
            
        steps = 0
        p_walk = 0.5
        
        while unsat_count > 0 and steps < MAX_STEPS_PER_RESTART:
            if time.time() > deadline:
                if unsat_count < min_unsat_seen:
                    min_unsat_seen = unsat_count
                    best_assignment = assignment[:]
                return best_assignment
                
            if not unsat_indices:
                break
                
            clause_idx = rng.choice(unsat_indices)
            clause = clauses[clause_idx]
            
            if rng.random() < p_walk:
                literal = rng.choice(clause)
            else:
                best_literal = None
                min_loss = float('inf')
                
                for l in clause:
                    var = abs(l) - 1
                    loss = 0
                    
                    if assignment[var]:
                        # var is True. +var is True.
                        # Check pos_clauses[var]
                        for c_idx in pos_clauses[var]:
                            # If clause c_idx is currently satisfied (clause_sat[c_idx] is True)
                            # and clause_true_count[c_idx] == 1, then it is satisfied ONLY by this literal.
                            # Wait, is it satisfied by THIS literal?
                            # The literal in c_idx corresponding to var is +var.
                            # Since var is True, +var is True.
                            # So if clause_true_count[c_idx] == 1, then +var is the only true literal.
                            # Flipping var will make +var False, so clause becomes unsat.
                            if clause_sat[c_idx] and clause_true_count[c_idx] == 1:
                                loss += 1
                    else:
                        # var is False. -var is True.
                        # Check neg_clauses[var]
                        for c_idx in neg_clauses[var]:
                            if clause_sat[c_idx] and clause_true_count[c_idx] == 1:
                                loss += 1
                                
                    if loss < min_loss:
                        min_loss = loss
                        best_literal = l
                        
                literal = best_literal
                
            var = abs(literal) - 1
            old_val = assignment[var]
            assignment[var] = not old_val
            
            # Update clause states for affected clauses
            # Affected clauses are those containing var.
            # If old_val was True:
            #   +var becomes False. -var becomes True.
            #   For c in pos_clauses[var]: clause_true_count[c] -= 1
            #   For c in neg_clauses[var]: clause_true_count[c] += 1
            # If old_val was False:
            #   +var becomes True. -var becomes False.
            #   For c in pos_clauses[var]: clause_true_count[c] += 1
            #   For c in neg_clauses[var]: clause_true_count[c] -= 1
            
            if old_val:
                # var True -> False
                for c_idx in pos_clauses[var]:
                    clause_true_count[c_idx] -= 1
                    if clause_true_count[c_idx] == 0:
                        if clause_sat[c_idx]:
                            clause_sat[c_idx] = False
                            unsat_count += 1
                            # Add to unsat_indices? 
                            # Managing the list dynamically is tricky.
                            # We can just rebuild unsat_indices every few steps or use a set.
                            # Or just check `if not clause_sat[i]` when picking.
                            # For simplicity and speed, let's maintain a set of unsat indices.
                    # If it was unsat, it stays unsat (count was 0, now -1? No, count cannot be negative if we track correctly).
                    # Wait, if clause was unsat (count 0), and we decrement, it becomes -1?
                    # No, if clause was unsat, it means no literals were true.
                    # If +var was true, the clause would have been sat.
                    # So if clause was unsat, +var was false.
                    # So if we flip var from True to False, +var becomes False.
                    # If the clause was unsat, it didn't have +var as true.
                    # So count was 0. Now +var is False. Count remains 0?
                    # No, the literal +var was true before?
                    # If old_val was True, +var was True.
                    # So the clause MUST have been satisfied (count >= 1).
                    # So we only decrement clauses that were satisfied.
                    # If count becomes 0, it becomes unsat.
                    
                for c_idx in neg_clauses[var]:
                    clause_true_count[c_idx] += 1
                    if not clause_sat[c_idx] and clause_true_count[c_idx] > 0:
                        clause_sat[c_idx] = True
                        unsat_count -= 1
            else:
                # var False -> True
                for c_idx in pos_clauses[var]:
                    clause_true_count[c_idx] += 1
                    if not clause_sat[c_idx] and clause_true_count[c_idx] > 0:
                        clause_sat[c_idx] = True
                        unsat_count -= 1
                for c_idx in neg_clauses[var]:
                    clause_true_count[c_idx] -= 1
                    if clause_true_count[c_idx] == 0:
                        if clause_sat[c_idx]:
                            clause_sat[c_idx] = False
                            unsat_count += 1
                            
            # Rebuild unsat_indices if it got too stale?
            # Or just use a set and update it.
            # Let's use a list and rebuild it if it's empty or every 100 steps?
            # Actually, `rng.choice(unsat_indices)` fails if empty.
            # We check `if not unsat_indices: break` at the start of loop.
            # But we need to update `unsat_indices` to reflect changes.
            # Rebuilding is O(M). 5000 steps * 200 = 1M ops. Negligible.
            unsat_indices = [i for i, s in enumerate(clause_sat) if not s]
            
            steps += 1
            
        if unsat_count == 0:
            return assignment
            
        if unsat_count < min_unsat_seen:
            min_unsat_seen = unsat_count
            best_assignment = assignment[:]
            
    return best_assignment
