import random
import time
import sys
import math

def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance with 50 variables.
    Returns a list of booleans (assignment) that satisfies all clauses.
    
    Strategy:
    1. Efficient WalkSAT with clause weights and neighbor-aware variable selection.
    2. Maintain clause satisfaction counts for O(1) updates.
    3. Restarts with perturbation from the best known state.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 330s wall-clock limit, use 325s as safety margin
    deadline = time.time() + 325
    
    rng = random.Random(42)
    
    # Preprocessing: Map literals to variable indices (0-based)
    # Store clauses as tuples of (var, is_positive) for faster access?
    # Or just keep signed literals and use abs(l)-1.
    
    m = len(clauses)
    
    # Data structures for efficient bookkeeping
    # clause_status[i] = number of satisfied literals in clause i
    # clause_satisfied[i] = bool, whether clause i is satisfied
    clause_status = [0] * m
    clause_satisfied = [False] * m
    
    # var_to_clauses[v] = list of indices of clauses containing variable v
    var_to_clauses = [[] for _ in range(n)]
    
    # Precompute clause literal variable indices
    # clause_vars[i] = list of (var_idx, is_pos)
    clause_vars = []
    for i, c in enumerate(clauses):
        vars_in_c = []
        for l in c:
            v = abs(l) - 1
            is_pos = l > 0
            vars_in_c.append((v, is_pos))
            var_to_clauses[v].append(i)
        clause_vars.append(vars_in_c)

    def init_assignment():
        """Initialize assignment using literal bias heuristic."""
        # Count literal frequencies to guess planted assignment
        lit_freq = {}
        for c in clauses:
            for l in c:
                lit_freq[l] = lit_freq.get(l, 0) + 1
        
        assign = [False] * n
        for v in range(1, n + 1):
            pos_freq = lit_freq.get(v, 0)
            neg_freq = lit_freq.get(-v, 0)
            # If tied, randomize
            if pos_freq > neg_freq:
                assign[v-1] = True
            elif neg_freq > pos_freq:
                assign[v-1] = False
            else:
                assign[v-1] = rng.random() < 0.5
        return assign

    def update_clause_status(clause_idx, assign):
        """Update status of a specific clause based on current assignment."""
        count_sat = 0
        for v, is_pos in clause_vars[clause_idx]:
            if assign[v] == is_pos:
                count_sat += 1
        clause_status[clause_idx] = count_sat
        clause_satisfied[clause_idx] = (count_sat > 0)

    def full_update(assign):
        """Update all clause statuses. O(m*3)."""
        for i in range(m):
            update_clause_status(i, assign)

    def count_unsat(assign):
        """Count unsatisfied clauses using precomputed status."""
        # If we maintain status, we just sum
        return sum(1 for i in range(m) if not clause_satisfied[i])
        
    def get_unsat_clauses():
        """Get list of unsatisfied clause indices."""
        return [i for i in range(m) if not clause_satisfied[i]]

    # --- Main Solver Loop ---
    
    # Initial Assignment
    assignment = init_assignment()
    full_update(assignment)
    
    best_assignment = assignment[:]
    best_unsat = count_unsat(assignment)
    
    if best_unsat == 0:
        return best_assignment

    # WalkSAT Parameters
    p = 0.5  # Randomness parameter: 0 is greedy (min damage), 1 is random
    max_steps_per_restart = 2000
    max_restarts = 1000
    
    # We will track the number of unsatisfied clauses quickly.
    # To avoid O(m) scan for unsat clauses every step, we can maintain a set or list.
    # However, for n=50, m is likely small (planted 3-SAT usually has m ~ 4n = 200).
    # Scanning 200 clauses is fast. Let's stick to simple list generation for clarity, 
    # but optimize the inner loop.
    
    # Actually, maintaining a list of unsat clauses and updating it incrementally is better.
    # Let's try a hybrid: recompute unsat list every K steps or when it changes significantly?
    # No, let's just recompute the list of unsat indices. It's O(m). 
    # But we can avoid re-computing clause_status if we update incrementally.
    
    # Let's implement incremental update for clause_status.
    
    def flip_var(v, assign):
        """Flip variable v and update clause statuses incrementally. Returns delta_unsat."""
        old_val = assign[v]
        new_val = not old_val
        assign[v] = new_val
        
        delta_unsat = 0
        for c_idx in var_to_clauses[v]:
            # Determine if this clause was satisfied before
            was_sat = clause_satisfied[c_idx]
            
            # Update status
            # The literal for v in this clause
            # We need to know the polarity of v in this clause.
            # We stored clause_vars[c_idx] as list of (v, is_pos).
            # We need to find the entry for v. Since clause size is 3, this is fast.
            # Optimization: Store polarity in var_to_clauses? 
            # var_to_clauses[v] currently just has clause indices.
            # Let's store (clause_idx, is_pos) in var_to_clauses for faster access.
            pass
        
        # The above approach requires changing var_to_clauses structure.
        # Let's refactor var_to_clauses to store (clause_idx, is_pos_in_clause).
        
        return delta_unsat

    # Refactor var_to_clauses
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            is_pos = l > 0
            var_to_clauses[v].append((i, is_pos))

    def flip_var(v, assign):
        old_val = assign[v]
        new_val = not old_val
        assign[v] = new_val
        
        delta_unsat = 0
        for c_idx, is_pos in var_to_clauses[v]:
            was_sat = clause_satisfied[c_idx]
            
            # Update clause status
            # If the literal (v, is_pos) was satisfied, it becomes unsatisfied now.
            # If it was unsatisfied, it becomes satisfied now.
            lit_was_sat = (old_val == is_pos)
            
            if lit_was_sat:
                # Literal becomes unsatisfied
                clause_status[c_idx] -= 1
            else:
                # Literal becomes satisfied
                clause_status[c_idx] += 1
                
            is_sat_now = (clause_status[c_idx] > 0)
            
            if was_sat and not is_sat_now:
                delta_unsat += 1
            elif not was_sat and is_sat_now:
                delta_unsat -= 1
                
            clause_satisfied[c_idx] = is_sat_now
            
        return delta_unsat

    # Reset status with new assignment
    full_update(assignment)
    current_unsat = count_unsat(assignment)
    
    if current_unsat == 0:
        return assignment

    best_unsat = current_unsat
    best_assignment = assignment[:]
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        
        # If we've been restarting too much without improvement, maybe increase p or decrease p?
        # For planted instances, greedy (low p) often works well initially, but high p helps escape.
        # Let's vary p slightly or use a fixed good value like 0.5.
        
        unsat_list = get_unsat_clauses()
        
        if not unsat_list:
            return assignment
            
        # Local Search Phase
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            if current_unsat == 0:
                return assignment
                
            # Pick an unsatisfied clause
            # Weights: We can just pick uniformly from unsat_list
            # Or pick the one with fewest satisfied literals (all are 0, so uniform is fine)
            c_idx = rng.choice(unsat_list)
            
            # Get literals in this clause
            # We need to pick a variable to flip.
            # Calculate "damage" for each variable in the clause.
            # Damage = number of currently satisfied clauses that contain the *negation* of the literal we would flip to.
            # Actually, if we flip variable v, we change the truth value of v.
            # A clause is broken if it was satisfied ONLY by the literal (v, is_pos) and we flip v to !is_pos.
            # So for each variable v in the unsatisfied clause c_idx:
            #   Let is_pos be the polarity of v in c_idx.
            #   If we flip v, the literal (v, is_pos) becomes false.
            #   We break any clause c' that is currently satisfied, contains (v, is_pos), 
            #   and has NO OTHER satisfied literals.
            
            # Efficient damage calculation:
            # For each v in clause_vars[c_idx]:
            #   damage = 0
            #   for (c_prime, is_pos_prime) in var_to_clauses[v]:
            #       if c_prime == c_idx: continue # We are trying to satisfy this one
            #       if clause_satisfied[c_prime] and clause_status[c_prime] == 1 and is_pos_prime == is_pos:
            #           # This clause is satisfied only by (v, is_pos). Flipping v breaks it.
            #           damage += 1
            
            # This is O(1) per variable in the clause because var_to_clauses[v] size is bounded 
            # by clause density, but typically small.
            
            candidates = []
            total_weight = 0
            for v, is_pos in clause_vars[c_idx]:
                damage = 0
                for c_prime, is_pos_prime in var_to_clauses[v]:
                    if c_prime == c_idx:
                        continue
                    if clause_satisfied[c_prime] and clause_status[c_prime] == 1 and is_pos_prime == is_pos:
                        damage += 1
                candidates.append((v, damage))
                total_weight += damage
                
            # Weighted selection
            # Higher damage = higher weight (greedy prefers low damage, so we want to pick low damage).
            # Standard WalkSAT: Probability proportional to damage? 
            # No, usually "MinConflicts" picks the one with min damage.
            # WalkSAT picks with probability proportional to damage? 
            # Original WalkSAT: "flip a literal in the clause with probability p, otherwise flip the one that breaks the fewest other clauses".
            # Let's implement: With prob p, pick random. With prob 1-p, pick min damage.
            # Or: Weight = damage. Pick with prob proportional to weight? That would prefer high damage (bad).
            # We want to prefer LOW damage.
            # So weight = 1 / (damage + 1) ? Or just use the "p" strategy.
            
            # Let's use the p strategy:
            if rng.random() < p:
                # Random
                v_to_flip = rng.choice(clause_vars[c_idx])[0]
            else:
                # Greedy: Min damage
                min_damage = min(d for _, d in candidates)
                # Pick one of the min damage candidates
                min_cands = [v for v, d in candidates if d == min_damage]
                v_to_flip = rng.choice(min_cands)
                
            # Perform flip
            delta = flip_var(v_to_flip, assignment)
            current_unsat += delta
            
            # Update unsat_list
            # Instead of full recompute, we can just recompute if delta != 0
            # But maintaining the list incrementally is tricky.
            # Given n=50, m~200, recomputing unsat_list is O(m) = 200 ops.
            # This is very fast. Let's just recompute every step? 
            # Or every few steps?
            # Let's recompute only if we need to pick the next clause.
            
            # Check if solved
            if current_unsat == 0:
                return assignment
                
            # Update best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                
        # End of local search steps for this restart
        
        # Restart strategy:
        # If we haven't improved in the last restart, perturb the best assignment.
        # Or just start from best_assignment with random flips.
        
        if restart_count % 10 == 0:
            # Full random restart
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb best
            assignment = best_assignment[:]
            # Flip 1-3 random variables
            num_flips = rng.randint(1, 3)
            for _ in range(num_flips):
                v = rng.randrange(n)
                flip_var(v, assignment) # This updates status too
                
        # Recalculate status after perturbation/random restart
        # flip_var updates status, so if we used random restart, we need full_update.
        if restart_count % 10 == 0:
            full_update(assignment)
        
        current_unsat = count_unsat(assignment)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            
        if best_unsat == 0:
            return best_assignment

    # Time's up
    return best_assignment

# EVOLVE-BLOCK-END
