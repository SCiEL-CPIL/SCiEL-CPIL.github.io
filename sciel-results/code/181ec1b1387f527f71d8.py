import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess clauses for faster access
    # Store as list of (var_index, is_positive)
    # var_index is 0-based
    preprocessed_clauses = []
    for clause in clauses:
        pre_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            pre_clause.append((var_idx, is_pos))
        preprocessed_clauses.append(pre_clause)
        
    m = len(preprocessed_clauses)
    
    # Helper to check if a clause is satisfied
    def is_satisfied(clause_idx, assignment):
        c = preprocessed_clauses[clause_idx]
        for var_idx, is_pos in c:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat.append(i)
        return unsat

    def evaluate_flip(clause_idx, var_idx, current_assignment):
        # Returns the change in number of satisfied clauses if we flip var_idx
        # This is expensive to compute for all vars in the clause if we do it naively, 
        # but we only need it for the 3 literals in the chosen clause.
        # Actually, WalkSAT usually picks a random literal or the best one from the unsat clause.
        # Let's implement the "best literal" heuristic.
        # We want to maximize the number of satisfied clauses after the flip.
        # Flipping var_idx affects all clauses containing var_idx.
        # This is O(m) per flip evaluation. 
        # Given m ~ 300, this is fine.
        
        # However, a simpler heuristic often used:
        # Pick a literal from the unsat clause.
        # If we pick the literal that is currently False (which they all are in an unsat clause), 
        # flipping it makes it True.
        # The "weight" of a literal is the number of other clauses it satisfies if flipped.
        
        pass # We will implement a simpler version first: random literal selection with restarts.

    rng = random.Random(0)
    start_time = time.time()
    deadline = 290.0 # Safety margin
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Maximum steps per restart
    max_steps_per_restart = 10000
    max_restarts = 100 # Hard cap on restarts, though time limit is primary
    
    for restart_iter in range(max_restarts):
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial evaluation
        unsat_clauses = get_unsat_clauses(assignment)
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        if current_unsat == 0:
            return assignment
        
        steps_in_restart = 0
        last_improvement = 0
        current_best_in_restart = current_unsat
        
        while steps_in_restart < max_steps_per_restart:
            # Time check
            if time.time() > deadline:
                break
            
            # Pick a random unsatisfied clause
            if not unsat_clauses:
                return assignment
            
            clause_idx = rng.choice(unsat_clauses)
            clause = preprocessed_clauses[clause_idx]
            
            # Pick a literal to flip
            # Heuristic: Pick the literal that satisfies the most other clauses (greedy)
            # Or random. Let's try a mix.
            # Calculate score for each literal in the clause
            best_var = -1
            best_score = -1
            scores = []
            
            for var_idx, is_pos in clause:
                # If we flip var_idx, what is the impact?
                # This requires checking all clauses containing var_idx.
                # To keep it fast, we might just pick random.
                # Given n=50, m~300, a full evaluation of all 3 literals' impact is 3 * (avg clauses per var) ~ 3 * 18 = 54 checks.
                # This is very fast.
                
                # Let's compute the delta in satisfied clauses.
                # Current state: clause_idx is unsat.
                # If we flip var_idx, clause_idx becomes sat (gain +1).
                # Other clauses containing var_idx might become unsat (loss -1 each) or remain sat.
                
                delta = 1 # For the current clause
                
                # Check other clauses
                for j in range(m):
                    if j == clause_idx:
                        continue
                    c_j = preprocessed_clauses[j]
                    # Does c_j contain var_idx?
                    # This is inefficient. Let's precompute variable-to-clause map.
                    pass
                
                # Optimization: Precompute var_to_clauses
                # We will do this outside the loop.
                
            # Let's restart the logic with precomputed maps for efficiency.
            
            break # Break out to restructure with precomputed maps

    # RESTRUCTURED APPROACH WITH PRECOMPUTED MAPS
    
    # Precompute variable to clause indices map
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(preprocessed_clauses):
        for var_idx, is_pos in clause:
            var_to_clauses[var_idx].append(i)

    def evaluate_flip_impact(var_idx, assignment, current_unsat_clauses_set):
        # Returns the change in number of unsatisfied clauses if we flip var_idx
        # delta = (new_unsat - old_unsat)
        # We want to minimize unsat, so we want negative delta.
        
        delta = 0
        for j in var_to_clauses[var_idx]:
            c = preprocessed_clauses[j]
            # Check if clause j is currently satisfied
            is_sat = False
            for v, p in c:
                if assignment[v] == p:
                    is_sat = True
                    break
            
            if is_sat:
                # If it's satisfied, does it remain satisfied after flip?
                # It remains satisfied if there is another literal in the clause that is true and doesn't involve var_idx, 
                # OR if the literal involving var_idx was the only true one, it becomes false, 
                # BUT if there are other true literals, it stays true.
                # Actually, simpler: 
                # Count true literals in clause j currently.
                true_count = 0
                for v, p in c:
                    if assignment[v] == p:
                        true_count += 1
                
                # If true_count > 1, flipping one variable doesn't make it unsat (unless it was the only one? No, if >1, at least one remains).
                # If true_count == 1, and that one is var_idx, then flipping makes it 0 -> unsat.
                # If true_count == 1, and that one is NOT var_idx, flipping var_idx (which is false) makes it true? No, var_idx is false in an unsat clause context?
                # Wait, we are evaluating a flip. 
                # Let's just re-evaluate the clause after the hypothetical flip. It's only 3 literals.
                
                # Hypothetical assignment: flip var_idx
                new_is_sat = False
                for v, p in c:
                    if v == var_idx:
                        val = not assignment[v]
                    else:
                        val = assignment[v]
                    if val == p:
                        new_is_sat = True
                        break
                
                if is_sat and not new_is_sat:
                    delta -= 1 # Lost a satisfied clause
                elif not is_sat and new_is_sat:
                    delta += 1 # Gained a satisfied clause
                # else no change
            
        return delta # Positive means more satisfied (good), negative means fewer satisfied (bad)
        # Wait, I defined delta as change in *unsatisfied*?
        # Let's stick to "change in satisfied count".
        # If delta > 0, we satisfy more clauses.
        
    # Let's refine evaluate_flip_impact to return gain in satisfied clauses.
    
    def gain_from_flip(var_idx, assignment):
        gain = 0
        for j in var_to_clauses[var_idx]:
            c = preprocessed_clauses[j]
            # Current status
            cur_sat = False
            for v, p in c:
                if assignment[v] == p:
                    cur_sat = True
                    break
            
            # New status
            new_sat = False
            for v, p in c:
                if v == var_idx:
                    val = not assignment[v]
                else:
                    val = assignment[v]
                if val == p:
                    new_sat = True
                    break
            
            if not cur_sat and new_sat:
                gain += 1
            elif cur_sat and not new_sat:
                gain -= 1
        return gain

    # Main Loop
    rng = random.Random(0)
    start_time = time.time()
    deadline = 290.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_steps_per_restart = 5000
    max_restarts = 50
    
    for restart_iter in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial check
        unsat_count = 0
        unsat_clauses = []
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_count += 1
                unsat_clauses.append(i)
                
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        if unsat_count == 0:
            return assignment
            
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
                
            if not unsat_clauses:
                return assignment
                
            # Pick random unsat clause
            clause_idx = rng.choice(unsat_clauses)
            clause = preprocessed_clauses[clause_idx]
            
            # Pick literal to flip
            # Strategy: 50% chance pick best literal, 50% random
            if rng.random() < 0.5:
                # Greedy: pick var with max gain
                best_var = -1
                best_gain = -100
                for var_idx, is_pos in clause:
                    g = gain_from_flip(var_idx, assignment)
                    if g > best_gain:
                        best_gain = g
                        best_var = var_idx
                var_to_flip = best_var
            else:
                var_idx, is_pos = rng.choice(clause)
                var_to_flip = var_idx
                
            # Perform flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update unsat clauses
            # This is the expensive part if done naively.
            # But n=50, m~300. 
            # We can just recalculate all? 300 * 3 = 900 ops.
            # 5000 steps * 900 = 4.5 million ops. Very fast.
            
            new_unsat_count = 0
            new_unsat_clauses = []
            for i in range(m):
                if not is_satisfied(i, assignment):
                    new_unsat_count += 1
                    new_unsat_clauses.append(i)
                    
            if new_unsat_count < best_unsat_count:
                best_unsat_count = new_unsat_count
                best_assignment = assignment[:]
                
            if new_unsat_count == 0:
                return assignment
                
            unsat_count = new_unsat_count
            unsat_clauses = new_unsat_clauses
            steps += 1
            
    return best_assignment

