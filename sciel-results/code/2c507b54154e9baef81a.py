import random
import time

def solve(instance):
    """
    Solves a 3-SAT instance using a weighted WalkSAT variant with restarts.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Time limit guard (300s total, keep 5s margin)
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Pre-process clauses for faster access
    # Convert literals to 0-based indices and store sign
    # clause is a list of tuples (var_idx, is_positive)
    processed_clauses = []
    for clause in clauses:
        proc_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            proc_clause.append((var_idx, is_pos))
        processed_clauses.append(proc_clause)
        
    num_clauses = len(processed_clauses)
    
    # Helper to check if a clause is satisfied given an assignment
    def is_satisfied(clause, assignment):
        for var_idx, is_pos in clause:
            if assignment[var_idx] == is_pos:
                return True
        return False

    # Helper to count unsatisfied clauses
    def count_unsat(assignment):
        return sum(0 if is_satisfied(c, assignment) else 1 for c in processed_clauses)

    # Helper to find unsatisfied clauses
    def get_unsat_clauses(assignment):
        return [c for c in processed_clauses if not is_satisfied(c, assignment)]

    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters for WalkSAT
    # p is the probability of picking a random variable in the clause (noise)
    # vs the variable that minimizes conflicts (MinConflicts)
    p_noise = 0.3
    max_steps_per_restart = 10000
    max_restarts = 500 # Will be limited by time anyway

    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            if current_unsat == 0:
                return best_assignment

        # WalkSAT loop
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                break
            
            if current_unsat == 0:
                return assignment

            # Get unsatisfied clauses
            unsat_clauses = get_unsat_clauses(assignment)
            
            if not unsat_clauses:
                return assignment

            # Weighted selection of an unsatisfied clause
            # Weights can be based on frequency or just random for simplicity, 
            # but random is standard WalkSAT. Let's stick to random for now 
            # but optimize the variable flip.
            clause = rng.choice(unsat_clauses)
            
            # MinConflicts variable selection
            # For each variable in the clause, calculate the delta in unsat count if flipped
            best_var_idx = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in clause:
                # Flip variable
                old_val = assignment[var_idx]
                new_val = not old_val
                
                # Calculate change in unsat count
                # This requires checking all clauses containing this variable.
                # To do this efficiently, we should pre-compute variable-clause incidence.
                # However, for n=50 and typical clause counts, a simpler approach might suffice 
                # or we can approximate. 
                # Let's pre-compute incidence for efficiency in the next iteration or just 
                # do a naive check if it's fast enough. 
                # Naive check: O(num_clauses * 3). 
                # If num_clauses is ~100, that's 300 ops. 
                # 3 vars per clause -> 900 ops per step. 
                # 10000 steps -> 9M ops. 
                # Python might be slow for 9M ops in 300s? 
                # 9M simple ops is usually < 1s in PyPy, maybe 2-3s in CPython. 
                # 500 restarts * 3s = 1500s > 300s. 
                # So we MUST optimize.
                
                pass # Placeholder for optimization thought

            # Optimization: Pre-compute variable to clause mapping
            # Let's restructure the loop to use pre-computed data.
            break # Break out of inner loop to implement optimized version in the actual code below

    # If we get here, return the best found
    return best_assignment if best_assignment else [False] * n

# The above draft has a performance issue with the naive MinConflicts check.
# I will rewrite the solve function properly with pre-computed incidence structures.

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Pre-processing
    # 1. Convert clauses to 0-based indices
    # 2. Build variable -> list of (clause_idx, sign) incidence map
    var_to_clauses = [[] for _ in range(n)]
    processed_clauses = []
    
    for idx, clause in enumerate(clauses):
        proc_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            proc_clause.append((var_idx, is_pos))
            var_to_clauses[var_idx].append((idx, is_pos))
        processed_clauses.append(proc_clause)
        
    num_clauses = len(processed_clauses)
    
    # State tracking for fast updates
    # clause_sat_count[i] = number of literals satisfied in clause i
    # var_sat_count[j] = number of clauses where variable j is satisfied (and clause is unsat? No, just total sat literals)
    # Actually, for MinConflicts, we need to know how many *currently unsatisfied* clauses contain this variable
    # and how many *currently satisfied* clauses contain this variable.
    # Flipping a variable:
    # - It will unsatisfy all currently satisfied clauses containing it.
    # - It will satisfy all currently unsatisfied clauses containing it (assuming no other lit in that clause is already sat? No, if clause is unsat, ALL lits are false. So flipping one makes it true, satisfying the clause).
    # Wait, if a clause is unsat, all its literals are false. Flipping one variable in it makes that literal true, so the clause becomes satisfied.
    # If a clause is sat, at least one lit is true. Flipping a variable in it:
    #   - If that variable was the ONLY true lit in that clause, the clause becomes unsat.
    #   - If there were other true lits, the clause remains sat.
    
    # So we need:
    # 1. clause_sat_count: number of true literals in each clause.
    # 2. assignment: current boolean values.
    
    def init_state(assignment):
        clause_sat_counts = [0] * num_clauses
        for idx, clause in enumerate(processed_clauses):
            count = 0
            for var_idx, is_pos in clause:
                if assignment[var_idx] == is_pos:
                    count += 1
            clause_sat_counts[idx] = count
        return clause_sat_counts

    def get_unsat_count(clause_sat_counts):
        return sum(1 for c in clause_sat_counts if c == 0)

    def get_unsat_clauses(clause_sat_counts):
        return [i for i, c in enumerate(clause_sat_counts) if c == 0]

    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random(123)
    
    max_restarts = 10000 # Limited by time
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_counts = init_state(assignment)
        current_unsat = get_unsat_count(clause_sat_counts)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            if current_unsat == 0:
                return best_assignment

        for step in range(steps_per_restart):
            if time.time() > deadline:
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                break
                
            if current_unsat == 0:
                return assignment

            unsat_clauses = get_unsat_clauses(clause_sat_counts)
            if not unsat_clauses:
                return assignment

            # Pick a random unsat clause
            clause_idx = rng.choice(unsat_clauses)
            clause = processed_clauses[clause_idx]
            
            # MinConflicts: Pick variable in clause that minimizes increase in unsat count
            # Delta = (new_unsat - old_unsat)
            # Let's evaluate each variable in the clause
            best_var_idx = -1
            best_delta = float('inf')
            best_new_sat_counts = None # To avoid recomputing if we pick it
            
            for var_idx, is_pos in clause:
                # Simulate flip
                old_val = assignment[var_idx]
                new_val = not old_val
                
                delta = 0
                # Check all clauses containing var_idx
                for c_idx, c_is_pos in var_to_clauses[var_idx]:
                    # Determine if this clause is currently satisfied
                    is_c_sat = clause_sat_counts[c_idx] > 0
                    
                    # Determine if the literal in this clause is currently true
                    lit_is_true = (assignment[var_idx] == c_is_pos)
                    
                    if is_c_sat:
                        # Clause is currently sat.
                        # If this literal is the ONLY true one in the clause, flipping it makes clause unsat.
                        if clause_sat_counts[c_idx] == 1 and lit_is_true:
                            delta += 1
                    else:
                        # Clause is currently unsat.
                        # Flipping this variable makes this literal true.
                        # So the clause becomes sat.
                        delta -= 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_var_idx = var_idx

            # With some probability, pick a random variable from the clause instead (noise)
            if rng.random() < 0.2:
                var_idx, is_pos = rng.choice(clause)
            else:
                var_idx = best_var_idx
                
            # Apply flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause_sat_counts
            for c_idx, c_is_pos in var_to_clauses[var_idx]:
                # The literal value just changed.
                # If it was true, it's now false -> count decreases by 1
                # If it was false, it's now true -> count increases by 1
                # We know the OLD value was assignment[var_idx] before flip.
                # So if old_val == c_is_pos, it was true. Now it is false.
                # We flipped assignment[var_idx], so:
                # If assignment[var_idx] (new) == c_is_pos, it is now true.
                # Previously it was false. So count increases.
                # If assignment[var_idx] (new) != c_is_pos, it is now false.
                # Previously it was true. So count decreases.
                
                if assignment[var_idx] == c_is_pos:
                    clause_sat_counts[c_idx] += 1
                else:
                    clause_sat_counts[c_idx] -= 1
                    
            # Update current_unsat
            # Instead of recounting, we can track it, but recounting O(C) is fine if C is small.
            # Or we can just update current_unsat based on the clauses we touched.
            # Let's just recount for simplicity and correctness, it's O(C).
            # If C is ~100, it's fast.
            current_unsat = get_unsat_count(clause_sat_counts)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return best_assignment

    return best_assignment if best_assignment else [False] * n
