import random
import time

def solve(instance):
    """
    Solves a 3-SAT instance using a weighted WalkSAT variant with restarts
    and efficient clause satisfaction tracking.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Time limit guard (330s total, keep 10s margin)
    start_time = time.time()
    deadline = start_time + 320.0

    # Pre-process clauses for faster access
    # Convert literals to 0-based indices and store sign
    # clause is a list of tuples (var_idx, is_positive)
    processed_clauses = []
    # var_to_clauses[var_idx] = list of (clause_idx, is_positive_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    
    for idx, clause in enumerate(clauses):
        proc_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            proc_clause.append((var_idx, is_pos))
            var_to_clauses[var_idx].append((idx, is_pos))
        processed_clauses.append(proc_clause)
        
    num_clauses = len(processed_clauses)
    
    # Helper to initialize clause satisfaction counts
    # clause_sat_count[i] = number of literals in clause i that are true
    def init_clause_sat_counts(assignment):
        counts = [0] * num_clauses
        for idx, clause in enumerate(processed_clauses):
            count = 0
            for var_idx, is_pos in clause:
                if assignment[var_idx] == is_pos:
                    count += 1
            counts[idx] = count
        return counts

    # Helper to count unsatisfied clauses
    def count_unsat(clause_sat_counts):
        return sum(1 for c in clause_sat_counts if c == 0)

    # Helper to get list of unsatisfied clause indices
    def get_unsat_clauses(clause_sat_counts):
        return [i for i, c in enumerate(clause_sat_counts) if c == 0]

    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters for WalkSAT
    # p_noise is the probability of picking a random variable in the clause
    p_noise = 0.3
    # We use a dynamic step limit based on time to allow more restarts if fast,
    # but cap it to prevent infinite loops in pathological cases.
    max_steps_per_restart = 20000
    max_restarts = 5000 # Will be limited by time anyway

    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_sat_counts = init_clause_sat_counts(assignment)
        current_unsat = count_unsat(clause_sat_counts)
        
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
            unsat_clauses = get_unsat_clauses(clause_sat_counts)
            
            if not unsat_clauses:
                return assignment

            # Weighted selection of an unsatisfied clause
            # Standard WalkSAT picks uniformly.
            clause_idx = rng.choice(unsat_clauses)
            clause = processed_clauses[clause_idx]
            
            # MinConflicts variable selection
            # For each variable in the clause, calculate the delta in unsat count if flipped
            best_var_idx = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in clause:
                # Calculate change in unsat count if we flip var_idx
                # We iterate through all clauses containing var_idx
                
                delta = 0
                
                for c_idx, c_is_pos in var_to_clauses[var_idx]:
                    # Current state of clause c_idx
                    is_c_sat = clause_sat_counts[c_idx] > 0
                    # Current truth value of the literal in c_idx for var_idx
                    lit_is_true = (assignment[var_idx] == c_is_pos)
                    
                    if is_c_sat:
                        # Clause is currently satisfied.
                        # If we flip var_idx, the literal flips.
                        # If the literal was True, it becomes False.
                        # The clause becomes unsat ONLY IF this was the only true literal.
                        if lit_is_true and clause_sat_counts[c_idx] == 1:
                            delta += 1
                    else:
                        # Clause is currently unsatisfied.
                        # All literals are False.
                        # If we flip var_idx, the literal becomes True.
                        # The clause becomes satisfied.
                        delta -= 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_var_idx = var_idx

            # With some probability, pick a random variable from the clause instead (noise)
            if rng.random() < p_noise:
                var_idx, _ = rng.choice(clause)
            else:
                var_idx = best_var_idx
                
            # Apply flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause_sat_counts
            # We need to update the counts for all clauses containing var_idx
            for c_idx, c_is_pos in var_to_clauses[var_idx]:
                # The literal value just changed.
                # If it was true (old_val == c_is_pos), it is now false. Count decreases by 1.
                # If it was false (old_val != c_is_pos), it is now true. Count increases by 1.
                
                # We can check the NEW value to determine the direction of change relative to old?
                # No, simpler:
                # Old value was assignment[var_idx] BEFORE flip.
                # New value is assignment[var_idx] AFTER flip.
                # If New Value == c_is_pos, the literal is now True. It was False before. So count +1.
                # If New Value != c_is_pos, the literal is now False. It was True before. So count -1.
                
                if assignment[var_idx] == c_is_pos:
                    clause_sat_counts[c_idx] += 1
                else:
                    clause_sat_counts[c_idx] -= 1
                    
            # Update current_unsat
            # Recounting is O(C). With C ~ 100-150, this is fast.
            # Optimized update:
            # We could track which clauses changed state (0<->1) but recounting is simpler and likely fast enough
            # given the savings from O(1) clause satisfaction check in MinConflicts.
            current_unsat = count_unsat(clause_sat_counts)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return best_assignment

    # If we get here, return the best found
    return best_assignment if best_assignment else [False] * n
