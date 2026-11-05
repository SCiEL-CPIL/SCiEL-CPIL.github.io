"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """Solve a planted 3-SAT instance using a hybrid WalkSAT/MiniSAT-like approach.
    
    Strategy:
    1. Quick unit propagation + pure literal elimination to simplify.
    2. If not solved, use WalkSAT with adaptive clause selection and variable scoring.
    3. Track best assignment by number of unsatisfied clauses.
    4. Deadline guard: return best found before 320s wall-clock.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    deadline = time.time() + 320.0
    
    rng = random.Random(42)
    
    # --- Step 1: Try simple initialization with unit propagation ---
    def try_solve_simple(clauses, n):
        """Try to solve via unit propagation and pure literal elimination."""
        # Copy clauses as lists of literals
        cl = [list(c) for c in clauses]
        assignment = [None] * n  # None means unassigned
        
        def propagate():
            """Unit propagation and pure literal elimination."""
            changed = True
            while changed:
                changed = False
                # Unit propagation
                new_cl = []
                for c in cl:
                    # Check if clause is satisfied
                    sat = False
                    unit_lit = None
                    for l in c:
                        idx = abs(l) - 1
                        if assignment[idx] is not None:
                            if assignment[idx] == (l > 0):
                                sat = True
                                break
                        else:
                            unit_lit = l
                    if sat:
                        continue
                    if len(c) == 1:
                        # Unit clause
                        l = c[0]
                        idx = abs(l) - 1
                        if assignment[idx] is None:
                            assignment[idx] = (l > 0)
                            changed = True
                        elif assignment[idx] != (l > 0):
                            return False  # Contradiction
                    else:
                        new_cl.append(c)
                cl[:] = new_cl
                
                # Pure literal elimination
                if not cl:
                    return True
                pos_count = [0] * n
                neg_count = [0] * n
                for c in cl:
                    for l in c:
                        idx = abs(l) - 1
                        if assignment[idx] is None:
                            if l > 0:
                                pos_count[idx] += 1
                            else:
                                neg_count[idx] += 1
                for i in range(n):
                    if assignment[i] is None:
                        if pos_count[i] > 0 and neg_count[i] == 0:
                            assignment[i] = True
                            changed = True
                        elif neg_count[i] > 0 and pos_count[i] == 0:
                            assignment[i] = False
                            changed = True
            # Check if all clauses satisfied
            for c in cl:
                sat = False
                for l in c:
                    idx = abs(l) - 1
                    if assignment[idx] is not None and assignment[idx] == (l > 0):
                        sat = True
                        break
                if not sat:
                    return False
            return True
        
        if propagate():
            # Fill in any unassigned variables
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = rng.random() < 0.5
            return assignment
        return None
    
    # Try simple solve first
    result = try_solve_simple(clauses, n)
    if result is not None:
        return result
    
    # --- Step 2: WalkSAT with improvements ---
    # Initialize assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Precompute clause data for faster access
    # Store clauses as tuples of (var_index, is_positive)
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            cd.append((abs(l) - 1, l > 0))
        clause_data.append(cd)
    
    num_clauses = len(clause_data)
    
    def count_unsat(clause_idx=None):
        """Count unsatisfied clauses. If clause_idx given, check just that one."""
        if clause_idx is not None:
            c = clause_data[clause_idx]
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    return 1  # satisfied
            return 0
        cnt = 0
        for c in clause_data:
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    cnt += 1
                    break
        return cnt
    
    # Initial unsat count
    unsat_count = num_clauses - count_unsat()
    
    best_assignment = assignment[:]
    best_unsat = unsat_count
    
    # Variable activity scores for better selection
    var_activity = [0.0] * n
    
    # Adaptive parameters
    alpha = 0.01  # Weight for greedy vs random variable selection
    flip_budget_per_restart = 5000
    max_restarts = 5000
    
    restart_count = 0
    global_step = 0
    
    while restart_count < max_restarts and time.time() < deadline:
        # Check if solved
        if best_unsat == 0:
            return best_assignment
        
        # WalkSAT loop
        for step in range(flip_budget_per_restart):
            if time.time() >= deadline - 5:
                break
            
            global_step += 1
            
            # Find an unsatisfied clause
            # Use a random sampling approach for speed
            found_clause = None
            for _ in range(50):
                ci = rng.randrange(num_clauses)
                if count_unsat(ci) == 0:
                    found_clause = ci
                    break
            if found_clause is None:
                # Linear scan
                for ci in range(num_clauses):
                    if count_unsat(ci) == 0:
                        found_clause = ci
                        break
            if found_clause is None:
                break  # All satisfied
            
            c = clause_data[found_clause]
            
            # Score each variable in the clause
            best_var = -1
            best_score = -1.0
            for var_idx, is_pos in c:
                # If we flip this variable, how many other clauses are affected?
                # Count clauses where this variable is the only false literal (gain)
                # vs clauses where it's the only true literal (loss)
                gain = 0
                loss = 0
                for other_ci in range(num_clauses):
                    if other_ci == found_clause:
                        continue
                    oc = clause_data[other_ci]
                    # Check if this variable appears in other clause
                    var_in_other = False
                    other_satisfied = False
                    for ov, op in oc:
                        if ov == var_idx:
                            var_in_other = True
                            # Current state: is this variable satisfying this clause?
                            if assignment[var_idx] == op:
                                other_satisfied = True
                            break
                    if var_in_other:
                        if other_satisfied:
                            loss += 1
                        else:
                            gain += 1
                
                score = gain - loss + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var == -1:
                break
            
            # Decide whether to flip based on greedy vs random
            if rng.random() < alpha:
                # Flip the best variable
                flip_var = best_var
            else:
                # Flip a random variable in the clause
                var_idx, _ = c[rng.randrange(len(c))]
                flip_var = var_idx
            
            # Flip
            assignment[flip_var] = not assignment[flip_var]
            
            # Update unsat count approximately
            # Recalculate for simplicity (could optimize with delta)
            new_unsat = num_clauses - count_unsat()
            unsat_count = new_unsat
            
            # Update activity
            var_activity[flip_var] *= 0.99
            var_activity[flip_var] += 1.0
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            if unsat_count == 0:
                return assignment
        
        # Restart: randomize assignment, keep some structure
        restart_count += 1
        if restart_count % 10 == 0:
            # Full random restart
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Partial restart: flip some variables
            for i in range(n):
                if rng.random() < 0.3:
                    assignment[i] = not assignment[i]
        
        unsat_count = num_clauses - count_unsat()
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
