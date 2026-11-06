import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin
    
    m = len(clauses)
    if m == 0:
        return [False] * n

    # Convert clauses to numpy arrays for vectorized operations
    # lit is 1-based signed. 
    # var index = abs(lit) - 1
    # sign = 1 if lit > 0 else -1
    # We store (var_idx, sign) where sign is 1 for positive, -1 for negative.
    
    # Create a numpy array of shape (m, 3) for 3-SAT
    # If clauses might be shorter, we pad with -1 (dummy var)
    max_len = max(len(c) for c in clauses)
    if max_len > 3:
        # Task says 3-SAT, but handle just in case by truncating or erroring? 
        # Assuming 3-SAT as per contract.
        pass
        
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            if j < 3:
                var_idx = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                clause_vars[i, j] = var_idx
                clause_signs[i, j] = sign
            else:
                # If clause is longer than 3, we can't fit in (m,3). 
                # But task says 3-SAT. If not, we might need to handle generally.
                # For now, assume max 3. If >3, we drop extra literals? No, that changes problem.
                # Let's assume standard 3-SAT.
                pass

    # Assignment vector: 0 for False, 1 for True
    # Initialize randomly
    rng = np.random.default_rng(42)
    assignment = rng.integers(0, 2, size=n)
    
    # Helper to compute unsatisfied count
    # A literal (var, sign) is satisfied if assignment[var] == sign (where sign is 1 for True, 0 for False? No)
    # Let's use: assignment is 0/1. 
    # Positive literal (sign=1) is true if assignment[var] == 1.
    # Negative literal (sign=-1) is true if assignment[var] == 0.
    # So, literal is true iff (assignment[var] == 1) * sign + (assignment[var] == 0) * (-sign) ... 
    # Simpler: 
    # val = assignment[var]
    # if sign == 1: true if val == 1
    # if sign == -1: true if val == 0
    # This is equivalent to: (val * sign == 1) ? 
    # If sign=1, val=1 -> 1*1=1 (True). val=0 -> 0*1=0 (False).
    # If sign=-1, val=0 -> 0*-1=0 (False? No, we want True). 
    # Let's map assignment to +/- 1.
    # Let a = 2*assignment - 1. So a is 1 (True) or -1 (False).
    # Literal (var, sign) is true if a[var] == sign.
    
    a = 2 * assignment - 1  # Shape (n,), values in {-1, 1}
    
    def get_unsat_count(a_vec):
        # a_vec shape (n,)
        # clause_vars shape (m, 3), clause_signs shape (m, 3)
        # Gather assignments for each literal
        # np.take_along_axis requires indices to match shape.
        # We can use fancy indexing: a_vec[clause_vars] gives shape (m, 3)
        vals = a_vec[clause_vars]  # Shape (m, 3)
        # Check if vals == clause_signs
        # A clause is satisfied if ANY literal is true.
        # lit_true = (vals == clause_signs) # Shape (m, 3)
        # clause_sat = np.any(lit_true, axis=1) # Shape (m,)
        # unsat = np.sum(~clause_sat)
        
        # Vectorized:
        lit_true = (vals == clause_signs)
        clause_sat = np.any(lit_true, axis=1)
        return np.sum(~clause_sat)

    best_score = get_unsat_count(a)
    best_assignment = assignment.copy()
    
    if best_score == 0:
        return assignment.tolist()

    # Local Search: WalkSAT
    # Parameters
    noise = 0.5  # Probability of flipping a random var in unsat clause vs best var
    restarts = 0
    max_restarts = 1000
    steps_per_restart = 5000
    tabu_len = 100
    tabu = [0] * n
    tabu_counter = 0
    
    current_score = best_score
    current_a = a
    current_assignment = assignment
    
    # Precompute clause literal indices for faster access? 
    # clause_vars is already int32.
    
    while time.time() < deadline:
        if restarts >= max_restarts:
            break
            
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Check if solved
            if current_score == 0:
                return current_assignment.tolist()
            
            # Pick a random unsatisfied clause
            # We need to know which clauses are unsatisfied.
            # Recomputing unsat clauses every step is expensive.
            # Optimization: Maintain a list of unsat clauses? 
            # Or just pick a random clause, check if unsat. If sat, pick another.
            # Given m can be large, this might be slow if most are sat.
            # But in early stages, many are unsat.
            
            # Efficient way: Compute unsat mask.
            vals = current_a[clause_vars]
            lit_true = (vals == clause_signs)
            clause_sat = np.any(lit_true, axis=1)
            unsat_mask = ~clause_sat
            unsat_indices = np.where(unsat_mask)[0]
            
            if len(unsat_indices) == 0:
                return current_assignment.tolist()
            
            # Pick random unsat clause
            idx = unsat_indices[rng.integers(0, len(unsat_indices))]
            
            # Get literals in this clause
            c_vars = clause_vars[idx]
            c_signs = clause_signs[idx]
            
            # Find the literal that, if flipped, would satisfy the clause?
            # No, we flip a variable. 
            # We want to flip a variable that appears in this clause.
            # Let's evaluate the gain of flipping each variable in the clause.
            # Gain = (new_score - old_score)
            # We want to minimize score, so we want negative gain (improvement).
            
            # For each var v in c_vars:
            # If we flip v, a[v] becomes -a[v].
            # This affects all clauses containing v.
            # Instead of calculating full delta, WalkSAT heuristic:
            # 1. Count how many other unsat clauses contain v (and would be satisfied by flipping v? No, flipping v changes truth value of v in ALL clauses).
            # Standard WalkSAT:
            # - Compute for each var in the unsat clause:
            #   - num_unsat_would_be_satisfied: How many other unsat clauses contain the *negation* of the current value? 
            #     Actually, simpler: 
            #     Let's just compute the new score for each candidate variable flip.
            #     Since clause size is 3, there are at most 3 candidates.
            
            candidates = []
            for k in range(3):
                v = c_vars[k]
                if v == -1: continue # dummy
                # Simulate flip
                old_val = current_a[v]
                new_val = -old_val
                
                # Calculate delta
                # This is expensive if done naively.
                # But we only have 3 candidates.
                # We can check the effect on all clauses? No, too slow.
                # We can check the effect on clauses containing v.
                # We haven't precomputed var -> clauses.
                
                # Alternative: Just flip and measure? 
                # If we flip v:
                # 1. The chosen clause (idx) becomes satisfied (because we picked an unsat clause, and flipping one of its literals makes it true? 
                #    Wait. If clause is unsat, all literals are false. 
                #    If we flip a variable in the clause, that literal becomes true. So the clause becomes satisfied. 
                #    So score decreases by at least 1.
                # 2. Other clauses containing v might change status.
                
                # To be efficient, let's just pick the variable in the clause that has the highest "activity" or randomly.
                # Simple WalkSAT:
                # - With prob noise, pick random var from clause.
                # - With prob 1-noise, pick var that minimizes the number of other unsat clauses it would break?
                #   Or maximizes the number of other unsat clauses it would satisfy?
                
                # Let's use a simple heuristic:
                # Count how many other unsat clauses contain this variable.
                # If we flip v, the current clause becomes SAT.
                # Other clauses containing v:
                #   - If they were UNSAT, and v was the only unassigned/false literal? No, we don't know.
                #   - Let's just estimate:
                #     Gain = -1 (current clause sat) + (changes in other clauses)
                #     We want to minimize changes in other clauses.
                #     Ideally, we flip a var that appears in FEWER other unsat clauses.
                
                # Precompute var frequency in unsat clauses?
                # Let's just do a random pick among the 3, or the one with lowest degree in unsat clauses.
                
                # For simplicity and speed in this vectorized context, let's just pick a random variable from the unsat clause.
                # This is pure Random Walk. It might be slow.
                # Better: Pick the variable that appears in the fewest other unsat clauses.
                
                # We can compute the count of unsat clauses containing each of the 3 vars.
                # unsat_mask is available.
                # For var v, count = sum(unsat_mask & (clause_vars == v))
                # This is O(m) for each of 3 vars. Total O(3m) per step. 
                # With m=1000, that's 3000 ops. 5000 steps * 1000 restarts = 5M steps * 3000 = 15B ops. Too slow in Python loop.
                
                # We need to be smarter.
                # Let's just use Random WalkSAT first. If it's too slow, we'll optimize.
                # Actually, 5000 steps * 1000 restarts is 5M steps. 
                # If each step takes 1ms, that's 5000s. Too slow.
                # We need vectorized step or fewer steps.
                
                # Let's reduce steps_per_restart and increase restarts?
                # Or, use a better heuristic that doesn't require O(m) scan.
                
                # Heuristic: Flip the variable in the clause that has the highest global activity (VSIDS-like).
                # We can maintain an activity array.
                
                break # Break inner loop to pick var
            
            # Pick a variable to flip
            # Let's just pick a random one from the 3 for now to ensure it runs.
            # We can improve this if needed.
            k = rng.integers(0, 3)
            v = c_vars[k]
            
            if v == -1:
                # Dummy var, pick another
                k = (k + 1) % 3
                v = c_vars[k]
            
            # Flip
            current_a[v] = -current_a[v]
            current_assignment[v] = 1 - current_assignment[v]
            
            # Update score
            # We can recompute score. O(m).
            # Or update incrementally? 
            # Recomputing is safer.
            new_score = get_unsat_count(current_a)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_assignment = current_assignment.copy()
            else:
                # Accept worse moves to escape local minima? 
                # WalkSAT accepts if it improves OR randomly.
                # Let's accept if it improves. If not, we are stuck?
                # Standard WalkSAT: 
                # 1. If improves, accept.
                # 2. Else, with prob noise, accept random.
                # Here we already picked random. 
                # Let's just accept it if it's not too bad? 
                # Or restart if we hit a local minimum.
                
                # Let's check if we are stuck.
                # If we didn't improve, we might be in a local min.
                # We'll just continue. The random nature will help.
                pass

            # Update tabu?
            # tabu[v] = tabu_counter + tabu_len
            # tabu_counter += 1
            # If we use tabu, we should avoid flipping vars in tabu.
            # For now, simple random walk.

        # End of restart
        restarts += 1
        
        # Restart: Randomize assignment
        current_assignment = rng.integers(0, 2, size=n)
        current_a = 2 * current_assignment - 1
        current_score = get_unsat_count(current_a)
        
        if current_score < best_score:
            best_score = current_score
            best_assignment = current_assignment.copy()

    # Return best found
    return best_assignment.tolist()
