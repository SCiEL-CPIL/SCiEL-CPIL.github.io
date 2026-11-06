import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280  # Safety margin

    m = len(clauses)
    
    # Precompute matrices
    # Pos[i, j] is 1 if literal +x_{j+1} is in clause i
    # Neg[i, j] is 1 if literal -x_{j+1} is in clause i
    Pos = np.zeros((m, n), dtype=np.int8)
    Neg = np.zeros((m, n), dtype=np.int8)
    
    # Also store clause literals for easy access
    clause_lits = [list(c) for c in clauses]
    # Store variable indices for each clause for faster access
    clause_vars = [[abs(lit)-1 for lit in c] for c in clauses]

    # Precompute for each variable: which clauses contain it and as what sign
    # var_pos_clauses[v] = list of clause indices where +x_v appears
    # var_neg_clauses[v] = list of clause indices where -x_v appears
    var_pos_clauses = [[] for _ in range(n)]
    var_neg_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0:
                Pos[i, var_idx] = 1
                var_pos_clauses[var_idx].append(i)
            else:
                Neg[i, var_idx] = 1
                var_neg_clauses[var_idx].append(i)
                
    # Convert to numpy arrays for vectorized operations
    var_pos_clauses_arr = [np.array(l, dtype=np.int32) for l in var_pos_clauses]
    var_neg_clauses_arr = [np.array(l, dtype=np.int32) for l in var_neg_clauses]

    def count_violated(A):
        """A is a numpy boolean array of size n."""
        A_int = A.astype(np.int8)
        SatCount = np.sum(Pos * A_int, axis=1) + np.sum(Neg * (1 - A_int), axis=1)
        return np.sum(SatCount == 0)

    def solve_walksat(A, max_steps):
        """
        A: numpy boolean array of size n
        max_steps: int
        Returns True if solved, False otherwise.
        """
        A_int = A.astype(np.int8)
        # Initial SatCount
        SatCount = np.sum(Pos * A_int, axis=1) + np.sum(Neg * (1 - A_int), axis=1)
        
        violated_indices = np.where(SatCount == 0)[0]
        
        if len(violated_indices) == 0:
            return True

        p_greedy = 0.5
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            
            if len(violated_indices) == 0:
                return True

            # Pick a random violated clause
            ci = rng.choice(violated_indices.tolist())
            clause_vars_in_ci = clause_vars[ci]
            
            # Candidates are the variables in this clause
            # Compute gain for each
            gains = []
            
            for var_idx in clause_vars_in_ci:
                # Compute delta efficiently using precomputed clause lists
                # Vectorized gain calculation
                
                if A_int[var_idx] == 0:
                    # Flipping to 1
                    # Satisfy: clauses in pos_arr where SatCount == 0
                    # Violate: clauses in neg_arr where SatCount == 1
                    pos_arr = var_pos_clauses_arr[var_idx]
                    neg_arr = var_neg_clauses_arr[var_idx]
                    
                    become_sat = np.sum(SatCount[pos_arr] == 0)
                    become_violated = np.sum(SatCount[neg_arr] == 1)
                else:
                    # Flipping to 0
                    # Satisfy: clauses in neg_arr where SatCount == 0
                    # Violate: clauses in pos_arr where SatCount == 1
                    pos_arr = var_pos_clauses_arr[var_idx]
                    neg_arr = var_neg_clauses_arr[var_idx]
                    
                    become_sat = np.sum(SatCount[neg_arr] == 0)
                    become_violated = np.sum(SatCount[pos_arr] == 1)
                
                gain = become_sat - become_violated
                gains.append((gain, var_idx))
            
            # Selection
            if rng.random() < p_greedy:
                # Greedy
                gains.sort(key=lambda x: -x[0])
                best_gain, best_var = gains[0]
            else:
                # Random from clause
                best_var = rng.choice(clause_vars_in_ci)
            
            # Update A and A_int
            old_val = A_int[best_var]
            new_val = 1 - old_val
            A_int[best_var] = new_val
            A[best_var] = bool(new_val)
            
            # Update SatCount using vectorized operations
            pos_arr = var_pos_clauses_arr[best_var]
            neg_arr = var_neg_clauses_arr[best_var]
            
            if old_val == 0:
                # Flipping to 1
                # Clauses with +x_best_var: SatCount += 1
                # Clauses with -x_best_var: SatCount -= 1
                SatCount[pos_arr] += 1
                SatCount[neg_arr] -= 1
            else:
                # Flipping to 0
                # Clauses with -x_best_var: SatCount += 1
                # Clauses with +x_best_var: SatCount -= 1
                SatCount[neg_arr] += 1
                SatCount[pos_arr] -= 1
            
            # Update violated_indices
            violated_indices = np.where(SatCount == 0)[0]

        return False

    # Main loop with restarts
    best_assignment = None
    best_violated = m + 1

    # Allow many restarts since each step is fast
    max_restarts = 5000
    base_steps = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        A = np.random.rand(n) < 0.5
        
        # Try WalkSAT
        steps = base_steps + rng.randint(0, 10000)
        
        if solve_walksat(A, steps):
            return A.tolist()
            
        # Track best
        v = count_violated(A)
        if v < best_violated:
            best_violated = v
            best_assignment = A.copy()

    if best_assignment is not None and count_violated(best_assignment) == 0:
        return best_assignment.tolist()
        
    if best_assignment is not None:
        return best_assignment.tolist()
        
    return [rng.random() < 0.5 for _ in range(n)]
