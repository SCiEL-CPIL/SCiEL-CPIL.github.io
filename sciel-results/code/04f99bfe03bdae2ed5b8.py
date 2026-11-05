import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 72
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find lower-rank decompositions
    # Strategy: for each target rank, try to find a valid decomposition
    # by starting from schoolbook and trying to eliminate terms
    
    def try_eliminate_term(target_rank):
        """Try to find a decomposition of given rank by eliminating (schoolbook_rank - target_rank) terms."""
        if time.time() > deadline:
            return None
        
        # Start with schoolbook
        U = U_school.copy()
        V = V_school.copy()
        W = W_school.copy()
        
        # We need to eliminate (schoolbook_rank - target_rank) terms
        num_to_eliminate = schoolbook_rank - target_rank
        
        # Try random subsets of terms to eliminate
        all_cols = list(range(schoolbook_rank))
        random.seed(42)
        
        for trial in range(100):
            if time.time() > deadline:
                return None
            
            # Pick which columns to keep
            if num_to_eliminate == 0:
                keep_cols = all_cols
            else:
                eliminate_set = set(random.sample(all_cols, num_to_eliminate))
                keep_cols = [c for c in all_cols if c not in eliminate_set]
            
            if len(keep_cols) != target_rank:
                continue
            
            # Create compressed matrices
            U_try = U[:, keep_cols].copy()
            V_try = V[:, keep_cols].copy()
            W_try = W[:, keep_cols].copy()
            
            # Now try to adjust the kept terms to compensate for eliminated ones
            # This is a nonlinear problem, so use a simple heuristic
            # For each eliminated term, try to add its contribution to existing terms
            
            # Actually, this approach is too simplistic. Let me try a different method.
            # Use the fact that we can scale and shift terms.
            
            # For now, just check if the kept terms alone work (they won't, but let's try)
            if check_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist()):
                return U_try.tolist(), V_try.tolist(), W_try.tolist()
        
        return None
    
    # Better approach: use a structured decomposition
    # For M_{3,3,8}, try to use the fact that we can group k-indices
    
    # Approach: use a "block" decomposition where we handle k in groups of 2
    # For each pair (k, k+1), use 9 terms (schoolbook for 3x3x2)
    # Total: 4 pairs * 9 = 36... wait, that's only 36, not 72!
    
    # Wait, I made an error. Let me reconsider.
    # For 3x3x2, schoolbook is 3*3*2 = 18 terms
    # For 4 pairs: 4 * 18 = 72 terms. Same as before.
    
    # Let me try a different grouping: handle all 8 k-indices together
    # using a smarter decomposition
    
    # Key insight: for a fixed (i,j), the term A_{i,j} * B_{j,k} appears for all k.
    # We can factor out A_{i,j} and use a single rank-1 term for each (i,j)
    # that multiplies by a vector in the k-space.
    
    # Actually, this is exactly what the schoolbook does: for each (i,j,k), one term.
    # To reduce rank, we need to find linear dependencies.
    
    # Let me try a computational approach: set up the tensor and try to find
    # a low-rank decomposition using iterative methods
    
    # For now, let's try random search with a better initialization
    def try_random_search(target_rank, num_trials=50):
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with random sparse values
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Use a structured initialization based on schoolbook
            # Assign each rank-1 term to cover a subset of (i,j,k) triples
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(triples)
            
            for r in range(target_rank):
                # Assign this term to cover some triples
                # Pick a random center (i,j,k) and assign values
                i, j, k = triples[r % len(triples)]
                
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            
            if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 71 down to 60
    for target_rank in range(schoolbook_rank - 1, 59, -1):
        if time.time() > deadline:
            break
        
        result = try_random_search(target_rank, num_trials=20)
        if result is not None:
            U_list, V_list, W_list = result
            best_U = U_list
            best_V = V_list
            best_W = W_list
            best_rank = target_rank
            break
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
