"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    # First, let's create the schoolbook decomposition as a fallback
    def make_schoolbook():
        rank = nmp
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Fast verify using numpy - vectorized over all (i,j,k) triples
    def verify(U, V, W):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        rank = U_np.shape[1]
        
        # For each rank component c, compute the outer product pattern
        # and check that the sum gives exactly 1 for each (i,j,k)
        
        # Build the target tensor: T[i,j,k] = 1 for all i,j,k
        # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Vectorized approach: for each c, form a 3D array and accumulate
        # This is O(rank * n*m*p) which is fine
        
        result = np.zeros((n, m, p), dtype=np.int64)
        for c in range(rank):
            uc = U_np[:, c]  # length n*m
            vc = V_np[:, c]  # length m*p
            wc = W_np[:, c]  # length p*n
            
            # Reshape and compute outer product contribution
            u3d = uc.reshape(n, m)
            v3d = vc.reshape(m, p)
            w3d = wc.reshape(p, n)
            
            # For each (i,j,k): u3d[i,j] * v3d[j,k] * w3d[k,i]
            # We can do this efficiently
            for i in range(n):
                for j in range(m):
                    uval = u3d[i, j]
                    if uval == 0:
                        continue
                    for k in range(p):
                        result[i, j, k] += uval * v3d[j, k] * w3d[k, i]
        
        # Check all entries are 1
        return np.all(result == 1)
    
    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Strategy: Start from schoolbook and try to merge terms
    # The key insight: if two rank-1 terms can be combined into one,
    # we reduce the rank by 1.
    
    # Alternative: try to find a decomposition where some U,V,W entries
    # are shared across multiple rank components
    
    # Let's try a different approach: use the structure of matrix multiplication
    # to propose better initial guesses
    
    # For M_{n,m,p}, the tensor has a specific structure.
    # Let's try to find a low-rank decomposition using a smarter search.
    
    # Key idea: try ranks from 83 down, and for each rank, try to construct
    # a valid decomposition by starting from schoolbook and modifying it.
    
    def try_modify_schoolbook(target_rank, max_modifications=5):
        """Try to reduce schoolbook rank by modifying some entries."""
        if target_rank >= nmp:
            return None
        
        U = np.array(make_schoolbook()[0], dtype=np.int64)
        V = np.array(make_schoolbook()[1], dtype=np.int64)
        W = np.array(make_schoolbook()[2], dtype=np.int64)
        
        # We need to eliminate (nmp - target_rank) columns
        # by redistributing their contributions to other columns
        
        num_to_eliminate = nmp - target_rank
        if num_to_eliminate > 10:
            return None  # Too many to handle efficiently
        
        # Try to merge random pairs of columns
        columns_to_try = list(range(nmp))
        random.shuffle(columns_to_try)
        
        for elim_idx in range(num_to_eliminate):
            # Pick a column to eliminate
            col_to_elim = columns_to_try[elim_idx]
            
            # Try to distribute its contribution to other columns
            # The contribution of column c is: U[:,c] * V[:,c] * W[:,c]
            # We need to add this to other columns' contributions
            
            # Simple approach: add the column's pattern to a random other column
            # This won't work in general, but let's try
            
            # Actually, let's try a different approach:
            # For each (i,j,k), the contribution from column c is
            # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We need this to be preserved when we remove column c
            
            # This is complex. Let's try a simpler heuristic:
            # Just zero out some columns and see if we can fix the rest.
            
            pass
        
        return None
    
    # Better approach: use a local search / hill climbing
    # Start from schoolbook, try random perturbations, keep if valid
    
    # Actually, the most promising approach for finding ANY improvement:
    # Try to find a rank-83 decomposition by checking if any single
    # schoolbook term can be removed and compensated by modifying others.
    
    # Let's try: for each column c in schoolbook, see if we can set
    # U[:,c]=0, V[:,c]=0, W[:,c]=0 and adjust other columns to compensate.
    
    # This is equivalent to asking if the tensor has a "redundant" term.
    
    # For matrix multiplication, the schoolbook decomposition is known to be
    # minimal for small cases, but for 3x4x7, there might be improvements.
    
    # Let's try a more direct approach: random search with better initialization
    
    # Use the fact that entries should be small integers (likely -1, 0, 1)
    
    def random_search(target_rank, num_attempts=100):
        """Random search for a valid decomposition at given rank."""
        for attempt in range(num_attempts):
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            W = np.zeros((p * n, target_rank), dtype=np.int64)
            
            # Initialize with random sparse patterns
            for c in range(target_rank):
                # Each column should have a few non-zero entries
                num_nonzero_U = random.randint(2, 5)
                num_nonzero_V = random.randint(2, 5)
                num_nonzero_W = random.randint(2, 5)
                
                U_idx = random.sample(range(n * m), num_nonzero_U)
                V_idx = random.sample(range(m * p), num_nonzero_V)
                W_idx = random.sample(range(p * n), num_nonzero_W)
                
                for idx in U_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in V_idx:
                    V[idx, c] = random.choice([-1, 1])
                for idx in W_idx:
                    W[idx, c] = random.choice([-1, 1])
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return U, V, W
        
        return None
    
    # Try ranks from 83 down to 70
    for target_rank in range(83, 69, -1):
        if time.time() > deadline - 30:
            break
        
        result = random_search(target_rank, num_attempts=50)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            break
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
