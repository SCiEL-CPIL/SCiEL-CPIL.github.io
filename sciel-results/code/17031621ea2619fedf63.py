import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    n, m, p = 2, 3, 3
    schoolbook_rank = n * m * p  # 18
    
    # Build the tensor T[i, j, k] for matrix multiplication
    # T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # i.e., sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    # Try to find lower rank solutions
    # Strategy: Use structured vectors from 2x2x2 rank-7 building blocks
    
    # Known rank-7 decomposition for 2x2x2 (Strassen-like with 7 terms)
    # For multiplying 2x2 matrices, rank 7 exists (Coppersmith-Winograd / known results)
    # Actually, the standard Strassen is rank 7 for 2x2x2.
    
    # Let's think differently. Try random search with integer/half-integer entries.
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try rank 17 and below using random structured search
    # The constraint: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Use a smarter approach: fix structure based on symmetries
    # Try small ranks with careful vector design
    
    # Approach: For each rank R from 17 down to 14, try to find a solution
    # using vectors with entries in {0, 1/2, 1}
    
    def check_solution(U, V, W):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try a parameterized family based on the 2x2x2 rank-7 structure
    # For the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}), use rank 7.
    # The remaining indices (j=2 or k=2) need additional rank.
    
    # Let's try a direct search for rank 16 with specific structure.
    # The key insight: we can try to share terms between different (i,j,k) positions.
    
    # Random search with restarts
    np.random.seed(42)
    
    def random_search(rank, num_trials=5000):
        """Random search for a decomposition of given rank."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random vectors with entries in {0, 1/2, 1}
            U = np.random.choice([0, 0.5, 1], size=(n * m, rank))
            V = np.random.choice([0, 0.5, 1], size=(m * p, rank))
            W = np.random.choice([0, 0.5, 1], size=(p * n, rank))
            
            if check_solution(U, V, W):
                return U, V, W
        return None
    
    # Try structured search: use the fact that we need sum of products = 1
    # Try ranks 16, 15, 14 with more targeted approaches
    
    # More structured: try to build from known patterns
    # For 2x3x3, try rank 16
    
    # Alternative: use the fact that matrix multiplication tensor has specific structure
    # Let's try a greedy/iterative approach
    
    # Actually, let's try a more systematic approach with small rank
    # We know the minimum possible is at least max(n*m, m*p, p*n) = max(6,9,6) = 9
    # and at most 18. Known lower bound is higher.
    
    # Try rank 16 with a structured random search
    for target_rank in [17, 16, 15, 14]:
        if time.time() > deadline - 5:
            break
        
        # Structured random search
        found = random_search(target_rank, num_trials=20000)
        if found is not None:
            U, V, W = found
            if check_solution(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # If we found something better, use it; otherwise fall back to schoolbook
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
