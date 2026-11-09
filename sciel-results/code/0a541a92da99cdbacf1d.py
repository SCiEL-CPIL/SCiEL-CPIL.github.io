"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s budget, leave margin for 120s limit
    
    n, m, p = DIMENSIONS
    # U: shape (n*m, rank) = (8, rank)
    # V: shape (m*p, rank) = (24, rank)
    # W: shape (p*n, rank) = (12, rank)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if all entries match."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using numpy."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try structured decompositions with lower rank
    # Key insight: For matrix multiplication M_{n,m,p}, we can exploit
    # the rectangular structure to find decompositions below n*m*p.
    
    # Strategy 1: Use the fact that for fixed (i,j), the k-dependence
    # is just a scalar 1, so we can potentially share factors across k.
    
    # Let's try rank = n*m*p - some_reduction
    # Known lower bound: max(nm, mp, pn) = max(8, 24, 12) = 24
    
    # Try a randomized search for rank < 48
    def try_random_rank(target_rank, max_attempts=100):
        """Try to find a decomposition of given rank via random initialization."""
        import numpy as np
        np.random.seed(42)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small values
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Convert to list of lists for verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Strategy 2: Structured construction using block decomposition
    # For M_{2,4,6}, try to use the structure where we group k values
    
    def try_structured():
        """Try structured constructions with various groupings."""
        
        # Construction A: Group k into pairs
        # For each (i, j, k_group), create a factor
        # This gives rank = n * m * (p/2) = 2 * 4 * 3 = 24
        
        # But we need to be careful about the W factor
        # Let's try: for k in {0,1}, {2,3}, {4,5}
        groups = [(0, 1), (2, 3), (4, 5)]
        rank = n * m * len(groups)  # 24
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for g_idx, (k1, k2) in enumerate(groups):
                    # Factor for (i, j, group)
                    U[i*m+j][col] = 1
                    V[j*p+k1][col] = 1
                    V[j*p+k2][col] = 1
                    W[k1*n+i][col] = 1
                    W[k2*n+i][col] = 1
                    col += 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        # Construction B: Use half-integer coefficients
        # Try to find a decomposition where some entries are 1/2
        # This might allow lower rank
        
        # Let's try rank = 36 (48 - 12)
        # Idea: For each i, j, instead of 6 separate k-factors,
        # use 3 factors that each cover 2 k-values with appropriate weights
        
        for num_k_groups in [2, 3, 4]:
            if time.time() > deadline - 10:
                break
            
            # Create k groups
            k_groups = []
            k_per_group = p // num_k_groups
            remainder = p % num_k_groups
            k_start = 0
            for g in range(num_k_groups):
                sz = k_per_group + (1 if g < remainder else 0)
                k_groups.append(list(range(k_start, k_start + sz)))
                k_start += sz
            
            rank = n * m * num_k_groups
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for g in range(num_k_groups):
                        U[i*m+j][col] = 1
                        for k in k_groups[g]:
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                        col += 1
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Strategy 3: Use known results about matrix multiplication rank
    # For M_{2,4,6}, the rank is at most 2*4*6 = 48 (schoolbook)
    # But we can potentially do better using the rectangular structure
    
    # Let's try a more sophisticated approach:
    # For each i (2 values), the subproblem is M_{1,4,6} which has rank 4*6=24
    # But we can potentially share factors between i=0 and i=1
    
    # Actually, let's just try to find any rank < 48 via systematic search
    
    # First, try the structured approach
    result = try_structured()
    if result is not None:
        return result
    
    # Try random search for rank 40
    if time.time() < deadline - 30:
        result = try_random_rank(40, max_attempts=50)
        if result is not None:
            return result
    
    # Try random search for rank 36
    if time.time() < deadline - 20:
        result = try_random_rank(36, max_attempts=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
