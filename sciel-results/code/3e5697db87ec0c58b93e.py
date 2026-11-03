"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def validate_decomp(U, V, W):
        """Validate that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check all (i,j,k) tuples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition for 2x3x2 sub-problem
    # Then tile it 3 times for 2x3x6
    def find_2x3x2_decomp(max_time=60):
        """Find a valid decomposition for 2x3x2 with minimal rank."""
        n_sub, m_sub, p_sub = 2, 3, 2
        lower_bound = max(n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)  # max(6,6,4)=6
        
        # Try ranks from lower_bound upward
        for target_rank in range(lower_bound, 13):  # Up to schoolbook (12)
            if time.time() > start_time + max_time:
                return None
            
            # Systematic search: try structured patterns
            # Each rank-1 term contributes to a subset of (i,j,k) entries
            # For 2x3x2, there are 12 (i,j,k) tuples
            
            # Strategy: try to find a pattern where each term covers 
            # a "line" or "plane" in the tensor
            found = _search_2x3x2(target_rank, n_sub, m_sub, p_sub)
            if found is not None:
                return found
        
        return None
    
    def _search_2x3x2(target_rank, n, m, p):
        """Search for a valid 2x3x2 decomposition with given rank."""
        # For 2x3x2: U is 6xR, V is 6xR, W is 4xR
        # T[i,j,k] = sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        
        # Try a structured approach: use the fact that for small tensors,
        # we can try specific patterns
        
        # Pattern 1: Try to use rank-6 with a specific structure
        # Based on the idea that we can decompose using "outer product" patterns
        
        if target_rank >= 6:
            # Try a deterministic construction
            # For 2x3x2, try to find U,V,W with specific sparsity patterns
            
            # Approach: use the fact that the tensor is all-ones
            # Try to find a factorization where each term has a specific support
            
            # Let's try a more direct approach: solve the system
            # We need: for all i,j,k: sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r] = 1
            
            # This is a system of 12 equations in R*6*2 + R*6*2 + R*4*2 = 16R unknowns
            # But it's nonlinear (trilinear)
            
            # Try random initialization with small values
            np.random.seed(42 + target_rank)
            for trial in range(10000):
                if time.time() > start_time + 140:
                    return None
                
                # Initialize with small random values
                U = np.random.randint(-2, 3, size=(6, target_rank)) / 2.0
                V = np.random.randint(-2, 3, size=(6, target_rank)) / 2.0
                W = np.random.randint(-2, 3, size=(4, target_rank)) / 2.0
                
                # Check validity
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(target_rank):
                                total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            if abs(total - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to find a 2x3x2 decomposition
    result_2x3x2 = find_2x3x2_decomp()
    
    if result_2x3x2 is not None:
        U_sub, V_sub, W_sub = result_2x3x2
        rank_sub = len(U_sub[0])
        
        # Tile 3 times for 2x3x6
        full_rank = rank_sub * 3
        U_full = [[0] * full_rank for _ in range(n * m)]
        V_full = [[0] * full_rank for _ in range(m * p)]
        W_full = [[0] * full_rank for _ in range(p * n)]
        
        for rep in range(3):
            offset = rep * rank_sub
            # U: same for all reps
            for i in range(n):
                for j in range(m):
                    U_full[i*m + j][offset:offset+rank_sub] = U_sub[i*m + j]
            
            # V: map k in [rep*2, rep*2+2) to k' in [0,2)
            for j in range(m):
                for k_prime in range(2):
                    k = rep * 2 + k_prime
                    V_full[j*p + k][offset:offset+rank_sub] = V_sub[j*2 + k_prime]
            
            # W: map k in [rep*2, rep*2+2) to k' in [0,2)
            for k_prime in range(2):
                k = rep * 2 + k_prime
                for i in range(n):
                    W_full[k*n + i][offset:offset+rank_sub] = W_sub[k_prime*n + i]
        
        decomp = {"U": U_full, "V": V_full, "W": W_full}
        if validate_decomp(U_full, V_full, W_full):
            return decomp
    
    # Try a direct search for 2x3x6 with rank < 36
    # Use a more targeted approach: try rank 18-35
    for target_rank in range(18, 36):
        if time.time() > deadline:
            break
        
        # Try a few random attempts
        np.random.seed(1000 + target_rank)
        for trial in range(100):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-2, 3, size=(6, target_rank)) / 2.0
            V = np.random.randint(-2, 3, size=(18, target_rank)) / 2.0
            W = np.random.randint(-2, 3, size=(12, target_rank)) / 2.0
            
            # Quick check on a few entries
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for r in range(target_rank):
                            total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
