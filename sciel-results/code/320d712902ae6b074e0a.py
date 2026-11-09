"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_numpy(U, V, W, n, m, p):
    """Verify exact decomposition using numpy for speed."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution as baseline
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Strategy: Tensor slice / exterior algebra inspired decomposition
    # Key insight: 2x3 x 3x8 can be viewed as block structure
    # A is 2x3, B is 3x8. Write B as [B1 B2] where B1, B2 are 3x4
    # Then C = A*B = [A*B1, A*B2]
    # Each A*B1 is 2x4, which can be done in rank 8 (schoolbook) or less
    # But we can do better by sharing factors
    
    # Let's try a structured approach:
    # Partition the 3x8 matrix into blocks of 3x2: B = [B0, B1, B2, B3]
    # Then C = [A*B0, A*B1, A*B2, A*B3]
    # Each A*Bk is 2x2, which has rank 7 (Strassen)
    # But we can share the A factors across all blocks
    
    # Actually, let's think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, this is a linear combination of rows of B
    # 
    # Let's try a different approach: use the fact that we have 2 rows in A
    # and 3 columns in A (which are the 3 rows of B)
    
    # Tensor decomposition approach:
    # The tensor T_{i,j,k} = delta_{i,j} * delta_{j,k} is not quite right
    # Actually T is defined by: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # But in tensor decomposition form, we want:
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to find a low-rank decomposition using structured patterns
    
    # For 2x3 x 3x8, let's try ranks from 24 to 47
    # (lower bound is max(6, 24, 16) = 24)
    
    vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.0])
    
    # Try structured initializations based on Kronecker products and block structure
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        
        # Try multiple random initializations with structure
        for trial in range(500):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.RandomState(trial * 7919 + target_rank * 137 + 42)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Sparse random initialization with structure
            density = 0.2 + 0.4 * (target_rank / best_rank)
            for idx in range(n*m):
                mask = rng.random(target_rank) < density
                U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(m*p):
                mask = rng.random(target_rank) < density
                V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(p*n):
                mask = rng.random(target_rank) < density
                W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            if _verify_numpy(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break
    
    # If we found a better rank, try to refine it
    if best_rank < rank_school:
        # Try to reduce rank further with more trials
        for target_rank in range(lower_bound, best_rank):
            if time.time() > deadline - 5.0:
                break
            
            for trial in range(200):
                if time.time() > deadline - 2.0:
                    break
                
                rng = np.random.RandomState(trial * 31337 + target_rank * 271 + 99)
                
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                density = 0.3
                for idx in range(n*m):
                    mask = rng.random(target_rank) < density
                    U[idx, mask] = rng.choice(vals, size=np.sum(mask))
                for idx in range(m*p):
                    mask = rng.random(target_rank) < density
                    V[idx, mask] = rng.choice(vals, size=np.sum(mask))
                for idx in range(p*n):
                    mask = rng.random(target_rank) < density
                    W[idx, mask] = rng.choice(vals, size=np.sum(mask))
                
                if _verify_numpy(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
