"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    Condition: for all i, k, j1, j2:
    sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1, j2}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = np.dot(U[i*m+j1, :] * V[j2*p+k, :], W[k*n+i, :])
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try structured decompositions
    # Key insight: p=9=3*3, so we can exploit the block structure
    # For each k = 3*k1 + k2, we have a 2x3x3 subproblem
    
    # Try rank 27 first (theoretical lower bound)
    # Structure: use 3 groups of 9 columns, one for each k2 in [0,3)
    # Within each group, solve the 2x3x3 problem
    
    # For 2x3x3: rank lower bound = max(2*3, 3*3, 3*2) = max(6,9,6) = 9
    # Schoolbook for 2x3x3 is 18, but we can try to do better
    
    # Let's try a systematic search for small ranks using structured patterns
    
    np.random.seed(42)
    
    # Try various structured approaches
    # Approach 1: Block diagonal with 3 copies of a 2x3x3 decomposition
    # If we find rank-r for 2x3x3, we get rank 3r for 2x3x9
    
    # For 2x3x3, let's try rank 9 (the lower bound)
    # The condition for 2x3x3: for all i in [0,2), k in [0,3), j1,j2 in [0,3):
    # sum_c U[i*3+j1,c] * V[j2*3+k,c] * W[k*2+i,c] = delta_{j1,j2}
    
    # Try to find rank-9 decomposition for 2x3x3
    found_233 = False
    U_233 = None
    V_233 = None
    W_233 = None
    
    for target_r in range(9, 19):
        if time.time() > deadline - 30:
            break
        for trial in range(50):
            if time.time() > deadline - 25:
                break
            # Random sparse initialization
            U = np.zeros((6, target_r))
            V = np.zeros((9, target_r))
            W = np.zeros((6, target_r))
            
            for c in range(target_r):
                i = np.random.randint(0, 2)
                j = np.random.randint(0, 3)
                k = np.random.randint(0, 3)
                U[i*3+j, c] = 1.0
                V[j*3+k, c] = 1.0
                W[k*2+i, c] = 1.0
            
            if _verify_decomposition(U, V, W, 2, 3, 3, tol=1e-9):
                U_233, V_233, W_233 = U, V, W
                found_233 = True
                break
        if found_233:
            break
    
    if found_233 and U_233 is not None:
        r_233 = U_233.shape[1]
        total_rank = 3 * r_233
        
        if total_rank < best_rank:
            # Build 2x3x9 decomposition from 2x3x3
            U_full = np.zeros((6, total_rank))
            V_full = np.zeros((27, total_rank))
            W_full = np.zeros((18, total_rank))
            
            for k2 in range(3):
                offset = k2 * r_233
                # U is the same for all k2 blocks
                U_full[:, offset:offset+r_233] = U_233
                
                # V: for k = 3*k1 + k2, V[j*9+k, c] = V_233[j*3+k1, c-offset]
                for j in range(3):
                    for k1 in range(3):
                        k = 3*k1 + k2
                        V_full[j*9+k, offset:offset+r_233] = V_233[j*3+k1, :]
                
                # W: for k = 3*k1 + k2, W[k*2+i, c] = W_233[k1*2+i, c-offset]
                for k1 in range(3):
                    for i in range(2):
                        k = 3*k1 + k2
                        W_full[k*2+i, offset:offset+r_233] = W_233[k1*2+i, :]
            
            if _verify_decomposition(U_full, V_full, W_full, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = total_rank
    
    # Try direct structured search for lower ranks
    # Use a more systematic approach with half-integer values
    
    for target_rank in range(max(27, best_rank - 10), min(best_rank, 54)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            
            # Structured initialization with some randomness
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a pattern based on the indices
            for c in range(target_rank):
                # Distribute columns across the tensor structure
                # Use a hash-like function to assign positions
                h = (c * 7 + trial * 13) % (n * m * p)
                i = h // (m * p)
                j = (h // p) % m
                k = h % p
                
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # Convert to lists for the return interface
    U_out = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_out = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_out = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
