"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T for 3x3x6: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    T_full = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1
    
    def verify_full_decomp(U, V, W, T):
        """Check if U, V, W form a valid decomposition of T."""
        rank = U.shape[1]
        # Efficient check: for each rank term, add outer product
        T_recon = np.zeros_like(T, dtype=int)
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # Only process non-zero entries for efficiency
            nz_u = np.nonzero(u_col)[0]
            nz_v = np.nonzero(v_col)[0]
            nz_w = np.nonzero(w_col)[0]
            for a in nz_u:
                ua = u_col[a]
                for b in nz_v:
                    ub = ua * v_col[b]
                    if ub != 0:
                        for c in nz_w:
                            T_recon[a, b, c] += ub * w_col[c]
        return np.array_equal(T_recon, T)
    
    # Strategy: Use tensor product structure T_{3,3,6} = T_{3,3,3} ⊗ T_{1,1,2}
    # T_{1,1,2} has rank 2: it's two disjoint 1x1x1 terms
    # So if we have a rank-R decomposition for 3x3x3, we get rank 2R for 3x3x6
    
    # First, find a good decomposition for 3x3x3
    n3, m3, p3 = 3, 3, 3
    nm3, mp3, pn3 = n3*m3, m3*p3, p3*n3
    
    T3 = np.zeros((nm3, mp3, pn3), dtype=int)
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                a = i * m3 + j
                b = j * p3 + k
                c = k * n3 + i
                T3[a, b, c] = 1
    
    def verify_3x3(U, V, W):
        """Check if U, V, W form a valid decomposition of T3."""
        rank = U.shape[1]
        T_recon = np.zeros_like(T3, dtype=int)
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            nz_u = np.nonzero(u_col)[0]
            nz_v = np.nonzero(v_col)[0]
            nz_w = np.nonzero(w_col)[0]
            for a in nz_u:
                ua = u_col[a]
                for b in nz_v:
                    ub = ua * v_col[b]
                    if ub != 0:
                        for c in nz_w:
                            T_recon[a, b, c] += ub * w_col[c]
        return np.array_equal(T_recon, T3)
    
    # Try to find a rank-23 decomposition for 3x3x3 using a structured search
    # Start with schoolbook (rank 27) and try to reduce
    
    best_rank_3 = 27
    best_U3 = None
    best_V3 = None
    best_W3 = None
    
    # Schoolbook for 3x3x3
    U_school3 = np.zeros((nm3, 27), dtype=int)
    V_school3 = np.zeros((mp3, 27), dtype=int)
    W_school3 = np.zeros((pn3, 27), dtype=int)
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U_school3[i * m3 + j, col] = 1
                V_school3[j * p3 + k, col] = 1
                W_school3[k * n3 + i, col] = 1
                col += 1
    
    if verify_3x3(U_school3, V_school3, W_school3):
        best_U3, best_V3, best_W3 = U_school3, V_school3, W_school3
        best_rank_3 = 27
    
    # Try to find lower rank decompositions using a local search
    # We'll try ranks from 23 up to 26
    for target_rank in range(23, 27):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
                
            # Initialize with random sparse matrices
            U_rand = np.zeros((nm3, target_rank), dtype=int)
            V_rand = np.zeros((mp3, target_rank), dtype=int)
            W_rand = np.zeros((pn3, target_rank), dtype=int)
            
            # Start from schoolbook and randomly merge/remove terms
            if target_rank < 27:
                # Copy first target_rank terms from schoolbook
                U_rand[:, :target_rank] = U_school3[:, :target_rank]
                V_rand[:, :target_rank] = V_school3[:, :target_rank]
                W_rand[:, :target_rank] = W_school3[:, :target_rank]
                
                # Try to fix by adjusting
                # This is a heuristic - we'll use a different approach
                pass
            
            # Alternative: use a structured initialization based on Strassen for 2x2x2
            # Strassen for 2x2x2 has rank 7
            # We can embed this and add terms for the third dimension
            
            # Let's use a different strategy: start from a known good decomposition
            # and perturb it
            if attempt == 0:
                # Use the schoolbook as starting point and try to merge
                U_curr = U_school3[:, :target_rank].copy()
                V_curr = V_school3[:, :target_rank].copy()
                W_curr = W_school3[:, :target_rank].copy()
                
                # Try to find if this partial decomposition is valid
                if verify_3x3(U_curr, V_curr, W_curr):
                    best_U3, best_V3, best_W3 = U_curr, V_curr, W_curr
                    best_rank_3 = target_rank
                    break
            else:
                # Random perturbation approach
                U_curr = U_school3[:, :target_rank].copy()
                V_curr = V_school3[:, :target_rank].copy()
                W_curr = W_school3[:, :target_rank].copy()
                
                # Randomly modify some entries
                for _ in range(20):
                    r = random.randint(0, target_rank - 1)
                    a = random.randint(0, nm3 - 1)
                    U_curr[a, r] = random.choice([-1, 0, 1, 1, 1])
                    
                    b = random.randint(0, mp3 - 1)
                    V_curr[b, r] = random.choice([-1, 0, 1, 1, 1])
                    
                    c = random.randint(0, pn3 - 1)
                    W_curr[c, r] = random.choice([-1, 0, 1, 1, 1])
                
                if verify_3x3(U_curr, V_curr, W_curr):
                    best_U3, best_V3, best_W3 = U_curr, V_curr, W_curr
                    best_rank_3 = target_rank
                    break
    
    # If we found a good 3x3x3 decomposition, use tensor product to get 3x3x6
    # T_{3,3,6} = T_{3,3,3} ⊗ T_{1,1,2}
    # T_{1,1,2} decomposition: rank 2
    # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1 (for k=0)
    # Term 2: U[0,1]=1, V[1,1]=1, W[1,1]=1 (for k=1)
    # But we need to map this to the 3x3x6 indices
    
    # For 3x3x6:
    # U has 9 rows (i,j for i,j in 0..2)
    # V has 18 rows (j,k for j in 0..2, k in 0..5)
    # W has 18 rows (k,i for k in 0..5, i in 0..2)
    
    # The tensor product structure:
    # k in 0..5 can be written as k = 3*s + k' where s in {0,1}, k' in {0,1,2}
    # So we have two "blocks" corresponding to s=0 and s=1
    
    # For each term r in the 3x3x3 decomposition, we create two terms in 3x3x6:
    # Term 2r: corresponds to s=0 (k' = k, so k = 0,1,2)
    # Term 2r+1: corresponds to s=1 (k' = k-3, so k = 3,4,5)
    
    if best_U3 is not None and best_rank_3 < 27:
        R3 = best_rank_3
        R6 = 2 * R3
        
        U6 = np.zeros((nm, R6), dtype=int)
        V6 = np.zeros((mp, R6), dtype=int)
        W6 = np.zeros((pn, R6), dtype=int)
        
        for r in range(R3):
            # Term 2r: s=0
            # U6[i*m+j, 2r] = U3[i*m+j, r]
            U6[:, 2*r] = best_U3[:, r]
            
            # V6[j*p+k, 2r] = V3[j*p3+k', r] where k = k' (s=0)
            # k in 0..2 for s=0
            for j in range(m3):
                for k_prime in range(p3):
                    k = k_prime  # s=0
                    V6[j * p + k, 2*r] = best_V3[j * p3 + k_prime, r]
            
            # W6[k*n+i, 2r] = W3[k'*n3+i, r] where k = k' (s=0)
            for k_prime in range(p3):
                k = k_prime  # s=0
                for i in range(n3):
                    W6[k * n + i, 2*r] = best_W3[k_prime * n3 + i, r]
            
            # Term 2r+1: s=1
            # U6[i*m+j, 2r+1] = U3[i*m+j, r]
            U6[:, 2*r+1] = best_U3[:, r]
            
            # V6[j*p+k, 2r+1] = V3[j*p3+k', r] where k = 3+k' (s=1)
            for j in range(m3):
                for k_prime in range(p3):
                    k = 3 + k_prime  # s=1
                    V6[j * p + k, 2*r+1] = best_V3[j * p3 + k_prime, r]
            
            # W6[k*n+i, 2r+1] = W3[k'*n3+i, r] where k = 3+k' (s=1)
            for k_prime in range(p3):
                k = 3 + k_prime  # s=1
                for i in range(n3):
                    W6[k * n + i, 2*r+1] = best_W3[k_prime * n3 + i, r]
        
        if verify_full_decomp(U6, V6, W6, T_full):
            return {
                "U": U6.tolist(),
                "V": V6.tolist(),
                "W": W6.tolist()
            }
    
    # Fallback: use schoolbook for 3x3x6 (rank 54)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=int)
    V_school = np.zeros((mp, rank_school), dtype=int)
    W_school = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
