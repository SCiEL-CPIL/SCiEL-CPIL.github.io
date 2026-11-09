import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    def verify(U, V, W, T, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T)) < tol:
                return True
        except:
            pass
        return False
    
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result

    # Strategy: Split 3x6 into two 3x3 blocks
    # Find rank-9 decomposition for 2x3 x 3x3
    n_sub, m_sub, p_sub = 2, 3, 3
    T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub), dtype=float)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1.0
    
    # Search for rank-9 decomposition of T_sub
    found_sub = False
    U_sub = None
    V_sub = None
    W_sub = None
    
    # Try structured approaches first
    # Approach 1: Use the known structure of 2x3x3 matrix multiplication
    # The 2x3x3 tensor can be decomposed using 9 terms if we exploit linearity
    
    # Let's try a systematic search with better initialization
    target_rank_sub = 9
    max_attempts_sub = 2000
    random.seed(42)
    
    for attempt in range(max_attempts_sub):
        if time.time() > deadline - 30:
            break
        
        U_try = np.zeros((n_sub*m_sub, target_rank_sub))
        V_try = np.zeros((m_sub*p_sub, target_rank_sub))
        W_try = np.zeros((p_sub*n_sub, target_rank_sub))
        
        # Better initialization: use structured patterns
        # Each rank-1 term should correspond to a "partial product"
        
        for r in range(target_rank_sub):
            # Try different sparsity patterns
            if attempt % 3 == 0:
                # Dense random
                for idx in range(n_sub*m_sub):
                    if random.random() < 0.7:
                        U_try[idx, r] = random.choice([-1, 1])
                for idx in range(m_sub*p_sub):
                    if random.random() < 0.7:
                        V_try[idx, r] = random.choice([-1, 1])
                for idx in range(p_sub*n_sub):
                    if random.random() < 0.7:
                        W_try[idx, r] = random.choice([-1, 1])
            elif attempt % 3 == 1:
                # Structured: each term handles one (i,j,k) combination
                # but with some mixing
                i_val = r % n_sub
                j_val = (r // n_sub) % m_sub
                k_val = (r // (n_sub*m_sub)) % p_sub
                
                U_try[i_val*m_sub+j_val, r] = 1
                # Add some neighboring terms
                if random.random() < 0.3:
                    U_try[(i_val+1)%n_sub*m_sub+j_val, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    U_try[i_val*m_sub+(j_val+1)%m_sub, r] = random.choice([-1, 1])
                
                V_try[j_val*p_sub+k_val, r] = 1
                if random.random() < 0.3:
                    V_try[j_val*p_sub+(k_val+1)%p_sub, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    V_try[(j_val+1)%m_sub*p_sub+k_val, r] = random.choice([-1, 1])
                
                W_try[k_val*n_sub+i_val, r] = 1
                if random.random() < 0.3:
                    W_try[(k_val+1)%p_sub*n_sub+i_val, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    W_try[k_val*n_sub+(i_val+1)%n_sub, r] = random.choice([-1, 1])
            else:
                # Binary sparse with higher density
                for idx in range(n_sub*m_sub):
                    if random.random() < 0.8:
                        U_try[idx, r] = random.choice([-1, 1])
                for idx in range(m_sub*p_sub):
                    if random.random() < 0.8:
                        V_try[idx, r] = random.choice([-1, 1])
                for idx in range(p_sub*n_sub):
                    if random.random() < 0.8:
                        W_try[idx, r] = random.choice([-1, 1])
        
        if verify(U_try, V_try, W_try, T_sub):
            U_sub = U_try
            V_sub = V_try
            W_sub = W_try
            found_sub = True
            break
    
    if found_sub:
        # Construct the full rank-18 solution
        U_full = np.zeros((n*m, 18))
        V_full = np.zeros((m*p, 18))
        W_full = np.zeros((p*n, 18))
        
        # First 9 columns for the first 3x3 block
        U_full[:, :9] = U_sub
        # V_full for the first block: rows 0-8 (j=0,1,2 and k=0,1,2)
        V_full[:9, :9] = V_sub
        # W_full for the first block: rows 0-5 (k=0,1,2 and i=0,1)
        W_full[:6, :9] = W_sub
        
        # Next 9 columns for the second 3x3 block
        U_full[:, 9:] = U_sub
        # V_full for the second block: rows 9-17 (j=0,1,2 and k=3,4,5)
        V_full[9:, 9:] = V_sub
        # W_full for the second block: rows 6-11 (k=3,4,5 and i=0,1)
        W_full[6:, 9:] = W_sub
        
        if verify(U_full, V_full, W_full, T):
            best_rank = 18
            best_U, best_V, best_W = U_full, U_full, W_full
            best_U, best_V, best_W = U_full, V_full, W_full
    
    # If no lower rank found, return schoolbook
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
