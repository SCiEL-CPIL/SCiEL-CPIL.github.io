import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Strategy: Recursive block decomposition
    # 6x6 = 2x2 blocks of 3x3
    # Apply Strassen (rank 7) at block level, schoolbook (rank 27) at sub-block level
    # Total rank = 7 * 27 = 189
    
    # Alternatively, try 161 = 7 * 23 (Strassen x Laderman) if we have valid Laderman terms
    # But Laderman coefficients are tricky to get exactly right.
    # For now, stick with 189 which is proven correct.
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # Indices within the 2x2 block
                    # Block (i, k) for A, block (k, j) for B, block (j, i) for C
                    # Within each 3x3 block, we use schoolbook: 9 terms
                    # But we're using Strassen on the 2x2 block structure
                    
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    c00 = (2*j) * 6 + 2*i      
                    c01 = (2*j+1) * 6 + 2*i    
                    c10 = (2*j) * 6 + (2*i+1)  
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:
                        U[a00, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1; W[c11, idx] = 1
                    elif s == 1:
                        U[a10, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1; W[c11, idx] = -1
                    elif s == 2:
                        U[a00, idx] = 1
                        V[b01, idx] = 1; V[b11, idx] = -1
                        W[c01, idx] = 1; W[c11, idx] = 1
                    elif s == 3:
                        U[a11, idx] = 1
                        V[b10, idx] = 1; V[b00, idx] = -1
                        W[c00, idx] = 1; W[c10, idx] = 1
                    elif s == 4:
                        U[a00, idx] = 1; U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1; W[c01, idx] = 1
                    elif s == 5:
                        U[a10, idx] = 1; U[a00, idx] = -1
                        V[b00, idx] = 1; V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:
                        U[a01, idx] = 1; U[a11, idx] = -1
                        V[b10, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verification using the official trace convention: W[k*n+i, r]
    # C[i,j] = sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) * W[j*n+i, r]
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        # U has shape (n*m, rank), indexed as U[i*m+k, r]
        # V has shape (m*p, rank), indexed as V[k*p+j, r]
        # W has shape (p*n, rank), indexed as W[j*n+i, r]
        
        # Compute C[i,j] = sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) * W[j*n+i, r]
        # First compute M[i,j,r] = sum_k U[i*m+k, r] * V[k*p+j, r]
        U_reshaped = U.astype(float).reshape(n, m, rank)  # U[i,k,r]
        V_reshaped = V.astype(float).reshape(m, p, rank)  # V[k,j,r]
        W_reshaped = W.astype(float).reshape(p, n, rank)  # W[j,i,r]
        
        # M[i,j,r] = sum_k U[i,k,r] * V[k,j,r]
        M = np.einsum('ikr,kjr->ijr', U_reshaped, V_reshaped)
        
        # C[i,j] = sum_r M[i,j,r] * W[j,i,r]
        C_computed = np.einsum('ijr,jir->ij', M, W_reshaped)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Try alternative W convention: W[i*n+j, r]
            W_reshaped_alt = W.astype(float).reshape(n, p, rank)  # W[i,j,r]
            C_computed_alt = np.einsum('ijr,ijr->ij', M, W_reshaped_alt)
            if np.allclose(C_computed_alt, C_expected, atol=1e-9):
                # The W matrix is using W[i*n+j, r] convention, need to transpose
                W_fixed = W_reshaped_alt.transpose(1, 0, 2).reshape(p * n, rank)
                return {"U": U.tolist(), "V": V.tolist(), "W": W_fixed.astype(int).tolist()}
            
            # Fallback to schoolbook with correct W convention W[j*n+i, r]
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
