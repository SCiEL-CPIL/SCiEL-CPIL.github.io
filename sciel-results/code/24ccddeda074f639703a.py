import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0

    # Helper to verify a decomposition
    def verify(U, V, W, rank):
        if time.time() > deadline - 5:
            return False
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float).reshape(n, m, rank)
        V_arr = V.astype(float).reshape(m, p, rank)
        W_arr = W.astype(float).reshape(p, n, rank)
        
        prod = np.einsum('ikt,kjt->ijt', U_arr, V_arr)
        C_computed = np.einsum('ijt,jit->ij', prod, W_arr)
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    # Fallback: Rank 189 (Strassen 2x2 x Schoolbook 3x3)
    def get_rank_189():
        rank = 189
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        idx = 0
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for s in range(7):
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
        
        assert idx == rank
        return U, V, W, rank

    # Attempt Rank 133 (Laderman 3x3 x Strassen 2x2)
    # Laderman 3x3 rank 19
    # We need the 19 terms for 3x3.
    # If we cannot verify the 19 terms, we fallback.
    
    # I will use a known valid set of 19 terms.
    # Source: Laderman 1976.
    # Indices 0-8 for 3x3.
    
    U3 = np.zeros((9, 19), dtype=int)
    V3 = np.zeros((9, 19), dtype=int)
    W3 = np.zeros((9, 19), dtype=int)
    
    # Term 1: A00(B00+B11)
    U3[0, 0] = 1
    V3[0, 0] = 1; V3[4, 0] = 1
    W3[0, 0] = 1 # C00
    
    # Term 2: (A10+A11)B00
    U3[3, 1] = 1; U3[4, 1] = 1
    V3[0, 1] = 1
    W3[3, 1] = 1; W3[4, 1] = -1 # C10, C11? No, C10 = A10B00 + ..., C11 = A11B00...
    # This manual entry is too risky.
    
    # Instead, I will use the Rank 189 solution which is guaranteed correct.
    # The improvement from 189 to 133 is significant, but the risk of implementation error is high.
    # Given the constraints, I will stick to the robust 189 solution.
    
    U, V, W, rank = get_rank_189()
    
    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 216 if 189 fails (should not happen)
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

# EVOLVE-BLOCK-END
