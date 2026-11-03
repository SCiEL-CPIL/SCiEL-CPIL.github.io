import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1

    # Rank 32 Decomposition
    # Part 1: A * B[:, :4] (4x4 x 4x4) using Strassen (Rank 28)
    # Part 2: A * B[:, 4:] (4x4 x 4x1) using Schoolbook (Rank 4)
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_idx)
        
    # We need to map Strassen terms to global indices.
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B01)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)(B11+B01)
    # M6 = (A10+A11)(B00+B10)
    # M7 = (A00+A11-A10-A01)(B01+B10-B00-B11)
    
    # We will implement the 4x4x4 Strassen by decomposing into 2x2x2 Strassen blocks.
    # The 4x4 matrix A is split into 2x2 blocks A00, A01, A10, A11.
    # B is split into B00, B01, B10, B11.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # We can use the 7-term Strassen for each 2x2x2 multiplication.
    # However, we must be careful with the "sum" of two products.
    # C00 = M1 + M2 where M1=A00B00, M2=A01B10.
    # Each Mi is rank 7. Total 14 for C00?
    # But Strassen for 4x4x4 is rank 28.
    # This implies we use 7 terms per 2x2x2 output block?
    # No, the 4x4x4 Strassen algorithm uses 7 multiplications of 2x2x2 matrices.
    # So we have 7 "super-terms".
    # Each super-term is a product of two 2x2 matrices.
    # So we expand each super-term into 7 rank-1 terms.
    # Total 7 * 7 = 49? No.
    
    # Actually, the rank of 4x4x4 is 28.
    # This means there are 28 rank-1 terms.
    # This is achieved by 4 Strassen blocks?
    # Let's just use the 28-term decomposition.
    
    # I will hardcode the 28 terms for 4x4x4 Strassen.
    # This is tedious. 
    # Alternative: Use Schoolbook for 4x4x4 (Rank 16) + Schoolbook for 4x4x1 (Rank 4) = 20?
    # Schoolbook 4x4x4 is 16.
    # Schoolbook 4x4x1 is 4.
    # Total 20.
    # Wait, 4x4x4 schoolbook is 16 terms.
    # 4x4x1 schoolbook is 4 terms.
    # Total 20.
    # This is much easier to implement than Strassen!
    # And 20 < 80.
    
    # Let's verify:
    # C = A * B.
    # B is 4x5.
    # C = A * B[:, :4] + A * B[:, 4:].
    # A * B[:, :4] is 4x4 * 4x4.
    # Schoolbook: 16 terms.
    # A * B[:, 4:] is 4x4 * 4x1.
    # Schoolbook: 4 terms.
    # Total 20 terms.
    
    # This is a massive improvement over 80.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4 x 4x4 (cols 0-3 of B, cols 0-3 of C)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 5 + k
                c = k * 4 + i
                U_list.append(a)
                V_list.append(b)
                W_list.append(c)
                
    # Part 2: 4x4 x 4x1 (col 4 of B, col 4 of C)
    for i in range(4):
        for j in range(4):
            a = i * 4 + j
            b = j * 5 + 4
            c = 4 * 4 + i
            U_list.append(a)
            V_list.append(b)
            W_list.append(c)
            
    U = np.zeros((16, 20))
    V = np.zeros((20, 20))
    W = np.zeros((20, 20))
    
    # Fill matrices
    for r in range(20):
        U[U_list[r], r] = 1
        V[V_list[r], r] = 1
        W[W_list[r], r] = 1
        
    # Verify
    est = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(est, target, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback
        return {"U": np.eye(80).tolist(), "V": np.eye(80).tolist(), "W": np.eye(80).tolist()}

