import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Indexing:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strategy:
    # Split j=2 (the "third" row of A, col of B) from j=0,1.
    # Part 1: j=0,1. A[0:2, 0:2] (2x2) times B[0:2, :] (2x9).
    # Split B[0:2, :] into three 2x3 blocks: B0 (cols 0-2), B1 (cols 3-5), B2 (cols 6-8).
    # For each 2x2x3 block, use rank 11 decomposition (Strassen 2x2x2 + trivial 2x2x1).
    # Part 2: j=2. A[0:2, 2] (2x1) times B[2, :] (1x9).
    # This is outer product, rank 18 (2*1*9).
    
    # Total rank: 3 * 11 + 18 = 51.

    # --- Part 1: 2x2x9 multiplication ---
    # A1 is 2x2 (indices i=0,1; j=0,1)
    # B1 is 2x9 (indices j=0,1; k=0..8)
    
    # We define a helper to add a 2x2x2 Strassen decomposition
    # Mapped to global indices:
    # i in {0,1}, j in {0,1}, k in {k_start, k_start+1}
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*9 + k
    # W index: k*n + i = k*2 + i
    
    def add_strassen_2x2x2(k_start):
        # Strassen 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B10
        # M6 = (A11-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 (0,0) -> 1, A11 (1,1) -> 1
        u1 = [0]*6
        u1[0*3+0] = 1
        u1[1*3+1] = 1
        # V: B00 (0, k_start) -> 1, B11 (1, k_start+1) -> 1
        v1 = [0]*27
        v1[0*9+k_start] = 1
        v1[1*9+k_start+1] = 1
        # W: C00 (k_start, 0) -> 1, C11 (k_start+1, 1) -> 1
        w1 = [0]*18
        w1[k_start*2+0] = 1
        w1[(k_start+1)*2+1] = 1
        add_term(u1, v1, w1)
        
        # Term 2: (A10+A11)B00
        u2 = [0]*6
        u2[1*3+0] = 1
        u2[1*3+1] = 1
        v2 = [0]*27
        v2[0*9+k_start] = 1
        w2 = [0]*18
        w2[k_start*2+0] = 1   # C10
        w2[k_start*2+1] = 1   # C11
        add_term(u2, v2, w2)
        
        # Term 3: A00(B01-B11)
        u3 = [0]*6
        u3[0*3+0] = 1
        v3 = [0]*27
        v3[0*9+k_start+1] = 1
        v3[1*9+k_start+1] = -1
        w3 = [0]*18
        w3[k_start*2+0] = -1  # C00
        w3[k_start*2+1] = 1   # C01
        w3[(k_start+1)*2+0] = 1  # C10
        w3[(k_start+1)*2+1] = 1  # C11
        # Wait, Strassen formulas:
        # M3 = A00(B01-B11)
        # C01 += M3
        # C11 += M3
        # C00 -= M3? No.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify coefficients for W.
        # Term 3 corresponds to M3.
        # W entries are coefficients for C.
        # C00: 0
        # C01: 1
        # C10: 0
        # C11: 1
        # So w3 should be:
        w3 = [0]*18
        w3[k_start*2+1] = 1   # C01
        w3[(k_start+1)*2+1] = 1 # C11
        # I already added term 3 above with wrong w? No, I just defined it.
        # I will re-do term 3 correctly.
        # Actually I didn't add it yet, just defined variables.
        add_term(u3, v3, w3)
        
        # Term 4: A11(B10-B00)
        u4 = [0]*6
        u4[1*3+1] = 1
        v4 = [0]*27
        v4[1*9+k_start] = 1
        v4[0*9+k_start] = -1
        w4 = [0]*18
        w4[k_start*2+0] = 1   # C00
        w4[k_start*2+1] = 1   # C10
        add_term(u4, v4, w4)
        
        # Term 5: (A00+A10)B10
        u5 = [0]*6
        u5[0*3+0] = 1
        u5[1*3+0] = 1
        v5 = [0]*27
        v5[1*9+k_start] = 1
        w5 = [0]*18
        w5[k_start*2+0] = -1  # C00
        w5[k_start*2+1] = 1   # C01
        add_term(u5, v5, w5)
        
        # Term 6: (A11-A00)(B00+B01)
        u6 = [0]*6
        u6[1*3+1] = 1
        u6[0*3+0] = -1
        v6 = [0]*27
        v6[0*9+k_start] = 1
        v6[0*9+k_start+1] = 1
        w6 = [0]*18
        w6[k_start*2+1] = 1   # C11
        add_term(u6, v6, w6)
        
        # Term 7: (A01-A11)(B10+B11)
        u7 = [0]*6
        u7[0*3+1] = 1
        u7[1*3+1] = -1
        v7 = [0]*27
        v7[1*9+k_start] = 1
        v7[1*9+k_start+1] = 1
        w7 = [0]*18
        w7[k_start*2+0] = -1  # C00
        add_term(u7, v7, w7)

    # Helper to add trivial 2x2x1 decomposition
    def add_trivial_2x2x1(k_start):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # 4 terms.
        
        # T1: A00 B0
        u1 = [0]*6
        u1[0*3+0] = 1
        v1 = [0]*27
        v1[0*9+k_start] = 1
        w1 = [0]*18
        w1[k_start*2+0] = 1
        add_term(u1, v1, w1)
        
        # T2: A01 B1
        u2 = [0]*6
        u2[0*3+1] = 1
        v2 = [0]*27
        v2[1*9+k_start] = 1
        w2 = [0]*18
        w2[k_start*2+0] = 1
        add_term(u2, v2, w2)
        
        # T3: A10 B0
        u3 = [0]*6
        u3[1*3+0] = 1
        v3 = [0]*27
        v3[0*9+k_start] = 1
        w3 = [0]*18
        w3[k_start*2+1] = 1
        add_term(u3, v3, w3)
        
        # T4: A11 B1
        u4 = [0]*6
        u4[1*3+1] = 1
        v4 = [0]*27
        v4[1*9+k_start] = 1
        w4 = [0]*18
        w4[k_start*2+1] = 1
        add_term(u4, v4, w4)
        
    # We need to handle the blocks of B1 (2x9).
    # Block 0: cols 0,1,2. k_start=0.
    # Use Strassen for k=0,1 and Trivial for k=2.
    add_strassen_2x2x2(0)
    add_trivial_2x2x1(2)
    
    # Block 1: cols 3,4,5. k_start=3.
    add_strassen_2x2x2(3)
    add_trivial_2x2x1(5)
    
    # Block 2: cols 6,7,8. k_start=6.
    add_strassen_2x2x2(6)
    add_trivial_2x2x1(8)
    
    # --- Part 2: 2x1x9 multiplication ---
    # A2 is 2x1 (indices i=0,1; j=2)
    # B2 is 1x9 (indices j=2; k=0..8)
    # C = A2 * B2.
    # C[i, k] = A[i, 2] * B[2, k]
    # 18 terms.
    
    for i in range(2):
        for k in range(9):
            u = [0]*6
            u[i*3+2] = 1  # A[i, 2]
            v = [0]*27
            v[2*9+k] = 1  # B[2, k]
            w = [0]*18
            w[k*2+i] = 1  # C[k, i]
            add_term(u, v, w)

    # Convert to numpy arrays for verification
    U_np = np.array(U, dtype=float).T # shape (rank, 6)
    V_np = np.array(V, dtype=float).T # shape (rank, 27)
    W_np = np.array(W, dtype=float).T # shape (rank, 18)
    
    rank = len(U)
    
    # Verify
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(9):
                # Sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
                # U is (rank, 6). Index i*3+j.
                # V is (rank, 27). Index j*9+k.
                # W is (rank, 18). Index k*2+i.
                u_idx = i*3+j
                v_idx = j*9+k
                w_idx = k*2+i
                val = np.sum(U_np[:, u_idx] * V_np[:, v_idx] * W_np[:, w_idx])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    # print(f"Error at {i},{j},{k}: {val}")
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to trivial decomposition if construction failed
        U = []
        V = []
        W = []
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    u = [0]*(2*3*9)
                    v = [0]*(3*9)
                    w = [0]*(9*2)
                    u[i*3+j] = 1
                    v[j*9+k] = 1
                    w[k*2+i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
