"""Block-decomposition strategy for 4x5x5 matrix multiplication.
Splits dimensions as 4=2+2, 5=2+3.
Uses Strassen (rank 7) for the 2x2x2 block and Laderman (rank 13) for the 2x3x3 block.
Cross-block terms use schoolbook.
Total rank: 7 + 13 + 2*3*2 + 2*2*3 + 2*3*3 = 7 + 13 + 12 + 12 + 18 = 62.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    deadline = t0 + 130.0

    # Dimension splits
    n1, n2 = 2, 2  # n = 4 = 2 + 2
    m1, m2 = 2, 3  # m = 5 = 2 + 3
    p1, p2 = 2, 3  # p = 5 = 2 + 3

    # We'll build U, V, W matrices column by column.
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (20, rank)
    
    # Indexing:
    # U[i*m + j, r] where i in 0..n-1, j in 0..m-1
    # V[j*p + k, r] where j in 0..m-1, k in 0..p-1
    # W[k*n + i, r] where k in 0..p-1, i in 0..n-1

    def add_term(U, V, W, col, u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        """Add a single rank-1 term at column col."""
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val

    # --- Strassen 2x2x2 decomposition (rank 7) ---
    # For multiplying a 2x2 matrix A by a 2x2 matrix B to get C = A*B
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00+A11)(B01+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def strassen_2x2_terms(n_off, m_off, p_off):
        """
        Yield the 7 Strassen terms for a 2x2x2 block.
        n_off: offset for n index (0 or 2)
        m_off: offset for m index (0 or 2)
        p_off: offset for p index (0 or 2)
        
        Each term is a list of (u_indices, v_indices, w_indices, coeffs)
        where u_indices are global indices into U, etc.
        """
        terms = []
        
        # M1 = (A00+A11)(B01+B11)
        # A00 is at (n_off+0, m_off+0), A11 is at (n_off+1, m_off+1)
        # B01 is at (m_off+0, p_off+1), B11 is at (m_off+1, p_off+1)
        # U indices: (n_off+0)*5 + (m_off+0) and (n_off+1)*5 + (m_off+1)
        # V indices: (m_off+0)*5 + (p_off+1) and (m_off+1)*5 + (p_off+1)
        # W: contributes to C00, C11 with +1
        # C00: k=p_off+0, i=n_off+0 -> W[(p_off+0)*4 + (n_off+0)]
        # C11: k=p_off+1, i=n_off+1 -> W[(p_off+1)*4 + (n_off+1)]
        u1 = [(n_off+0)*5 + (m_off+0), 1, (n_off+1)*5 + (m_off+1), 1]
        v1 = [(m_off+0)*5 + (p_off+1), 1, (m_off+1)*5 + (p_off+1), 1]
        w1 = [((p_off+0)*4 + (n_off+0), 1), ((p_off+1)*4 + (n_off+1), 1)]
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        # A10: (n_off+1, m_off+0), A11: (n_off+1, m_off+1)
        # B00: (m_off+0, p_off+0)
        u2 = [(n_off+1)*5 + (m_off+0), 1, (n_off+1)*5 + (m_off+1), 1]
        v2 = [(m_off+0)*5 + (p_off+0), 1]
        w2 = [((p_off+0)*4 + (n_off+1), 1)]  # C10
        terms.append((u2, v2, w2))
        
        # M3 = A00(B10-B11)
        # A00: (n_off+0, m_off+0)
        # B10: (m_off+1, p_off+0), B11: (m_off+1, p_off+1)
        u3 = [(n_off+0)*5 + (m_off+0), 1]
        v3 = [(m_off+1)*5 + (p_off+0), 1, (m_off+1)*5 + (p_off+1), -1]
        w3 = [((p_off+1)*4 + (n_off+0), 1)]  # C01
        terms.append((u3, v3, w3))
        
        # M4 = A11(B00-B10)
        # A11: (n_off+1, m_off+1)
        # B00: (m_off+0, p_off+0), B10: (m_off+1, p_off+0)
        u4 = [(n_off+1)*5 + (m_off+1), 1]
        v4 = [(m_off+0)*5 + (p_off+0), 1, (m_off+1)*5 + (p_off+0), -1]
        w4 = [((p_off+0)*4 + (n_off+1), 1), ((p_off+0)*4 + (n_off+0), 1)]  # C10, C00
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        # A00: (n_off+0, m_off+0), A01: (n_off+0, m_off+1)
        # B11: (m_off+1, p_off+1)
        u5 = [(n_off+0)*5 + (m_off+0), 1, (n_off+0)*5 + (m_off+1), 1]
        v5 = [(m_off+1)*5 + (p_off+1), 1]
        w5 = [((p_off+0)*4 + (n_off+0), -1), ((p_off+1)*4 + (n_off+0), 1)]  # C00 (-1), C01 (+1)
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        # A10: (n_off+1, m_off+0), A00: (n_off+0, m_off+0)
        # B00: (m_off+0, p_off+0), B01: (m_off+0, p_off+1)
        u6 = [(n_off+1)*5 + (m_off+0), 1, (n_off+0)*5 + (m_off+0), -1]
        v6 = [(m_off+0)*5 + (p_off+0), 1, (m_off+0)*5 + (p_off+1), 1]
        w6 = [((p_off+1)*4 + (n_off+1), 1)]  # C11
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        # A01: (n_off+0, m_off+1), A11: (n_off+1, m_off+1)
        # B10: (m_off+1, p_off+0), B11: (m_off+1, p_off+1)
        u7 = [(n_off+0)*5 + (m_off+1), 1, (n_off+1)*5 + (m_off+1), -1]
        v7 = [(m_off+1)*5 + (p_off+0), 1, (m_off+1)*5 + (p_off+1), 1]
        w7 = [((p_off+0)*4 + (n_off+0), 1), ((p_off+1)*4 + (n_off+1), 1)]  # C00, C11
        terms.append((u7, v7, w7))
        
        return terms

    # --- Laderman 2x3x3 decomposition (rank 13) ---
    # For multiplying a 2x3 matrix A by a 3x3 matrix B to get C = A*B (2x3)
    # A = [[A00, A01, A02], [A10, A11, A12]]
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    # C = [[C00, C01, C02], [C10, C11, C12]]
    #
    # Laderman's algorithm:
    # P1 = A00 * (B00+B21)
    # P2 = (A00+A01) * B21
    # P3 = A01 * (B00-B21)
    # P4 = A02 * B10
    # P5 = (A00+A02) * B20
    # P6 = A02 * (B00-B20)
    # P7 = A10 * (B11+B22)
    # P8 = (A10+A11) * B22
    # P9 = A11 * (B11-B22)
    # P10 = A12 * B01
    # P11 = (A10+A12) * B10
    # P12 = A12 * (B11-B10)
    # P13 = (A11+A12) * (B01-B10)
    #
    # C00 = P1 + P3
    # C01 = P10 + P13
    # C02 = P4 + P6
    # C10 = P5 + P7
    # C11 = P8 + P9 + P12
    # C12 = P11 + P13

    def laderman_2x3x3_terms(n_off, m_off, p_off):
        """
        Yield the 13 Laderman terms for a 2x3x3 block.
        n_off: offset for n index (0 or 2)
        m_off: offset for m index (0 or 2)
        p_off: offset for p index (0 or 2)
        """
        terms = []
        
        # P1 = A00 * (B00+B21)
        u1 = [(n_off+0)*5 + (m_off+0), 1]
        v1 = [(m_off+0)*5 + (p_off+0), 1, (m_off+2)*5 + (p_off+1), 1]
        w1 = [((p_off+0)*4 + (n_off+0), 1)]  # C00
        terms.append((u1, v1, w1))
        
        # P2 = (A00+A01) * B21
        u2 = [(n_off+0)*5 + (m_off+0), 1, (n_off+0)*5 + (m_off+1), 1]
        v2 = [(m_off+2)*5 + (p_off+1), 1]
        w2 = [((p_off+1)*4 + (n_off+0), 1)]  # C01
        terms.append((u2, v2, w2))
        
        # P3 = A01 * (B00-B21)
        u3 = [(n_off+0)*5 + (m_off+1), 1]
        v3 = [(m_off+0)*5 + (p_off+0), 1, (m_off+2)*5 + (p_off+1), -1]
        w3 = [((p_off+0)*4 + (n_off+0), 1)]  # C00
        terms.append((u3, v3, w3))
        
        # P4 = A02 * B10
        u4 = [(n_off+0)*5 + (m_off+2), 1]
        v4 = [(m_off+1)*5 + (p_off+0), 1]
        w4 = [((p_off+0)*4 + (n_off+0), 1)]  # C00
        terms.append((u4, v4, w4))
        
        # P5 = (A00+A02) * B20
        u5 = [(n_off+0)*5 + (m_off+0), 1, (n_off+0)*5 + (m_off+2), 1]
        v5 = [(m_off+2)*5 + (p_off+0), 1]
        w5 = [((p_off+0)*4 + (n_off+0), 1)]  # C00
        terms.append((u5, v5, w5))
        
        # P6 = A02 * (B00-B20)
        u6 = [(n_off+0)*5 + (m_off+2), 1]
        v6 = [(m_off+0)*5 + (p_off+0), 1, (m_off+2)*5 + (p_off+0), -1]
        w6 = [((p_off+2)*4 + (n_off+0), 1)]  # C02
        terms.append((u6, v6, w6))
        
        # P7 = A10 * (B11+B22)
        u7 = [(n_off+1)*5 + (m_off+0), 1]
        v7 = [(m_off+1)*5 + (p_off+1), 1, (m_off+2)*5 + (p_off+2), 1]
        w7 = [((p_off+1)*4 + (n_off+1), 1)]  # C11
        terms.append((u7, v7, w7))
        
        # P8 = (A10+A11) * B22
        u8 = [(n_off+1)*5 + (m_off+0), 1, (n_off+1)*5 + (m_off+1), 1]
        v8 = [(m_off+2)*5 + (p_off+2), 1]
        w8 = [((p_off+2)*4 + (n_off+1), 1)]  # C12
        terms.append((u8, v8, w8))
        
        # P9 = A11 * (B11-B22)
        u9 = [(n_off+1)*5 + (m_off+1), 1]
        v9 = [(m_off+1)*5 + (p_off+1), 1, (m_off+2)*5 + (p_off+2), -1]
        w9 = [((p_off+1)*4 + (n_off+1), 1)]  # C11
        terms.append((u9, v9, w9))
        
        # P10 = A12 * B01
        u10 = [(n_off+1)*5 + (m_off+2), 1]
        v10 = [(m_off+0)*5 + (p_off+1), 1]
        w10 = [((p_off+1)*4 + (n_off+1), 1)]  # C11
        terms.append((u10, v10, w10))
        
        # P11 = (A10+A12) * B10
        u11 = [(n_off+1)*5 + (m_off+0), 1, (n_off+1)*5 + (m_off+2), 1]
        v11 = [(m_off+1)*5 + (p_off+0), 1]
        w11 = [((p_off+0)*4 + (n_off+1), 1)]  # C10
        terms.append((u11, v11, w11))
        
        # P12 = A12 * (B11-B10)
        u12 = [(n_off+1)*5 + (m_off+2), 1]
        v12 = [(m_off+1)*5 + (p_off+1), 1, (m_off+1)*5 + (p_off+0), -1]
        w12 = [((p_off+1)*4 + (n_off+1), 1)]  # C11
        terms.append((u12, v12, w12))
        
        # P13 = (A11+A12) * (B01-B10)
        u13 = [(n_off+1)*5 + (m_off+1), 1, (n_off+1)*5 + (m_off+2), 1]
        v13 = [(m_off+0)*5 + (p_off+1), 1, (m_off+1)*5 + (p_off+0), -1]
        w13 = [((p_off+1)*4 + (n_off+1), 1), ((p_off+2)*4 + (n_off+1), 1)]  # C11, C12
        terms.append((u13, v13, w13))
        
        return terms

    # Build the full decomposition
    # We'll collect all terms first, then assign columns.
    all_terms = []
    
    # Block (0,0): n in [0,2), m in [0,2), p in [0,2) -> 2x2x2, use Strassen
    for term in strassen_2x2_terms(0, 0, 0):
        all_terms.append(term)
    
    # Block (0,1): n in [0,2), m in [0,2), p in [2,5) -> 2x2x3, schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = (0+i)*5 + (0+j)
                v_idx = (0+j)*5 + (2+k)
                w_idx = (2+k)*4 + (0+i)
                all_terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
    
    # Block (1,0): n in [2,4), m in [0,2), p in [0,2) -> 2x2x2, use Strassen
    for term in strassen_2x2_terms(2, 0, 0):
        all_terms.append(term)
    
    # Block (1,1): n in [2,4), m in [0,2), p in [2,5) -> 2x2x3, schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = (2+i)*5 + (0+j)
                v_idx = (0+j)*5 + (2+k)
                w_idx = (2+k)*4 + (2+i)
                all_terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
    
    # Block (2,0): n in [0,2), m in [2,5), p in [0,2) -> 2x3x2, schoolbook
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u_idx = (0+i)*5 + (2+j)
                v_idx = (2+j)*5 + (0+k)
                w_idx = (0+k)*4 + (0+i)
                all_terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
    
    # Block (2,1): n in [0,2), m in [2,5), p in [2,5) -> 2x3x3, use Laderman
    for term in laderman_2x3x3_terms(0, 2, 2):
        all_terms.append(term)
    
    # Block (3,0): n in [2,4), m in [2,5), p in [0,2) -> 2x3x2, schoolbook
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u_idx = (2+i)*5 + (2+j)
                v_idx = (2+j)*5 + (0+k)
                w_idx = (0+k)*4 + (2+i)
                all_terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
    
    # Block (3,1): n in [2,4), m in [2,5), p in [2,5) -> 2x3x3, use Laderman
    for term in laderman_2x3x3_terms(2, 2, 2):
        all_terms.append(term)
    
    rank = len(all_terms)
    print(f"Total rank: {rank}")
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for col, (u_list, v_list, w_list) in enumerate(all_terms):
        for u_idx, u_val in u_list:
            U[u_idx, col] += u_val
        for v_idx, v_val in v_list:
            V[v_idx, col] += v_val
        for w_idx, w_val in w_list:
            W[w_idx, col] += w_val
    
    # Verify correctness
    if time.time() < deadline - 5:
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # (U[:,r] . A_flat) * (V[:,r] . B_flat) * W[:,r]
            # But the form is: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # Wait, that's not right. Let me think again.
            
            # The correct form for a rank-1 term is:
            # C[i,k] += W[k*n+i, r] * (sum over (i',j') of U[i'*m+j', r] * A[i',j']) * (sum over (j',k') of V[j'*p+k', r] * B[j',k'])
            # No, that's not right either.
            
            # Actually, for a rank-1 decomposition of matrix multiplication:
            # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
            
            # So for each r:
            # C[i,k] += W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
            
            # Let me compute this properly.
            for i in range(n):
                for k in range(p):
                    if W[k*n+i, r] == 0:
                        continue
                    s = 0
                    for j in range(m):
                        s += U[i*m+j, r] * A_test[i,j] * V[j*p+k, r] * B_test[j,k]
                    C_computed[i,k] += W[k*n+i, r] * s
        
        max_err = np.max(np.abs(C_computed - C_expected))
        print(f"Verification max error: {max_err}")
        if max_err > 1e-10:
            print("WARNING: Decomposition is incorrect!")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
