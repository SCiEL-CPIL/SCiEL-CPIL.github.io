import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0
    
    # Try 2x2 blocks of 3x3 with rank-19 3x3 decomposition: total rank = 8 * 19 = 152
    # If that fails, fall back to 3x3 blocks of 2x2 with rank-7 2x2: total rank = 27 * 7 = 189
    
    def build_2x2_block_strassen():
        """3x3 blocks of 2x2, rank = 27*7 = 189"""
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
        return U, V, W, rank
    
    def build_2x2_blocks_3x3_rank19():
        """2x2 blocks of 3x3, using rank-19 3x3 decomposition: total rank = 8*19 = 152"""
        # 19-term decomposition for 3x3 matrix multiplication
        # Each term: (u_vec, v_vec, w_vec) where u_vec has 9 entries (for 3x3 A block),
        # v_vec has 9 entries (for 3x3 B block), w_vec has 9 entries (for 3x3 C block)
        # Using 0-indexed: A[i,j] is at position i*3+j
        
        # The 19 bilinear forms for 3x3 (Landsberg 1993 construction)
        # I'll use the known 19-term set
        terms_3x3 = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # Term 0: (a00+a11+a22)(b00+b11+b22) -> contributes to c00,c11,c22
            ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
            # Term 1: (a01+a12+a20)(b01+b12+b20) -> contributes to c01,c12,c20
            ([0,1,0, 0,0,1, 1,0,0], [0,1,0, 0,0,1, 1,0,0], [0,1,0, 0,0,1, 1,0,0]),
            # Term 2: (a02+a10+a21)(b02+b10+b21) -> contributes to c02,c10,c21
            ([0,0,1, 1,0,0, 0,1,0], [0,0,1, 1,0,0, 0,1,0], [0,0,1, 1,0,0, 0,1,0]),
            # Term 3: (a00+a11-a22)(b00+b11-b22)
            ([1,0,0, 0,1,0, 0,0,-1], [1,0,0, 0,1,0, 0,0,-1], [1,0,0, 0,1,0, 0,0,-1]),
            # Term 4: (a01+a12-a20)(b01+b12-b20)
            ([0,1,0, 0,0,1, -1,0,0], [0,1,0, 0,0,1, -1,0,0], [0,1,0, 0,0,1, -1,0,0]),
            # Term 5: (a02+a10-a21)(b02+b10-b21)
            ([0,0,1, 1,0,0, 0,-1,0], [0,0,1, 1,0,0, 0,-1,0], [0,0,1, 1,0,0, 0,-1,0]),
            # Term 6: (a00-a11+a22)(b00-b11+b22)
            ([1,0,0, 0,-1,0, 0,0,1], [1,0,0, 0,-1,0, 0,0,1], [1,0,0, 0,-1,0, 0,0,1]),
            # Term 7: (a01-a12+a20)(b01-b12+b20)
            ([0,1,0, 0,0,-1, 1,0,0], [0,1,0, 0,0,-1, 1,0,0], [0,1,0, 0,0,-1, 1,0,0]),
            # Term 8: (a02-a10+a21)(b02-b10+b21)
            ([0,0,1, -1,0,0, 0,1,0], [0,0,1, -1,0,0, 0,1,0], [0,0,1, -1,0,0, 0,1,0]),
            # Term 9-18: additional terms to complete the 19-term decomposition
            # These are more complex and involve specific combinations
            # For safety, I'll use a verified approach
        ]
        
        # The above 9 terms are not sufficient for exact 3x3 decomposition.
        # The full 19-term decomposition is complex. Let me use a different strategy.
        # Instead, I'll verify if a simpler improvement is possible.
        
        # Actually, let me use the known result: the best known rank for 3x3 is 19,
        # but implementing it correctly is complex. Let me try a different approach.
        
        # Use 2x2 Strassen on 3x3 blocks won't work directly since 3x3 isn't 2x2.
        
        # Alternative: use the fact that 6 = 2*3, and apply Strassen at the 2x2 level
        # but with 3x3 entries. This is essentially what we'd get with the block approach.
        
        # Let me just return None to indicate this approach is not ready,
        # and fall back to the 189-rank solution.
        return None
    
    # Try the 152-rank solution first
    result_152 = build_2x2_blocks_3x3_rank19()
    
    if result_152 is not None:
        U, V, W, rank = result_152
    else:
        # Fall back to 189-rank solution
        U, V, W, rank = build_2x2_block_strassen()
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for t in range(rank):
                    s = 0.0
                    for k in range(m):
                        s += U_arr[i*m+k, t] * V_arr[k*p+j, t]
                    total += W_arr[j*n+i, t] * s
                C_computed[i, j] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
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
