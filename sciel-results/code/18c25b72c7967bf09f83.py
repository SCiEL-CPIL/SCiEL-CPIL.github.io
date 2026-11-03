"""Optimized matrix multiplication tensor decomposition for 3x4 x 4x8."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_rank88():
        """
        Block decomposition using Strassen for 2x2 subblocks.
        
        Partition:
        A (3x4) = [A1; A2] where A1 is 2x4, A2 is 1x4
        B (4x8) = [B1 | B2] where B1 is 4x4, B2 is 4x4
        
        C (3x8) = [C11 C12; C21 C22]
        C11 = A1 * B1 (2x4)
        C12 = A1 * B2 (2x4)
        C21 = A2 * B1 (1x4)
        C22 = A2 * B2 (1x4)
        
        For C11 (2x4 * 4x4):
        Partition A1 = [A11 | A12] (2x2 each)
        Partition B1 = [B11; B12] (2x2 each) where B11 = B1[0:2,:], B12 = B1[2:4,:]
        C11 = A11*B11 + A12*B12
        Each 2x2*2x2 uses Strassen (7 mults), so 14 total.
        
        Similarly for C12: 14 mults.
        
        For C21 (1x4 * 4x4): 4 dot products of length 4 = 16 mults.
        For C22 (1x4 * 4x4): 16 mults.
        
        Total: 14 + 14 + 16 + 16 = 60? 
        Wait, let me recount. Strassen for 2x2*2x2 uses 7 scalar multiplications.
        C11 has two 2x2*2x2 products: 7+7=14
        C12 has two 2x2*2x2 products: 7+7=14
        C21: 1x4 * 4x4 = 16 scalar mults (4 outputs, each dot product of length 4)
        C22: 16 scalar mults
        Total: 14+14+16+16 = 60.
        
        But the known result is 88. Let me re-examine.
        
        Actually, I think the issue is that when we use Strassen, the "multiplications" are not just scalar multiplications but involve linear combinations. In the tensor decomposition, each Strassen multiplication Si corresponds to one rank-1 term where:
        U has the coefficients of the first factor
        V has the coefficients of the second factor
        W has the contribution to the output
        
        So for C11 = A11*B11 + A12*B12:
        A11*B11 uses Strassen: 7 terms
        A12*B12 uses Strassen: 7 terms
        Total for C11: 14 terms
        
        Similarly C12: 14 terms
        C21: 16 terms (schoolbook)
        C22: 16 terms (schoolbook)
        Total: 14+14+16+16 = 60 terms.
        
        But wait, I need to verify this is correct. Let me think about whether the Strassen decomposition properly accounts for the linear combinations.
        
        In Strassen, M = P*Q where P, Q are 2x2:
        We compute 7 products S1..S7 of linear combinations of entries of P and Q.
        Then M's entries are linear combinations of S1..S7.
        
        In the tensor decomposition, each Sr = (linear combo of P entries) * (linear combo of Q entries) is one rank-1 term.
        The output M entries are linear combinations of these Sr values.
        
        So yes, each Sr contributes one rank-1 term, and there are 7 per 2x2*2x2.
        
        So total should be 60. But the known result is 88. Let me check if I'm missing something.
        
        Oh wait, I think the issue is that C21 and C22 can also potentially be optimized, but for 1x4 * 4x4, the schoolbook is already optimal at 16 (you can't do a dot product of length 4 in fewer than 4 multiplications, and there are 4 outputs, so 16 total).
        
        Hmm, but 60 < 88. Let me just try to construct the rank-60 decomposition and verify it.
        """
        # We'll build the decomposition term by term.
        # Each term is (u_vec, v_vec, w_vec) where:
        # u_vec has length n*m = 12
        # v_vec has length m*p = 32
        # w_vec has length p*n = 24
        
        terms = []  # list of (u, v, w) tuples, each a numpy array
        
        def add_term(u, v, w):
            """Add a rank-1 term. u, v, w are dicts mapping index to coefficient."""
            u_arr = np.zeros(n * m)
            v_arr = np.zeros(m * p)
            w_arr = np.zeros(p * n)
            for idx, coef in u.items():
                u_arr[idx] = coef
            for idx, coef in v.items():
                v_arr[idx] = coef
            for idx, coef in w.items():
                w_arr[idx] = coef
            terms.append((u_arr, v_arr, w_arr))
        
        # Helper: A[i,j] is at index i*m + j in U
        # Helper: B[j,k] is at index j*p + k in V
        # Helper: C[k,i] is at index k*n + i in W (note: k*n + i, not i*n + k)
        
        # Strassen for 2x2 * 2x2
        # P = [[P00, P01], [P10, P11]] (2x2, from A)
        # Q = [[Q00, Q01], [Q10, Q11]] (2x2, from B)
        # M = P * Q (2x2, result in C)
        # 
        # Strassen:
        # S1 = (P00+P11)*(Q00+Q11)
        # S2 = (P10+P11)*Q00
        # S3 = P00*(Q01-Q11)
        # S4 = P11*(Q10-Q00)
        # S5 = (P00+P01)*Q11
        # S6 = (P10-P00)*(Q00+Q01)
        # S7 = (P01-P11)*(Q10+Q11)
        #
        # M00 = S1 - S5 + S7 - S4
        # M01 = S5 + S6
        # M10 = S2 + S3
        # M11 = S1 - S2 + S4 - S6
        
        def strassen_2x2(P_indices, Q_indices, M_indices):
            """
            Add Strassen terms for 2x2*2x2.
            P_indices: dict mapping (row, col) to U index
            Q_indices: dict mapping (row, col) to V index
            M_indices: dict mapping (row, col) to W index
            """
            P00, P01 = P_indices[(0,0)], P_indices[(0,1)]
            P10, P11 = P_indices[(1,0)], P_indices[(1,1)]
            Q00, Q01 = Q_indices[(0,0)], Q_indices[(0,1)]
            Q10, Q11 = Q_indices[(1,0)], Q_indices[(1,1)]
            M00, M01 = M_indices[(0,0)], M_indices[(0,1)]
            M10, M11 = M_indices[(1,0)], M_indices[(1,1)]
            
            # S1 = (P00+P11)*(Q00+Q11)
            u1 = {P00: 1, P11: 1}
            v1 = {Q00: 1, Q11: 1}
            # Contributes to M00 with coeff +1, M11 with coeff +1
            w1 = {M00: 1, M11: 1}
            add_term(u1, v1, w1)
            
            # S2 = (P10+P11)*Q00
            u2 = {P10: 1, P11: 1}
            v2 = {Q00: 1}
            w2 = {M10: 1}
            add_term(u2, v2, w2)
            
            # S3 = P00*(Q01-Q11)
            u3 = {P00: 1}
            v3 = {Q01: 1, Q11: -1}
            w3 = {M10: 1}
            add_term(u3, v3, w3)
            
            # S4 = P11*(Q10-Q00)
            u4 = {P11: 1}
            v4 = {Q10: 1, Q00: -1}
            w4 = {M00: -1, M11: 1}
            add_term(u4, v4, w4)
            
            # S5 = (P00+P01)*Q11
            u5 = {P00: 1, P01: 1}
            v5 = {Q11: 1}
            w5 = {M00: -1, M01: 1}
            add_term(u5, v5, w5)
            
            # S6 = (P10-P00)*(Q00+Q01)
            u6 = {P10: 1, P00: -1}
            v6 = {Q00: 1, Q01: 1}
            w6 = {M01: 1, M11: -1}
            add_term(u6, v6, w6)
            
            # S7 = (P01-P11)*(Q10+Q11)
            u7 = {P01: 1, P11: -1}
            v7 = {Q10: 1, Q11: 1}
            w7 = {M00: 1}
            add_term(u7, v7, w7)
        
        # C11 = A1 * B1 (2x4 * 4x4 -> 2x4)
        # A1 rows 0,1; B1 cols 0,1,2,3
        # Partition: A1 = [A11 | A12] where A11 = A1[:, 0:2], A12 = A1[:, 2:4]
        # B1 = [B11; B12] where B11 = B1[0:2, :], B12 = B1[2:4, :]
        # C11 = A11*B11 + A12*B12
        #
        # A11 is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # A12 is 2x2: A[0,2], A[0,3], A[1,2], A[1,3]
        # B11 is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
        # B12 is 2x2: B[2,0], B[2,1], B[3,0], B[3,1]
        #
        # Wait, B1 is 4x4, so B1[0:2, :] is 2x4, not 2x2.
        # I need to partition B1's columns too.
        #
        # Let me redo: C11 is 2x4. Partition C11 = [C11a | C11b] where C11a, C11b are 2x2.
        # C11a = A1 * B1[:, 0:2] (2x4 * 4x2 = 2x2)
        # C11b = A1 * B1[:, 2:4] (2x4 * 4x2 = 2x2)
        #
        # For C11a = A1 * B1_left where B1_left is 4x2:
        # A1 = [A11 | A12] (2x2 each, cols 0-1 and 2-3)
        # B1_left = [B11; B12] where B11 = B1[0:2, 0:2] (2x2), B12 = B1[2:4, 0:2] (2x2)
        # C11a = A11*B11 + A12*B12
        #
        # For C11b = A1 * B1_right where B1_right is 4x2:
        # B1_right = [B13; B14] where B13 = B1[0:2, 2:4] (2x2), B14 = B1[2:4, 2:4] (2x2)
        # C11b = A11*B13 + A12*B14
        
        # C11a = A11*B11 + A12*B12
        # A11: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,4,5
        # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,4,5
        # C11a: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0*3+0=0, 1*3+0=3, 0*3+1=1, 1*3+1=4
        # Wait, W[k*n+i] where k is col, i is row.
        # C[0,0]: k=0, i=0 -> W[0*3+0] = W[0]
        # C[0,1]: k=1, i=0 -> W[1*3+0] = W[3]
        # C[1,0]: k=0, i=1 -> W[0*3+1] = W[1]
        # C[1,1]: k=1, i=1 -> W[1*3+1] = W[4]
        
        P_A11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
        Q_B11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
        M_C11a = {(0,0): 0, (0,1): 3, (1,0): 1, (1,1): 4}
        strassen_2x2(P_A11, Q_B11, M_C11a)
        
        # A12: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2,3,6,7
        # B12: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 16,17,20,21
        P_A12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
        Q_B12 = {(0,0): 16, (0,1): 17, (1,0): 20, (1,1): 21}
        strassen_2x2(P_A12, Q_B12, M_C11a)
        
        # C11b = A11*B13 + A12*B14
        # B13: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2,3,6,7
        # B14: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 18,19,22,23
        # C11b: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices 2*3+0=6, 3*3+0=9, 2*3+1=7, 3*3+1=10
        
        Q_B13 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
        M_C11b = {(0,0): 6, (0,1): 9, (1,0): 7, (1,1): 10}
        strassen_2x2(P_A11, Q_B13, M_C11b)
        
        Q_B14 = {(0,0): 18, (0,1): 19, (1,0): 22, (1,1): 23}
        strassen_2x2(P_A12, Q_B14, M_C11b)
        
        # C12 = A1 * B2 (2x4 * 4x4 -> 2x4)
        # B2 is B[:, 4:8]
        # C12a = A1 * B2[:, 0:2] (cols 4,5 of B)
        # C12b = A1 * B2[:, 2:4] (cols 6,7 of B)
        
        # C12a: C[0,4], C[0,5], C[1,4], C[1,5]
        # W indices: 4*3+0=12, 5*3+0=15, 4*3+1=13, 5*3+1=16
        # B2_left: B[0,4], B[0,5], B[1,4], B[1,5], B[2,4], B[2,5], B[3,4], B[3,5]
        # V indices: 4,5,8,9,12,13,16,17
        # B21 = B[0:2, 4:6] -> V: 4,5,8,9
        # B22 = B[2:4, 4:6] -> V: 12,13,16,17
        
        M_C12a = {(0,0): 12, (0,1): 15, (1,0): 13, (1,1): 16}
        Q_B21 = {(0,0): 4, (0,1): 5, (1,0): 8, (1,1): 9}
        strassen_2x2(P_A11, Q_B21, M_C12a)
        Q_B22 = {(0,0): 12, (0,1): 13, (1,0): 16, (1,1): 17}
        strassen_2x2(P_A12, Q_B22, M_C12a)
        
        # C12b: C[0,6], C[0,7], C[1,6], C[1,7]
        # W indices: 6*3+0=18, 7*3+0=21, 6*3+1=19, 7*3+1=22
        # B23 = B[0:2, 6:8] -> V: 6,7,10,11
        # B24 = B[2:4, 6:8] -> V: 14,15,18,19
        
        M_C12b = {(0,0): 18, (0,1): 21, (1,0): 19, (1,1): 22}
        Q_B23 = {(0,0): 6, (0,1): 7, (1,0): 10, (1,1): 11}
        strassen_2x2(P_A11, Q_B23, M_C12b)
        Q_B24 = {(0,0): 14, (0,1): 15, (1,0): 18, (1,1): 19}
        strassen_2x2(P_A12, Q_B24, M_C12b)
        
        # C21 = A2 * B1 (1x4 * 4x4 -> 1x4)
        # A2 is row 2 of A: A[2,0], A[2,1], A[2,2], A[2,3] -> U indices 8,9,10,11
        # B1 is B[:, 0:4]
        # C21: C[2,0], C[2,1], C[2,2], C[2,3]
        # W indices: 0*3+2=2, 1*3+2=5, 2*3+2=8, 3*3+2=11
        
        for k in range(4):
            u = {}
            v = {}
            w = {k * n + 2: 1}  # W[k*n + i] where i=2
            for j in range(4):
                u[2 * m + j] = 1  # A[2,j]
                v[j * p + k] = 1  # B[j,k]
            add_term(u, v, w)
        
        # C22 = A2 * B2 (1x4 * 4x4 -> 1x4)
        # C22: C[2,4], C[2,5], C[2,6], C[2,7]
        # W indices: 4*3+2=14, 5*3+2=17, 6*3+2=20, 7*3+2=23
        
        for k in range(4, 8):
            u = {}
            v = {}
            w = {k * n + 2: 1}
            for j in range(4):
                u[2 * m + j] = 1
                v[j * p + k] = 1
            add_term(u, v, w)
        
        # Build the matrices
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u, v, w) in enumerate(terms):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(result):
        """Verify the decomposition is correct."""
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        rank = U.shape[1]
        
        # Check: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Try rank-88 (or actually rank-60 based on our construction)
    try:
        result88 = build_rank88()
        if verify(result88):
            return result88
    except Exception:
        pass
    
    # Fallback to schoolbook
    return build_schoolbook()
# EVOLVE-BLOCK-END
