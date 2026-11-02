import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Base decomposition for 2x2x2 (Strassen, rank 7)
    # Indices: A(i,j), B(j,k), C(k,i)
    # i,j,k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    strassen_222 = [
        # (u_indices, v_indices, w_indices)
        # M1 = (A11+A22)(B11+B22)
        ([ (0,0), (1,1) ], [ (0,0), (1,1) ], [ (0,0), (1,1) ]), # Coeff 1
        # M2 = (A21+A22)B11
        ([ (1,0), (1,1) ], [ (0,0) ], [ (1,0), (1,1) ]), # Coeff 1
        # M3 = A11(B12-B22)
        ([ (0,0) ], [ (0,1), (1,1) ], [ (1,0), (1,1) ]), # Coeff 1, -1
        # M4 = A22(B21-B11)
        ([ (1,1) ], [ (1,0), (0,0) ], [ (0,1), (0,0) ]), # Coeff 1, -1
        # M5 = (A11+A12)B22
        ([ (0,0), (0,1) ], [ (1,1) ], [ (1,1), (0,1) ]), # Coeff 1
        # M6 = (A21-A11)(B11+B12)
        ([ (1,0), (0,0) ], [ (0,0), (0,1) ], [ (0,0), (0,1) ]), # Coeff 1, -1
        # M7 = (A12-A22)(B21+B22)
        ([ (0,1), (1,1) ], [ (1,0), (1,1) ], [ (1,0), (1,1) ]), # Coeff 1, -1
    ]
    
    # This representation is tricky to handle with coefficients.
    # Let's just generate the U, V, W matrices directly for the base case.
    
    def get_base_222():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = 1
        
        # M3
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[2,2] = 1; W[3,2] = 1
        
        # M4
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[1,3] = 1; W[0,3] = -1
        
        # M5
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[3,4] = 1; W[1,4] = 1
        
        # M6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[0,5] = 1; W[1,5] = 1
        
        # M7
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[2,6] = 1; W[3,6] = 1
        
        return U, V, W

    U_base, V_base, W_base = get_base_222()
    
    # Function to compose decompositions
    # T1: (n1, m1, p1) rank r1
    # T2: (m1, p1, n1) ... wait.
    # If we have a block decomposition of size (Nb, Mb, Pb) with rank Rb,
    # and each block multiplication is (nb, mb, pb) with rank Rb2,
    # then total rank is Rb * Rb2.
    
    # For 4x4x4:
    # Block size 2x2x2. Nb=2, Mb=2, Pb=2.
    # Block decomposition is Strassen 2x2x2 (Rank 7).
    # Inner decomposition is Strassen 2x2x2 (Rank 7).
    
    def compose(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2):
        # n1 x m1 x p1 is the "outer" structure
        # n2 x m2 x p2 is the "inner" structure
        # Dimensions must match: m1 == n2, p1 == n2, n1 == p2?
        # No.
        # Outer: A (n1 x m1), B (m1 x p1), C (p1 x n1)
        # Inner: A_block (n2 x m2), B_block (m2 x p2), C_block (p2 x n2)
        # We require m1 == n2? No.
        # The blocks of A (n1 x m1) are of size n2 x m2. So n1 = n2 * N1, m1 = m2 * M1.
        # The blocks of B (m1 x p1) are of size m2 x p2. So m1 = m2 * M1, p1 = p2 * P1.
        # The blocks of C (p1 x n1) are of size p2 x n2. So p1 = p2 * P1, n1 = n2 * N1.
        
        # Total rank = Rank(Outer) * Rank(Inner)
        rank_out = U1.shape[1]
        rank_in = U2.shape[1]
        total_rank = rank_out * rank_in
        
        U_total = np.zeros((n1 * m1, total_rank))
        V_total = np.zeros((m1 * p1, total_rank))
        W_total = np.zeros((p1 * n1, total_rank))
        
        for k_out in range(rank_out):
            u_out = U1[:, k_out]
            v_out = V1[:, k_out]
            w_out = W1[:, k_out]
            
            for k_in in range(rank_in):
                col = k_out * rank_in + k_in
                u_in = U2[:, k_in]
                v_in = V2[:, k_in]
                w_in = W2[:, k_in]
                
                # Expand u_out (size n1*m1) using u_in (size n2*m2)
                # u_out[i*m1+j] is the coefficient for block (i,j)
                # u_in[a*m2+b] is the coefficient for element (a,b) in the block
                # Total index: (i*n2+a)*m1 + (j*m2+b)
                
                for i in range(n1 // n2):
                    for j in range(m1 // m2):
                        coeff = u_out[i * (m1//m2) + j]
                        if coeff == 0: continue
                        for a in range(n2):
                            for b in range(m2):
                                inner_coeff = u_in[a * m2 + b]
                                if inner_coeff == 0: continue
                                idx = (i * n2 + a) * m1 + (j * m2 + b)
                                U_total[idx, col] += coeff * inner_coeff
                                
                for j in range(m1 // m2):
                    for k in range(p1 // p2):
                        coeff = v_out[j * (p1//p2) + k]
                        if coeff == 0: continue
                        for b in range(m2):
                            for c in range(p2):
                                inner_coeff = v_in[b * p2 + c]
                                if inner_coeff == 0: continue
                                idx = (j * m2 + b) * p1 + (k * p2 + c)
                                V_total[idx, col] += coeff * inner_coeff
                                
                for k in range(p1 // p2):
                    for i in range(n1 // n2):
                        coeff = w_out[k * (n1//n2) + i]
                        if coeff == 0: continue
                        for c in range(p2):
                            for a in range(n2):
                                inner_coeff = w_in[c * n2 + a]
                                if inner_coeff == 0: continue
                                idx = (k * p2 + c) * n1 + (i * n2 + a)
                                W_total[idx, col] += coeff * inner_coeff
                                
        return U_total, V_total, W_total

    # 1. Get 4x4x4 decomposition
    # Outer: 2x2x2 (Strassen)
    # Inner: 2x2x2 (Strassen)
    U_444, V_444, W_444 = compose(2, 2, 2, U_base, V_base, W_base, 2, 2, 2, U_base, V_base, W_base)
    
    # 2. Get 4x4x2 decomposition
    # Outer: 2x2x1 (Rank 4)
    # Inner: 2x2x2 (Strassen)
    
    # Base 2x2x1 decomposition
    # A (2x2), B (2x1), C (1x2)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    
    U_221[0,0] = 1; V_221[0,0] = 1; W_221[0,0] = 1
    U_221[1,1] = 1; V_221[1,1] = 1; W_221[0,1] = 1
    U_221[2,2] = 1; V_221[0,2] = 1; W_221[1,2] = 1
    U_221[3,3] = 1; V_221[1,3] = 1; W_221[1,3] = 1
    
    U_442, V_442, W_442 = compose(2, 2, 1, U_221, V_221, W_221, 2, 2, 2, U_base, V_base, W_base)
    
    # 3. Combine 4x4x4 and 4x4x2 to get 4x4x6
    # A is 4x4. B is 4x6 = [4x4 | 4x2].
    # C is 4x6 = [4x4 | 4x2].
    
    U_final = np.hstack([U_444, U_442])
    V_final = np.vstack([V_444, V_442]) # V_444 is 4x4, V_442 is 4x2. Stacked vertically?
    # V shape for 4x4x4 is (4*4, rank) = (16, 49)
    # V shape for 4x4x2 is (4*2, rank) = (8, 28)
    # We need V_final to be (4*6, rank_total) = (24, 77)
    # V_444 corresponds to B_left (4x4). Rows 0-15.
    # V_442 corresponds to B_right (4x2). Rows 16-23.
    
    V_final = np.zeros((24, 77))
    V_final[:16, :49] = V_444
    V_final[16:, 49:] = V_442
    
    W_final = np.hstack([W_444, W_442])
    # W_444 is (4*4, 49) = (16, 49)
    # W_442 is (2*4, 28) = (8, 28)
    # W_final should be (6*4, 77) = (24, 77)
    # W_444 corresponds to C_left (4x4). Rows 0-15.
    # W_442 corresponds to C_right (2x4). Rows 16-23.
    
    W_final = np.zeros((24, 77))
    W_final[:16, :49] = W_444
    W_final[16:, 49:] = W_442
    
    # Convert to lists of lists for output
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
