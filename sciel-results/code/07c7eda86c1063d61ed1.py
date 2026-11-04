import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Decomposition for 2x2x2 (Rank 7)
    def get_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A10+A11)B00 -> C10
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3: A00(B01-B11) -> C01
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5: (A00+A01)B11 -> C00 (-), C01 (+)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00 (+), C11 (-)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    # 2. Decomposition for 2x2x1 (Rank 4)
    def get_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4)) # m=2, p=1 -> 2*1=2
        W = np.zeros((2, 4)) # p=1, n=2 -> 1*2=2
        
        # A00 B00 -> C00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # A01 B00 -> C10
        U[1, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        # A10 B10 -> C00
        U[2, 2] = 1; V[1, 2] = 1; W[0, 2] = 1
        # A11 B10 -> C10
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # 3. Decomposition for 2x2x3 (Rank 11)
    def get_2x2x3():
        U1, V1, W1 = get_2x2x2()
        U2, V2, W2 = get_2x2x1()
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11)) # m=2, p=3 -> 6
        W = np.zeros((6, 11)) # p=3, n=2 -> 6
        
        # Part 1: 2x2x2 for cols 0,1
        U[:, :7] = U1
        V[:4, :7] = V1 # V1 is 4x7 (j*2+k, j in 0..1, k in 0..1)
        # Map V1 indices (j*2+k) to V2 indices (j*3+k)
        # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 2; j=1, k=1 -> 3
        # V2 index: j*3+k
        # 0->0, 1->1, 2->3, 3->4
        V[0, :7] = V1[0]
        V[1, :7] = V1[1]
        V[3, :7] = V1[2]
        V[4, :7] = V1[3]
        
        # Map W1 indices (k*2+i) to W2 indices (k*2+i)
        # W1 is 4x7. W2 is 6x11.
        # W1 k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3
        # W2 index: k*2+i
        # Same mapping.
        W[:4, :7] = W1
        
        # Part 2: 2x2x1 for col 2
        U[:, 7:] = U2
        # V2 is 2x4 (j*1+k, j in 0..1, k=0)
        # Map to V indices (j*3+2)
        # j=0 -> 2; j=1 -> 5
        V[2, 7:] = V2[0]
        V[5, 7:] = V2[1]
        
        # W2 is 2x4 (k*2+i, k=0, i in 0..1)
        # Map to W indices (2*2+i) = 4+i
        # i=0 -> 4; i=1 -> 5
        W[4, 7:] = W2[0]
        W[5, 7:] = W2[1]
        
        return U, V, W

    # 4. Decomposition for 4x4x3 (Rank 44)
    def get_4x4x3():
        U, V, W = get_2x2x3()
        rank = 11
        
        U_total = np.zeros((16, 44))
        V_total = np.zeros((12, 44)) # m=4, p=3 -> 12
        W_total = np.zeros((12, 44)) # p=3, n=4 -> 12
        
        # 4 blocks
        # Block 0: A00 B_top -> C_top
        # A: i*4+j (i,j in 0..1)
        # B: j*3+k (j in 0..1, k in 0..2)
        # C: k*4+i (k in 0..2, i in 0..1)
        
        # Block 1: A01 B_bottom -> C_top
        # A: i*4+(j+2)
        # B: (j+2)*3+k
        # C: k*4+i
        
        # Block 2: A10 B_top -> C_bottom
        # A: (i+2)*4+j
        # B: j*3+k
        # C: k*4+(i+2)
        
        # Block 3: A11 B_bottom -> C_bottom
        # A: (i+2)*4+(j+2)
        # B: (j+2)*3+k
        # C: k*4+(i+2)
        
        for b in range(4):
            U_block = np.zeros((16, 11))
            V_block = np.zeros((12, 11))
            W_block = np.zeros((12, 11))
            
            for i in range(2):
                for j in range(2):
                    ui = i * 2 + j
                    U_global = (i + (b//2)*2) * 4 + (j + (b%2)*2)
                    U_block[U_global, :] = U[ui, :]
                    
            for j in range(2):
                for k in range(3):
                    vj = j * 3 + k
                    V_global = (j + (b%2)*2) * 3 + k
                    V_block[V_global, :] = V[vj, :]
                    
            for k in range(3):
                for i in range(2):
                    wk = k * 2 + i
                    W_global = k * 4 + (i + (b//2)*2)
                    W_block[W_global, :] = W[wk, :]
            
            U_total[:, b*11:(b+1)*11] = U_block
            V_total[:, b*11:(b+1)*11] = V_block
            W_total[:, b*11:(b+1)*11] = W_block
            
        return U_total, V_total, W_total

    # 5. Decomposition for 4x4x4 (Rank 49)
    def get_4x4x4():
        U, V, W = get_2x2x2()
        rank = 7
        
        U_total = np.zeros((16, 49))
        V_total = np.zeros((16, 49)) # m=4, p=4 -> 16
        W_total = np.zeros((16, 49)) # p=4, n=4 -> 16
        
        # Strassen 7 products
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # We need to map each M_r to a 2x2x2 decomposition.
        # Each M_r is a 2x2 matrix product.
        # A' is a 2x2 matrix, B' is a 2x2 matrix.
        # A'_{ij} = sum_{blocks} u_{block, r} A_{block}
        # B'_{jk} = sum_{blocks} v_{block, r} B_{block}
        
        # For each r, we have a 2x2x2 multiplication.
        # We use the rank-7 decomposition U, V, W.
        
        # Define the linear combinations
        # u_coeffs[r][block] where block is 0:00, 1:01, 2:10, 3:11
        u_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [0, 0, 1, -1], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        v_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3
            [0, 0, 1, -1], # M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # w_coeffs[r][out_block] where out_block is 0:00, 1:01, 2:10, 3:11
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        w_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 0], # M2
            [0, 1, 0, 0], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0], # M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, -1]  # M7
        ]
        
        for r in range(7):
            U_r = np.zeros((16, 7))
            V_r = np.zeros((16, 7))
            W_r = np.zeros((16, 7))
            
            # A' construction
            for i in range(2):
                for j in range(2):
                    ui = i * 2 + j
                    # A'_{ij} = sum_b u_coeffs[r][b] A_{b, ij}
                    # A_{b, ij} is at index (i_b * 2 + i) * 4 + (j_b * 2 + j)
                    # where b = i_b * 2 + j_b
                    for b in range(4):
                        i_b = b // 2
                        j_b = b % 2
                        A_idx = (i_b * 2 + i) * 4 + (j_b * 2 + j)
                        U_r[A_idx, :] = u_coeffs[r][b] * U[ui, :]
                        
            # B' construction
            for j in range(2):
                for k in range(2):
                    vj = j * 2 + k
                    # B'_{jk} = sum_b v_coeffs[r][b] B_{b, jk}
                    for b in range(4):
                        i_b = b // 2
                        j_b = b % 2
                        B_idx = (i_b * 2 + j) * 4 + (j_b * 2 + k) # Wait, B is m x p = 4x4.
                        # B index is j_global * 4 + k_global
                        # j_global = i_b * 2 + j? No.
                        # B is 4x4. Rows 0..3, Cols 0..3.
                        # Block b is rows i_b*2..i_b*2+1, cols j_b*2..j_b*2+1.
                        # So row = i_b * 2 + j, col = j_b * 2 + k.
                        B_idx = (i_b * 2 + j) * 4 + (j_b * 2 + k)
                        V_r[B_idx, :] = v_coeffs[r][b] * V[vj, :]
                        
            # C' construction
            for k in range(2):
                for i in range(2):
                    wk = k * 2 + i
                    # C'_{ki} = sum_b w_coeffs[r][b] C_{b, ki}
                    for b in range(4):
                        i_b = b // 2
                        j_b = b % 2
                        C_idx = (j_b * 2 + k) * 4 + (i_b * 2 + i) # C is n x p = 4x4.
                        # C index is k_global * 4 + i_global
                        # k_global = j_b * 2 + k, i_global = i_b * 2 + i
                        W_r[C_idx, :] = w_coeffs[r][b] * W[wk, :]
                        
            U_total[:, r*7:(r+1)*7] = U_r
            V_total[:, r*7:(r+1)*7] = V_r
            W_total[:, r*7:(r+1)*7] = W_r
            
        return U_total, V_total, W_total

    # Combine 4x4x4 and 4x4x3
    U1, V1, W1 = get_4x4x4()
    U2, V2, W2 = get_4x4x3()
    
    U = np.zeros((16, 93))
    V = np.zeros((28, 93)) # m=4, p=7 -> 28
    W = np.zeros((28, 93)) # p=7, n=4 -> 28
    
    U[:, :49] = U1
    U[:, 49:] = U2
    
    # V1 is 16x49 (4x4). V2 is 12x44 (4x3).
    # V1 indices: j*4+k (j,k in 0..3)
    # V2 indices: j*3+k (j in 0..3, k in 0..2)
    # V total indices: j*7+k (j in 0..3, k in 0..6)
    
    for j in range(4):
        for k in range(4):
            V[j*7+k, :49] = V1[j*4+k, :]
            
    for j in range(4):
        for k in range(3):
            V[j*7+(k+4), 49:] = V2[j*3+k, :]
            
    # W1 is 16x49 (4x4). W2 is 12x44 (4x3).
    # W1 indices: k*4+i (k,i in 0..3)
    # W2 indices: k*4+i (k in 0..2, i in 0..3)
    # W total indices: k*4+i (k in 0..6, i in 0..3)
    
    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W1[k*4+i, :]
            
    for k in range(3):
        for i in range(4):
            W[(k+4)*4+i, 49:] = W2[k*4+i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
