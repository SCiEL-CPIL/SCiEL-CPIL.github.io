import time

def construct():
    """
    Matrix multiplication 2x5 by 5x5 using Strassen for 2x2 blocks.
    
    Partition A (2x5) rows and B (5x5) rows/cols by index groups:
    G1={0,1}, G2={2,3}, G3={4}
    
    C[:,{0,1}] = A[:,G1]*B[G1,{0,1}] + A[:,G2]*B[G2,{0,1}] + A[:,G3]*B[G3,{0,1}]
    C[:,{2,3}] = A[:,G1]*B[G1,{2,3}] + A[:,G2]*B[G2,{2,3}] + A[:,G3]*B[G3,{2,3}]
    C[:,{4}]   = A[:,G1]*B[G1,{4}]   + A[:,G2]*B[G2,{4}]   + A[:,G3]*B[G3,{4}]
    
    Products:
    - 2x2 * 2x2 (4 of these, rank 7 each via Strassen) = 28
    - 2x1 * 2x2 (2 of these, rank 4 each) = 8
    - 2x2 * 2x1 (2 of these, rank 4 each) = 8
    - 2x1 * 2x1 (1 of these, rank 2) = 2
    Total rank = 46
    """
    rank = 46
    
    U = [[0] * rank for _ in range(2 * 5)]  # 10 x 46
    V = [[0] * rank for _ in range(5 * 5)]  # 25 x 46
    W = [[0] * rank for _ in range(5 * 2)]  # 10 x 46 (output: p*n = 5*2 = 10)
    
    col = 0
    
    def add_strassen(A_r0c0, A_r0c1, A_r1c0, A_r1c1, B_r0c0, B_r0c1, B_r1c0, B_r1c1, 
                     C_r0c0, C_r0c1, C_r1c0, C_r1c1):
        """Add Strassen rank-7 decomposition for 2x2*2x2.
        A_r?c? are U row indices for A[i,j], B_r?c? are V row indices for B[j,k],
        C_r?c? are W row indices for C output [k*n+i, r] i.e. W[k*2+i, r].
        """
        nonlocal col
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # M1: (A00+A11)(B00+B11)
        U[A_r0c0][col] += 1
        U[A_r1c1][col] += 1
        V[B_r0c0][col] += 1
        V[B_r1c1][col] += 1
        W[C_r0c0][col] += 1   # C[0,0]: +M1
        W[C_r0c1][col] += 0   # C[0,1]: M3+M5, no M1
        W[C_r1c0][col] += 0   # C[1,0]: M2+M4, no M1
        W[C_r1c1][col] += 1   # C[1,1]: +M1
        col += 1
        
        # M2: (A10+A11)B00
        U[A_r1c0][col] += 1
        U[A_r1c1][col] += 1
        V[B_r0c0][col] += 1
        W[C_r0c0][col] += 0
        W[C_r0c1][col] += 0
        W[C_r1c0][col] += 1   # C[1,0]: +M2
        W[C_r1c1][col] += -1  # C[1,1]: -M2
        col += 1
        
        # M3: A00(B01-B11)
        U[A_r0c0][col] += 1
        V[B_r0c1][col] += 1
        V[B_r1c1][col] += -1
        W[C_r0c0][col] += 0
        W[C_r0c1][col] += 1   # C[0,1]: +M3
        W[C_r1c0][col] += 0
        W[C_r1c1][col] += 1   # C[1,1]: +M3
        col += 1
        
        # M4: A11(B10-B00)
        U[A_r1c1][col] += 1
        V[B_r1c0][col] += 1
        V[B_r0c0][col] += -1
        W[C_r0c0][col] += 1   # C[0,0]: +M4
        W[C_r0c1][col] += 0
        W[C_r1c0][col] += 1   # C[1,0]: +M4
        W[C_r1c1][col] += 0
        col += 1
        
        # M5: (A00+A01)B11
        U[A_r0c0][col] += 1
        U[A_r0c1][col] += 1
        V[B_r1c1][col] += 1
        W[C_r0c0][col] += -1  # C[0,0]: -M5
        W[C_r0c1][col] += 1   # C[0,1]: +M5
        W[C_r1c0][col] += 0
        W[C_r1c1][col] += 0
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[A_r1c0][col] += 1
        U[A_r0c0][col] += -1
        V[B_r0c0][col] += 1
        V[B_r0c1][col] += 1
        W[C_r0c0][col] += 0
        W[C_r0c1][col] += 0
        W[C_r1c0][col] += 0
        W[C_r1c1][col] += 1   # C[1,1]: +M6
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[A_r0c1][col] += 1
        U[A_r1c1][col] += -1
        V[B_r1c0][col] += 1
        V[B_r1c1][col] += 1
        W[C_r0c0][col] += 1   # C[0,0]: +M7
        W[C_r0c1][col] += 0
        W[C_r1c0][col] += 0
        W[C_r1c1][col] += 0
        col += 1
    
    def add_rank4_2x1_2x2(A_rows, B_r0c0, B_r0c1, B_r1c0, B_r1c1, 
                          C_r0c0, C_r0c1, C_r1c0, C_r1c1):
        """Add rank-4 decomposition for (2x1)*(2x2) -> 2x2.
        C[i,k] = A[i]*B[k] (since A is 2x1, B is 2x2... wait)
        Actually A is 2x1, B should be 1x2 for this to work dimensionally.
        Hmm, let me reconsider.
        """
        pass
    
    # Let me reconsider the dimensions more carefully.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For G={j0,j1}: A_iG = [A[i,j0], A[i,j1]] (2x2), B_Gk = [B[j0,k], B[j1,k]] 
    # No wait, B[j,k] where j in {j0,j1}, k fixed -> that's a 2x1 vector for each k.
    # C[:, {k0,k1}] from group G={j0,j1}: 
    #   C[i,k0] = A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
    #   C[i,k1] = A[i,j0]*B[j0,k1] + A[i,j1]*B[j1,k1]
    # This is (2x2) * (2x2) where A_part is 2x2 and B_part is 2x2.
    
    # For G={j0} (singleton): 
    #   C[i,k0] = A[i,j0]*B[j0,k0]
    #   C[i,k1] = A[i,j0]*B[j0,k1]
    # This is (2x1) * (1x2) -> 2x2, rank 4 (schoolbook: 2*1*2=4)
    
    # For output single column k:
    # From G={j0,j1}: C[i,k] = A[i,j0]*B[j0,k] + A[i,j1]*B[j1,k]
    # This is (2x2)*(2x1) -> 2x1, rank 4 (schoolbook: 2*2*1=4)
    # From G={j0}: C[i,k] = A[i,j0]*B[j0,k], rank 2 (2*1*1=2)
    
    # So the products are:
    # Output cols {0,1}: A[:,{0,1}](2x2)*B[{0,1},{0,1}](2x2) + A[:,{2,3}](2x2)*B[{2,3},{0,1}](2x2) + A[:,{4}](2x1)*B[{4},{0,1}](1x2)
    #   Wait, B[{4},{0,1}] is B[4,0] and B[4,1], which is a 1x2 matrix.
    #   A[:,{4}] is A[0,4] and A[1,4], a 2x1 matrix.
    #   Product is (2x1)*(1x2) = 2x2, rank 4.
    
    # Output cols {2,3}: similar
    # Output col {4}: A[:,{0,1}](2x2)*B[{0,1},{4}](2x1) + A[:,{2,3}](2x2)*B[{2,3},{4}](2x1) + A[:,{4}](2x1)*B[{4},{4}](1x1)
    #   (2x2)*(2x1) = 2x1, rank 4 each
    #   (2x1)*(1x1) = 2x1, rank 2
    
    # Total: 4*(2x2*2x2, rank7) + 2*(2x1*1x2, rank4) + 2*(2x2*2x1, rank4) + 1*(2x1*1x1, rank2)
    #       = 28 + 8 + 8 + 2 = 46
    
    # Let me redo with correct index conventions.
    # U has 10 rows: A[i,j] at row i*5+j (i in 0,1; j in 0..4)
    # V has 25 rows: B[j,k] at row j*5+k (j in 0..4; k in 0..4)
    # W has 10 rows: C output at W[k*2+i] (k in 0..4; i in 0,1) -- trace convention W[k*n+i,r]
    
    col = 0
    
    # Helper for 2x2*2x2 using Strassen (rank 7)
    def strassen_2x2(A_r0c0, A_r0c1, A_r1c0, A_r1c1, B_r0c0, B_r0c1, B_r1c0, B_r1c1,
                     C_r0c0, C_r0c1, C_r1c0, C_r1c1):
        nonlocal col
        # A = [[A_r0c0, A_r0c1], [A_r1c0, A_r1c1]] (rows i=0,1; cols j=0,1 within group)
        # B = [[B_r0c0, B_r0c1], [B_r1c0, B_r1c1]] (rows j=0,1; cols k=0,1 within group)
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00(+), C11(+)
        U[A_r0c0][col] += 1; U[A_r1c1][col] += 1
        V[B_r0c0][col] += 1; V[B_r1c1][col] += 1
        W[C_r0c0][col] += 1; W[C_r1c1][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10(+), C11(-)
        U[A_r1c0][col] += 1; U[A_r1c1][col] += 1
        V[B_r0c0][col] += 1
        W[C_r1c0][col] += 1; W[C_r1c1][col] += -1
        col += 1
        
        # M3 = A00(B01-B11) -> C01(+), C11(+)
        U[A_r0c0][col] += 1
        V[B_r0c1][col] += 1; V[B_r1c1][col] += -1
        W[C_r0c1][col] += 1; W[C_r1c1][col] += 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00(+), C10(+)
        U[A_r1c1][col] += 1
        V[B_r1c0][col] += 1; V[B_r0c0][col] += -1
        W[C_r0c0][col] += 1; W[C_r1c0][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11 -> C00(-), C01(+)
        U[A_r0c0][col] += 1; U[A_r0c1][col] += 1
        V[B_r1c1][col] += 1
        W[C_r0c0][col] += -1; W[C_r0c1][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11(+)
        U[A_r1c0][col] += 1; U[A_r0c0][col] += -1
        V[B_r0c0][col] += 1; V[B_r0c1][col] += 1
        W[C_r1c1][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00(+)
        U[A_r0c1][col] += 1; U[A_r1c1][col] += -1
        V[B_r1c0][col] += 1; V[B_r1c1][col] += 1
        W[C_r0c0][col] += 1
        col += 1
    
    # Helper for (2x1)*(1x2) -> 2x2, rank 4 schoolbook
    def rank4_2x1_1x2(A_r0, A_r1, B_c0, B_c1, C_r0c0, C_r0c1, C_r1c0, C_r1c1):
        """A is 2x1: [A_r0; A_r1]. B is 1x2: [B_c0, B_c1]. C[i,k]=A[i]*B[k]"""
        nonlocal col
        # C00 = A0*B0
        U[A_r0][col] += 1; V[B_c0][col] += 1; W[C_r0c0][col] += 1; col += 1
        # C01 = A0*B1
        U[A_r0][col] += 1; V[B_c1][col] += 1; W[C_r0c1][col] += 1; col += 1
        # C10 = A1*B0
        U[A_r1][col] += 1; V[B_c0][col] += 1; W[C_r1c0][col] += 1; col += 1
        # C11 = A1*B1
        U[A_r1][col] += 1; V[B_c1][col] += 1; W[C_r1c1][col] += 1; col += 1
    
    # Helper for (2x2)*(2x1) -> 2x1, rank 4 schoolbook
    def rank4_2x2_2x1(A_r0c0, A_r0c1, A_r1c0, A_r1c1, B_r0, B_r1, C_r0, C_r1):
        """A is 2x2, B is 2x1: [B_r0; B_r1]. C[i] = A[i,0]*B[0] + A[i,1]*B[1]"""
        nonlocal col
        # C0 = A00*B0 + A01*B1
        U[A_r0c0][col] += 1; V[B_r0][col] += 1; W[C_r0][col] += 1; col += 1
        U[A_r0c1][col] += 1; V[B_r1][col] += 1; W[C_r0][col] += 1; col += 1
        # C1 = A10*B0 + A11*B1
        U[A_r1c0][col] += 1; V[B_r0][col] += 1; W[C_r1][col] += 1; col += 1
        U[A_r1c1][col] += 1; V[B_r1][col] += 1; W[C_r1][col] += 1; col += 1
    
    # Helper for (2x1)*(1x1) -> 2x1, rank 2 schoolbook
    def rank2_2x1_1x1(A_r0, A_r1, B_val, C_r0, C_r1):
        nonlocal col
        U[A_r0][col] += 1; V[B_val][col] += 1; W[C_r0][col] += 1; col += 1
        U[A_r1][col] += 1; V[B_val][col] += 1; W[C_r1][col] += 1; col += 1
    
    # Index helpers
    def u_idx(i, j): return i * 5 + j  # A[i,j], i in {0,1}, j in {0..4}
    def v_idx(j, k): return j * 5 + k  # B[j,k], j in {0..4}, k in {0..4}
    def w_idx(i, k): return k * 2 + i  # C[i,k] stored as W[k*2+i], k in {0..4}, i in {0,1}
    
    # Group 1: output columns {0,1}
    # Term 1: A[:,{0,1}] (2x2) * B[{0,1},{0,1}] (2x2)
    strassen_2x2(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),  # A[0,0],A[0,1],A[1,0],A[1,1]
        v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),  # B[0,0],B[0,1],B[1,0],B[1,1]
        w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)   # C[0,0],C[0,1],C[1,0],C[1,1]
    )
    
    # Term 2: A[:,{2,3}] (2x2) * B[{2,3},{0,1}] (2x2)
    strassen_2x2(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1),
        w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)
    )
    
    # Term 3: A[:,{4}] (2x1) * B[{4},{0,1}] (1x2)
    rank4_2x1_1x2(
        u_idx(0,4), u_idx(1,4),
        v_idx(4,0), v_idx(4,1),
        w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)
    )
    
    # Group 2: output columns {2,3}
    # Term 1: A[:,{0,1}] * B[{0,1},{2,3}]
    strassen_2x2(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
        w_idx(0,2), w_idx(1,2), w_idx(0,3), w_idx(1,3)
    )
    
    # Term 2: A[:,{2,3}] * B[{2,3},{2,3}]
    strassen_2x2(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3),
        w_idx(0,2), w_idx(1,2), w_idx(0,3), w_idx(1,3)
    )
    
    # Term 3: A[:,{4}] (2x1) * B[{4},{2,3}] (1x2)
    rank4_2x1_1x2(
        u_idx(0,4), u_idx(1,4),
        v_idx(4,2), v_idx(4,3),
        w_idx(0,2), w_idx(1,2), w_idx(0,3), w_idx(1,3)
    )
    
    # Group 3: output column {4}
    # Term 1: A[:,{0,1}] (2x2) * B[{0,1},{4}] (2x1)
    rank4_2x2_2x1(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,4), v_idx(1,4),
        w_idx(0,4), w_idx(1,4)
    )
    
    # Term 2: A[:,{2,3}] (2x2) * B[{2,3},{4}] (2x1)
    rank4_2x2_2x1(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,4), v_idx(3,4),
        w_idx(0,4), w_idx(1,4)
    )
    
    # Term 3: A[:,{4}] (2x1) * B[{4},{4}] (1x1)
    rank2_2x1_1x1(
        u_idx(0,4), u_idx(1,4),
        v_idx(4,4),
        w_idx(0,4), w_idx(1,4)
    )
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

