"""Program for 4x4 * 4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    
    # Use block Strassen: split A (4x4) into 2x2 blocks of 2x2
    # Split B (4x6) into 2x3 blocks of 2x2
    # C (4x6) has 2x3 blocks of 2x2
    #
    # For 2x2 * 2x2 Strassen rank-7 scheme:
    # Given matrices X=(x11,x12;x21,x22) and Y=(y11,y12;y21,y22)
    # Z = X*Y where Z = (z11,z12;z21,z22)
    #
    # Strassen products:
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    #
    # z11 = M1 - M4 + M6
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M3 + M7
    
    # For our problem:
    # A is 4x4, B is 4x6, C = A*B is 4x6
    # Block A: A[i,j] where i,j in {0,1}, each block is 2x2
    # Block B: B[i,j] where i in {0,1}, j in {0,1,2}, each block is 2x2
    # Block C: C[i,j] where i in {0,1}, j in {0,1,2}, each block is 2x2
    #
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    #
    # We need to compute 6 block products (i in {0,1}, j in {0,1,2}),
    # each is sum of two 2x2*2x2 products.
    # Each 2x2*2x2 can be done with rank 7 (Strassen).
    # So naive would be 6 * 2 * 7 = 84.
    #
    # But we can share: for each i, we compute
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]  
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    #
    # For fixed i, let X0 = A[i,0], X1 = A[i,1]
    # And Y0 = B[0,0], Y1 = B[1,0], etc.
    #
    # Actually, let's think of it as: for each column j of blocks,
    # we compute C[:,j] = A[:,:]*B[:,j] which is 4x2 * 4x2 -> 4x2... 
    # No wait, B is 4x6, so B[:,j] is 4x2 for j in {0,1,2}.
    #
    # Let me use a different approach. Let's treat this as:
    # For each of the 6 output 2x2 blocks, we need a bilinear computation.
    # But we can exploit that the same A blocks multiply different B blocks.
    #
    # Let's define the 16 A entries grouped: A00 (top-left 2x2), A01 (top-right 2x2),
    # A10 (bottom-left 2x2), A11 (bottom-right 2x2)
    # And B blocks: B0 (top-left 2x2 of 4x6, so rows 0-1, cols 0-1),
    # B1 (top-middle, rows 0-1, cols 2-3), B2 (top-right, rows 0-1, cols 4-5),
    # B3 (bottom-left, rows 2-3, cols 0-1), B4 (bottom-middle, rows 2-3, cols 2-3),
    # B5 (bottom-right, rows 2-3, cols 4-5)
    #
    # C0 (top-left 2x2 of 4x6) = A00*B0 + A10*B3
    # C1 (top-middle) = A00*B1 + A10*B4
    # C2 (top-right) = A00*B2 + A10*B5
    # C3 (bottom-left) = A01*B0 + A11*B3
    # C4 (bottom-middle) = A01*B1 + A11*B4
    # C5 (bottom-right) = A01*B2 + A11*B5
    
    # For C0 = A00*B0 + A10*B3, this is sum of two 2x2*2x2.
    # We can compute each with Strassen (rank 7), total rank 14 for C0.
    # 6 such blocks: 6*14 = 84.
    
    # But can we do better? If we compute A00*([B0,B1,B2]) which is 2x2 * 2x6,
    # that's rank... well, 2x2 * 2x6 can be done with rank 12 (two 2x2*2x2 Strassen? No, 
    # actually 2x2*2x6 = 2x2*(2x2+2x2+2x2) so 3*7=21? Or maybe we can do better).
    
    # Let me just go with the 84 rank approach using Strassen for each pair.
    
    # Define helper to get Strassen rank-7 decomposition for 2x2*2x2
    # Input: U has 4 rows (for 2x2 matrix X), V has 4 rows (for 2x2 matrix Y)
    # Output: 7 columns each with U part (4 entries), V part (4 entries), W part (4 entries for Z)
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2*2x2 -> 2x2 with rank 7.
        U: 4x7, V: 4x7, W: 4x7
        Convention: X[0]=x11, X[1]=x12, X[2]=x21, X[3]=x22
                    Y[0]=y11, Y[1]=y12, Y[2]=y21, Y[3]=y22
                    Z[0]=z11, Z[1]=z12, Z[2]=z21, Z[3]=z22
        """
        U_s = [
            [1, 0, 1, 0, 1, 1, 1],   # x11
            [0, 0, 0, 0, 1, 1, -1],  # x12
            [0, 1, 0, 0, 0, -1, 0],  # x21
            [1, 1, 0, 1, 0, 0, 1],   # x22
        ]
        V_s = [
            [1, 1, -1, 1, 0, 1, 0],  # y11
            [1, 0, 1, -1, 0, 1, 1],  # y12
            [0, 0, 0, -1, 1, 0, 1],  # y21
            [1, 0, -1, 0, 1, 0, 1],  # y22
        ]
        # Verify: M1 = (x11+x22)(y11+y22)
        # U col0: [1,0,0,1], V col0: [1,1,0,1] -> M1 = (x11+x22)(y11+y22) ✓
        # M2 = (x21+x22)y11
        # U col1: [0,0,1,1], V col1: [1,0,0,0] -> M2 = (x21+x22)y11 ✓
        # M3 = x11(y12-y22)
        # U col2: [1,0,0,0], V col2: [-1,1,0,-1] -> M3 = x11(y12-y22) ✓
        # M4 = x22(y21-y11)
        # U col3: [0,0,0,1], V col3: [1,-1,-1,0] -> M4 = x22(y21-y11) ✓
        # M5 = (x11+x12)y22
        # U col4: [1,1,0,0], V col4: [0,0,1,1] -> M5 = (x11+x12)(y21+y22)... wait
        # V col4: [0,0,1,1] means y21+y22. So M5 = (x11+x12)(y21+y22). 
        # Hmm, standard Strassen M5 = (x11+x12)y22. Let me recheck.
        
        # Actually, let me use the standard Strassen formulation:
        # M1 = (x11+x22)(y11+y22)
        # M2 = (x21+x22)(y11)
        # M3 = (x11)(y12-y22)
        # M4 = (x22)(y21-y11)
        # M5 = (x11+x12)(y22)
        # M6 = (x21-x11)(y11+y12)
        # M7 = (x12-x22)(y21+y22)
        
        # So:
        # U col4 should be [1,1,0,0] (x11+x12), V col4 should be [0,0,0,1] (y22)
        # Let me redo:
        U_s = [
            [1, 0, 1, 0, 1, 1, 1],    # x11
            [0, 0, 0, 0, 1, 1, -1],   # x12
            [0, 1, 0, 0, 0, -1, 0],   # x21
            [1, 1, 0, 1, 0, 0, 1],    # x22
        ]
        V_s = [
            [1, 1, 1, -1, 0, 1, 0],   # y11
            [1, 0, 1, 0, 0, 1, 1],    # y12
            [0, 0, 0, 1, 0, 0, 1],    # y21
            [1, 0, -1, 0, 1, 0, 1],   # y22
        ]
        # M1 = (x11+x22)(y11+y22): U=[1,0,0,1], V=[1,1,0,1] ✓
        # M2 = (x21+x22)(y11): U=[0,0,1,1], V=[1,0,0,0] ✓
        # M3 = (x11)(y12-y22): U=[1,0,0,0], V=[1,1,0,-1] ✓
        # M4 = (x22)(y21-y11): U=[0,0,0,1], V=[-1,0,1,0] ✓
        # M5 = (x11+x12)(y22): U=[1,1,0,0], V=[0,0,0,1] ✓
        # M6 = (x21-x11)(y11+y12): U=[-1,0,1,0]... wait U col5 = [1,1,-1,0]
        # That's x11+x12-x21, not x21-x11. Let me fix.
        
        # M6 = (x21-x11)(y11+y12): U should be [-1,0,1,0], V=[1,1,0,0]
        # M7 = (x12-x22)(y21+y22): U should be [0,1,0,-1], V=[0,0,1,1]
        
        U_s = [
            [1, 0, 1, 0, 1, -1, 0],   # x11
            [0, 0, 0, 0, 1, 0, 1],    # x12
            [0, 1, 0, 0, 0, 1, 0],    # x21
            [1, 1, 0, 1, 0, 0, -1],   # x22
        ]
        V_s = [
            [1, 1, 1, -1, 0, 1, 0],   # y11
            [1, 0, 1, 0, 0, 1, 0],    # y12
            [0, 0, 0, 1, 0, 0, 1],    # y21
            [1, 0, -1, 0, 1, 0, 1],   # y22
        ]
        # M1: U=[1,0,0,1], V=[1,1,0,1] -> (x11+x22)(y11+y22) ✓
        # M2: U=[0,0,1,1], V=[1,0,0,0] -> (x21+x22)y11 ✓
        # M3: U=[1,0,0,0], V=[1,1,0,-1] -> x11(y12-y22) ✓
        # M4: U=[0,0,0,1], V=[-1,0,1,0] -> x22(y21-y11) ✓
        # M5: U=[1,1,0,0], V=[0,0,0,1] -> (x11+x12)y22 ✓
        # M6: U=[-1,0,1,0], V=[1,1,0,0] -> (x21-x11)(y11+y12) ✓
        # M7: U=[0,1,0,-1], V=[0,0,1,1] -> (x12-x22)(y21+y22) ✓
        
        # Now W: Z = M1 - M4 + M6 for z11
        # Z = ... for z12, z21, z22
        # z11 = M1 - M4 + M6
        # z12 = M3 + M5
        # z21 = M2 + M4
        # z22 = M1 - M3 + M7
        
        W_s = [
            [1, 0, 0, -1, 0, 1, 0],   # z11
            [0, 0, 1, 0, 1, 0, 0],    # z12
            [0, 1, 0, 1, 0, 0, 0],    # z21
            [1, 0, -1, 0, 0, 0, 1],   # z22
        ]
        return U_s, V_s, W_s
    
    U_s, V_s, W_s = strassen_2x2()
    
    # Now build the full decomposition.
    # A has 16 entries: A[0..3][0..3]
    # B has 24 entries: B[0..3][0..5]
    # C has 24 entries: C[0..3][0..5]
    #
    # A blocks:
    # A00: A[0:2, 0:2] -> indices 0,1,4,5 in row-major? 
    # A is 4x4, so A[i,j] is at index i*4+j.
    # A00 = A[0:2,0:2] -> indices (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # A01 = A[0:2,2:4] -> indices (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # A10 = A[2:4,0:2] -> indices (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # A11 = A[2:4,2:4] -> indices (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    #
    # For 2x2 matrix [a b; c d], we map to [a,b,c,d]
    # A00: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1] -> indices [0,1,4,5]
    # A01: a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3] -> indices [2,3,6,7]
    # A10: a=A[2,0], b=A[2,1], c=A[3,0], d=A[3,1] -> indices [8,9,12,13]
    # A11: a=A[2,2], b=A[2,3], c=A[3,2], d=A[3,3] -> indices [10,11,14,15]
    
    a_block_indices = {
        'A00': [0, 1, 4, 5],
        'A01': [2, 3, 6, 7],
        'A10': [8, 9, 12, 13],
        'A11': [10, 11, 14, 15],
    }
    
    # B blocks (4x6):
    # B0 = B[0:2,0:2] -> indices (0,0)=0,(0,1)=1,(1,0)=6,(1,1)=7 -> [0,1,6,7]
    # B1 = B[0:2,2:4] -> indices (0,2)=2,(0,3)=3,(1,2)=8,(1,3)=9 -> [2,3,8,9]
    # B2 = B[0:2,4:6] -> indices (0,4)=4,(0,5)=5,(1,4)=10,(1,5)=11 -> [4,5,10,11]
    # B3 = B[2:4,0:2] -> indices (2,0)=12,(2,1)=13,(3,0)=18,(3,1)=19 -> [12,13,18,19]
    # B4 = B[2:4,2:4] -> indices (2,2)=14,(2,3)=15,(3,2)=20,(3,3)=21 -> [14,15,20,21]
    # B5 = B[2:4,4:6] -> indices (2,4)=16,(2,5)=17,(3,4)=22,(3,5)=23 -> [16,17,22,23]
    
    b_block_indices = {
        'B0': [0, 1, 6, 7],
        'B1': [2, 3, 8, 9],
        'B2': [4, 5, 10, 11],
        'B3': [12, 13, 18, 19],
        'B4': [14, 15, 20, 21],
        'B5': [16, 17, 22, 23],
    }
    
    # C blocks (4x6):
    # C0 = C[0:2,0:2] -> indices [0,1,6,7]
    # C1 = C[0:2,2:4] -> indices [2,3,8,9]
    # C2 = C[0:2,4:6] -> indices [4,5,10,11]
    # C3 = C[2:4,0:2] -> indices [12,13,18,19]
    # C4 = C[2:4,2:4] -> indices [14,15,20,21]
    # C5 = C[2:4,4:6] -> indices [16,17,22,23]
    
    c_block_indices = {
        'C0': [0, 1, 6, 7],
        'C1': [2, 3, 8, 9],
        'C2': [4, 5, 10, 11],
        'C3': [12, 13, 18, 19],
        'C4': [14, 15, 20, 21],
        'C5': [16, 17, 22, 23],
    }
    
    # Decomposition:
    # C0 = A00*B0 + A10*B3
    # C1 = A00*B1 + A10*B4
    # C2 = A00*B2 + A10*B5
    # C3 = A01*B0 + A11*B3
    # C4 = A01*B1 + A11*B4
    # C5 = A01*B2 + A11*B5
    
    # For each output block, we need sum of two 2x2*2x2 products.
    # Each 2x2*2x2 uses Strassen rank 7.
    # Total rank = 6 * 2 * 7 = 84.
    
    rank = 84
    U = [[0.0] * rank for _ in range(16)]  # n*m = 16
    V = [[0.0] * rank for _ in range(24)]  # m*p = 24
    W = [[0.0] * rank for _ in range(24)]  # p*n = 24
    
    col = 0
    
    # Define the 12 (actually 12) sub-problems:
    # (A_block, B_block, C_block, sign)
    sub_problems = [
        ('A00', 'B0', 'C0', 1),
        ('A10', 'B3', 'C0', 1),
        ('A00', 'B1', 'C1', 1),
        ('A10', 'B4', 'C1', 1),
        ('A00', 'B2', 'C2', 1),
        ('A10', 'B5', 'C2', 1),
        ('A01', 'B0', 'C3', 1),
        ('A11', 'B3', 'C3', 1),
        ('A01', 'B1', 'C4', 1),
        ('A11', 'B4', 'C4', 1),
        ('A01', 'B2', 'C5', 1),
        ('A11', 'B5', 'C5', 1),
    ]
    
    for a_blk, b_blk, c_blk, _ in sub_problems:
        a_idx = a_block_indices[a_blk]
        b_idx = b_block_indices[b_blk]
        c_idx = c_block_indices[c_blk]
        
        for k in range(7):
            # U part
            for i in range(4):
                U[a_idx[i]][col] = U_s[i][k]
            # V part
            for i in range(4):
                V[b_idx[i]][col] = V_s[i][k]
            # W part
            for i in range(4):
                W[c_idx[i]][col] = W_s[i][k]
            col += 1
    
    assert col == rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
