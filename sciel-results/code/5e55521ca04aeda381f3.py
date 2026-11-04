"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            max_err = np.max(np.abs(T - 1.0))
            return max_err < tol
        except Exception:
            return False
    
    def strassen_2x2x2():
        """Return rank-7 Strassen decomposition for 2x2x2.
        U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for C = A*B where A,B are 2x2
        # A indices: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
        # B indices: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
        # C indices: C[0,0], C[0,1], C[1,0], C[1,1]
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each rank component r, we need:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        
        # Let's define the 7 rank terms:
        # r=0: M1 = (A00+A11)(B00+B11)
        #   U: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
        #   V: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
        #   W: C00+=1, C11+=1 -> W[0,0]=1, W[3,0]=1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # r=1: M2 = (A10+A11)B00
        #   U: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
        #   V: B00=1 -> V[0,1]=1
        #   W: C10+=1 -> W[2,1]=1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # r=2: M3 = A00(B01-B11)
        #   U: A00=1 -> U[0,2]=1
        #   V: B01=1, B11=-1 -> V[1,2]=1, V[3,2]=-1
        #   W: C01+=1 -> W[1,2]=1
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        
        # r=3: M4 = A11(B10-B00)
        #   U: A11=1 -> U[3,3]=1
        #   V: B10=1, B00=-1 -> V[2,3]=1, V[0,3]=-1
        #   W: C00+=1, C10+=1 -> W[0,3]=1, W[2,3]=1
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # r=4: M5 = (A00+A01)B11
        #   U: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
        #   V: B11=1 -> V[3,4]=1
        #   W: C00-=1, C01+=1 -> W[0,4]=-1, W[1,4]=1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # r=5: M6 = (A10-A00)(B00+B01)
        #   U: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
        #   V: B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
        #   W: C11+=1 -> W[3,5]=1
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # r=6: M7 = (A01-A11)(B10+B11)
        #   U: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
        #   V: B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
        #   W: C00+=1, C11+=1 -> W[0,6]=1, W[3,6]=1
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W
    
    def try_strassen_blocks():
        """Try to build a rank-36 decomposition using Strassen blocks.
        
        For 2x4x5:
        - Split m=4 into two groups of 2: m1=[0,1], m2=[2,3]
        - Split p=5 into p1=4, p2=1
        - For each (m_group, p_group) block, use appropriate decomposition
        
        Actually, let's think differently:
        - n=2, m=4, p=5
        - We can view this as: for each output position (i,k), 
          C[i,k] = sum_j A[i,j] * B[j,k]
        
        Block decomposition approach:
        - Split j into {0,1} and {2,3}
        - Split k into {0,1,2,3} and {4}
        
        For j in {0,1}, k in {0,1}: 2x2x2 block -> rank 7 (Strassen)
        For j in {0,1}, k in {2,3}: 2x2x2 block -> rank 7 (Strassen)
        For j in {2,3}, k in {0,1}: 2x2x2 block -> rank 7 (Strassen)
        For j in {2,3}, k in {2,3}: 2x2x2 block -> rank 7 (Strassen)
        For j in {0,1}, k in {4}: 2x2x1 block -> rank 2
        For j in {2,3}, k in {4}: 2x2x1 block -> rank 2
        
        Total: 7+7+7+7+2+2 = 32? That seems too good.
        
        Wait, I need to be more careful. The Strassen decomposition works for 
        multiplying two 2x2 matrices. In our case, we have:
        - A is 2x4, B is 4x5
        - We're computing C = A*B which is 2x5
        
        If we split A into A_left (2x2) and A_right (2x2), and B into 
        B_top (4x4) and B_bottom (4x1), then:
        C = A_left * B_top_left + A_right * B_top_right + A_left * B_bottom_left + A_right * B_bottom_right
        
        Hmm, this isn't quite right either. Let me think again.
        
        Actually, the correct block decomposition:
        A = [A1 | A2] where A1 is 2x2, A2 is 2x2
        B = [B1; B2] where B1 is 2x5, B2 is 2x5
        
        C = A1*B1 + A2*B2
        
        Now A1*B1: 2x2 times 2x5. Split B1 into B1a (2x4) and B1b (2x1).
        A1*B1a: 2x2 times 2x4. Split into two 2x2x2 blocks.
        A1*B1b: 2x2 times 2x1. Rank 2.
        
        Similarly for A2.
        
        Total rank: 2*(7+7+2) + 2*(7+7+2) = 2*16 + 2*16 = 64? No, that's wrong.
        
        Let me reconsider. For 2x2 times 2x4:
        - Split the 4 columns into two groups of 2
        - Each 2x2x2 block has rank 7
        - Total: 7+7 = 14
        
        For 2x2 times 2x1: rank 2
        
        So A1*B1: 14 + 2 = 16
        A2*B2: 14 + 2 = 16
        Total: 32
        
        But wait, can we do better? The known lower bound for 2x4x5 is around 36.
        Let me just try rank 36 with a different approach.
        """
        pass
    
    # Try to construct explicit Strassen-based decomposition
    def build_strassen_decomposition():
        """Build a rank-36 decomposition for 2x4x5 using Strassen blocks."""
        n, m, p = 2, 4, 5
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Get Strassen 2x2x2 decomposition
        S_U, S_V, S_W = strassen_2x2x2()
        
        # Block structure:
        # A (2x4) = [A1 | A2] where A1 = A[:,0:2], A2 = A[:,2:4]
        # B (4x5) = [B1; B2] where B1 = B[0:2,:], B2 = B[2:4,:]
        # C = A1*B1 + A2*B2
        #
        # A1*B1: A1 is 2x2, B1 is 2x5
        # Split B1 into B1a (2x4) and B1b (2x1)
        # B1a = B[0:2, 0:4], B1b = B[0:2, 4]
        #
        # A1*B1a: 2x2 * 2x4, split into two 2x2x2 blocks
        # Block 1: A1 * B[0:2, 0:2] -> Strassen
        # Block 2: A1 * B[0:2, 2:4] -> Strassen
        # A1*B1b: 2x2 * 2x1 -> rank 2
        #
        # Similarly for A2*B2
        
        col = 0
        # A1*B1a block 1: j in {0,1}, k in {0,1}
        for r in range(7):
            U[0:2, col] = S_U[:, r] * 1  # A1 rows 0,1
            U[2:4, col] = 0  # A2 rows
            # Wait, this isn't right. U has shape (n*m, rank) = (8, 36)
            # U[i*m + j, r] where i in {0,1}, j in {0,1,2,3}
            # For A1*B1a block 1: i in {0,1}, j in {0,1}, k in {0,1}
            # U[i*4 + j, col] should be S_U[i*2 + j, r]
            # V[j*5 + k, col] should be S_V[j*2 + k, r]
            # W[k*2 + i, col] should be S_W[k*2 + i, r]
            
        # Let me redo this more carefully
        U = np.zeros((8, rank))
        V = np.zeros((20, rank))
        W = np.zeros((10, rank))
        
        col = 0
        # Block 1: A1 * B[0:2, 0:2]
        # i in {0,1}, j in {0,1}, k in {0,1}
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*4 + j, col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j*5 + k, col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2 + i, col] = S_W[k*2 + i, r]
            col += 1
        # col = 7
        
        # Block 2: A1 * B[0:2, 2:4]
        # i in {0,1}, j in {0,1}, k in {2,3}
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*4 + j, col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j*5 + (k+2), col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*2 + i, col] = S_W[k*2 + i, r]
            col += 1
        # col = 14
        
        # Block 3: A1 * B[0:2, 4:5]
        # i in {0,1}, j in {0,1}, k in {4}
        # Rank 2: for each j, U[i*4+j] * V[j*5+4] * W[4*2+i] = 1
        for j in range(2):
            for i in range(2):
                U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*2 + 0, col] = 1
            W[4*2 + 1, col] = 0
            # Wait, this doesn't work. Let me think...
            # For k=4, we need: sum_r U[i*4+j, r] * V[j*5+4, r] * W[8+i, r] = 1
            # For rank 2, we can use:
            # r=0: j=0: U[i*4+0, r]=1, V[0*5+4, r]=1, W[8+i, r]=1 for all i
            # r=1: j=1: U[i*4+1, r]=1, V[1*5+4, r]=1, W[8+i, r]=1 for all i
            # But then for j=0, k=4: sum = 1*1*1 + 0 = 1 ✓
            # For j=1, k=4: sum = 0 + 1*1*1 = 1 ✓
            # But we also need it to be 0 for other (i,j,k) combos... 
            # Actually no, the tensor T[i,j,k] = 1 for ALL i,j,k in the matmul tensor.
            # So we just need sum_r U[i*4+j,r] * V[j*5+k,r] * W[k*2+i,r] = 1 for all i,j,k.
            
            col += 1
        # col = 16
        
        # Block 4: A2 * B[2:4, 0:2]
        # i in {0,1}, j in {2,3}, k in {0,1}
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*4 + (j+2), col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j+2)*5 + k, col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2 + i, col] = S_W[k*2 + i, r]
            col += 1
        # col = 23
        
        # Block 5: A2 * B[2:4, 2:4]
        # i in {0,1}, j in {2,3}, k in {2,3}
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*4 + (j+2), col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j+2)*5 + (k+2), col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*2 + i, col] = S_W[k*2 + i, r]
            col += 1
        # col = 30
        
        # Block 6: A2 * B[2:4, 4:5]
        # i in {0,1}, j in {2,3}, k in {4}
        for j in range(2):
            for i in range(2):
                U[i*4 + (j+2), col] = 1
            V[(j+2)*5 + 4, col] = 1
            W[8+0, col] = 1
            W[8+1, col] = 0
            col += 1
        # col = 32
        
        # We have 32 rank terms. But we need to verify this works.
        # The issue is that the W matrix entries might conflict.
        
        # Actually, I realize the above construction has issues because 
        # the same W[k*2+i] entry is used by multiple blocks, and they 
        # need to add up correctly.
        
        # Let me try a different approach: just return schoolbook for now
        # and use gradient descent to try to improve.
        return None
    
    # Try explicit Strassen construction with proper verification
    def try_explicit_strassen():
        """Try to build and verify a Strassen-based decomposition."""
        n, m, p = 2, 4, 5
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        S_U, S_V, S_W = strassen_2x2x2()
        
        col = 0
        # Block 1: j in {0,1}, k in {0,1} - Strassen
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j, col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j*p + k, col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*n + i, col] = S_W[k*2 + i, r]
            col += 1
        
        # Block 2: j in {0,1}, k in {2,3} - Strassen
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j, col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j*p + (k+2), col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*n + i, col] = S_W[k*2 + i, r]
            col += 1
        
        # Block 3: j in {0,1}, k in {4} - rank 2
        for j in range(2):
            for i in range(2):
                U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            for i in range(2):
                W[4*n + i, col] = 1
            col += 1
        
        # Block 4: j in {2,3}, k in {0,1} - Strassen
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + (j+2), col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j+2)*p + k, col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*n + i, col] = S_W[k*2 + i, r]
            col += 1
        
        # Block 5: j in {2,3}, k in {2,3} - Strassen
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + (j+2), col] = S_U[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j+2)*p + (k+2), col] = S_V[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*n + i, col] = S_W[k*2 + i, r]
            col += 1
        
        # Block 6: j in {2,3}, k in {4} - rank 2
        for j in range(2):
            for i in range(2):
                U[i*m + (j+2), col] = 1
            V[(j+2)*p + 4, col] = 1
            for i in range(2):
                W[4*n + i, col] = 1
            col += 1
        
        # Total col = 7+7+2+7+7+2 = 32
        # We allocated 36, so we have 4 unused columns. Let's use 32.
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # First try explicit Strassen construction
    result = try_explicit_strassen()
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
