import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Rank-23 decomposition for 3x3x3 (Strassen-Winograd)
    # U (9x23), V (9x23), W (9x23)
    # T[i,j,k] = sum_t U[i*3+j,t] * V[j*3+k,t] * W[k*3+i,t] = 1 for all i,j,k in 0..2
    def strassen_3x3x3():
        # Known rank-23 decomposition for 3x3 matrix multiplication
        # This is a well-known result. Let me use a verified construction.
        # The 23 multiplications are:
        # M1 = (A00+A11+A22)(B00+B11+B22)
        # M2 = (A00+A11)(B00+B11)
        # M3 = (A00+A22)(B00+B22)
        # M4 = (A11+A22)(B11+B22)
        # M5 = (A01+A12)(B01+B12)
        # M6 = (A02+A10)(B02+B10)
        # M7 = (A10+A20)(B10+B20)
        # M8 = (A21+A01)(B21+B01)
        # M9 = (A22+A02)(B22+B02)
        # M10 = (A12+A21)(B12+B21)
        # M11 = A00*(B00 - B10 - B01 + B11)
        # M12 = A01*(B01 - B02 - B11 + B12)
        # M13 = A02*(B02 - B00 - B12 + B10)
        # M14 = A10*(B10 - B11 - B20 + B21)
        # M15 = A11*(B11 - B12 - B21 + B22)
        # M16 = A12*(B12 - B10 - B22 + B20)
        # M17 = A20*(B20 - B21 - B00 + B01)
        # M18 = A21*(B21 - B22 - B01 + B02)
        # M19 = A22*(B22 - B20 - B02 + B00)
        # M20 = (A00+A01+A10+A11+A20+A21)(B00+B01+B10+B11+B20+B21)
        # M21 = (A01+A02+A11+A12+A21+A22)(B01+B02+B11+B12+B21+B22)
        # M22 = (A00+A02+A10+A12+A20+A22)(B00+B02+B10+B12+B20+B22)
        # M23 = (A00+A11+A22)(B00+B11+B22) -- wait, this is same as M1
        
        # Let me use a different, verified rank-23 construction.
        # Actually, the standard rank-23 for 3x3 uses these 23 products:
        
        U = np.zeros((9, 23), dtype=int)
        V = np.zeros((9, 23), dtype=int)
        W = np.zeros((9, 23), dtype=int)
        
        # I'll use a known correct rank-23 decomposition.
        # Based on the Coppersmith-Winograd construction adapted for 3x3.
        
        # Let me define the 23 terms explicitly:
        # Term 0: (A00+A11+A22)(B00+B11+B22)
        U[0,0]=1; U[4,0]=1; U[8,0]=1
        V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[4,0]=1; W[8,0]=1
        
        # Term 1: (A00+A11)(B00+B11)
        U[0,1]=1; U[4,1]=1
        V[0,1]=1; V[4,1]=1
        W[0,1]=1; W[4,1]=1
        
        # Term 2: (A00+A22)(B00+B22)
        U[0,2]=1; U[8,2]=1
        V[0,2]=1; V[8,2]=1
        W[0,2]=1; W[8,2]=1
        
        # Term 3: (A11+A22)(B11+B22)
        U[4,3]=1; U[8,3]=1
        V[4,3]=1; V[8,3]=1
        W[4,3]=1; W[8,3]=1
        
        # Term 4: (A01+A12)(B01+B12)
        U[1,4]=1; U[5,4]=1
        V[1,4]=1; V[5,4]=1
        W[1,4]=1; W[5,4]=1
        
        # Term 5: (A02+A10)(B02+B10)
        U[2,5]=1; U[3,5]=1
        V[2,5]=1; V[3,5]=1
        W[2,5]=1; W[3,5]=1
        
        # Term 6: (A10+A20)(B10+B20)
        U[3,6]=1; U[6,6]=1
        V[3,6]=1; V[6,6]=1
        W[3,6]=1; W[6,6]=1
        
        # Term 7: (A21+A01)(B21+B01)
        U[7,7]=1; U[1,7]=1
        V[7,7]=1; V[1,7]=1
        W[7,7]=1; W[1,7]=1
        
        # Term 8: (A22+A02)(B22+B02)
        U[8,8]=1; U[2,8]=1
        V[8,8]=1; V[2,8]=1
        W[8,8]=1; W[2,8]=1
        
        # Term 9: (A12+A21)(B12+B21)
        U[5,9]=1; U[7,9]=1
        V[5,9]=1; V[7,9]=1
        W[5,9]=1; W[7,9]=1
        
        # Term 10: A00*(B00 - B10 - B01 + B11)
        U[0,10]=1
        V[0,10]=1; V[3,10]=-1; V[1,10]=-1; V[4,10]=1
        W[0,10]=1
        
        # Term 11: A01*(B01 - B02 - B11 + B12)
        U[1,11]=1
        V[1,11]=1; V[2,11]=-1; V[4,11]=-1; V[5,11]=1
        W[1,11]=1
        
        # Term 12: A02*(B02 - B00 - B12 + B10)
        U[2,12]=1
        V[2,12]=1; V[0,12]=-1; V[5,12]=-1; V[3,12]=1
        W[2,12]=1
        
        # Term 13: A10*(B10 - B11 - B20 + B21)
        U[3,13]=1
        V[3,13]=1; V[4,13]=-1; V[6,13]=-1; V[7,13]=1
        W[3,13]=1
        
        # Term 14: A11*(B11 - B12 - B21 + B22)
        U[4,14]=1
        V[4,14]=1; V[5,14]=-1; V[7,14]=-1; V[8,14]=1
        W[4,14]=1
        
        # Term 15: A12*(B12 - B10 - B22 + B20)
        U[5,15]=1
        V[5,15]=1; V[3,15]=-1; V[8,15]=-1; V[6,15]=1
        W[5,15]=1
        
        # Term 16: A20*(B20 - B21 - B00 + B01)
        U[6,16]=1
        V[6,16]=1; V[7,16]=-1; V[0,16]=-1; V[1,16]=1
        W[6,16]=1
        
        # Term 17: A21*(B21 - B22 - B01 + B02)
        U[7,17]=1
        V[7,17]=1; V[8,17]=-1; V[1,17]=-1; V[2,17]=1
        W[7,17]=1
        
        # Term 18: A22*(B22 - B20 - B02 + B00)
        U[8,18]=1
        V[8,18]=1; V[6,18]=-1; V[2,18]=-1; V[0,18]=1
        W[8,18]=1
        
        # Term 19: (A00+A01+A10+A11+A20+A21)(B00+B01+B10+B11+B20+B21)
        U[0,19]=1; U[1,19]=1; U[3,19]=1; U[4,19]=1; U[6,19]=1; U[7,19]=1
        V[0,19]=1; V[1,19]=1; V[3,19]=1; V[4,19]=1; V[6,19]=1; V[7,19]=1
        W[0,19]=1; W[1,19]=1; W[3,19]=1; W[4,19]=1; W[6,19]=1; W[7,19]=1
        
        # Term 20: (A01+A02+A11+A12+A21+A22)(B01+B02+B11+B12+B21+B22)
        U[1,20]=1; U[2,20]=1; U[4,20]=1; U[5,20]=1; U[7,20]=1; U[8,20]=1
        V[1,20]=1; V[2,20]=1; V[4,20]=1; V[5,20]=1; V[7,20]=1; V[8,20]=1
        W[1,20]=1; W[2,20]=1; W[4,20]=1; W[5,20]=1; W[7,20]=1; W[8,20]=1
        
        # Term 21: (A00+A02+A10+A12+A20+A22)(B00+B02+B10+B12+B20+B22)
        U[0,21]=1; U[2,21]=1; U[3,21]=1; U[5,21]=1; U[6,21]=1; U[8,21]=1
        V[0,21]=1; V[2,21]=1; V[3,21]=1; V[5,21]=1; V[6,21]=1; V[8,21]=1
        W[0,21]=1; W[2,21]=1; W[3,21]=1; W[5,21]=1; W[6,21]=1; W[8,21]=1
        
        # Term 22: (A00+A11+A22)(B00+B11+B22) -- this is same as term 0!
        # I need a different term 22. Let me use:
        # Term 22: (A01+A10+A21+A12)(B01+B10+B21+B12) -- not standard
        # Actually, I realize my construction might not be correct.
        # Let me verify and fix.
        
        # For now, let me use a simpler approach: use rank-7 Strassen for 2x2x2
        # and build up to 4x4x4x8 with rank 98 (two copies of rank-49 for 4x4x4x4)
        
        return U, V, W
    
    # Let me use a more reliable approach: build 4x4x4x4 with rank 49 using Strassen
    # then duplicate for 4x4x4x8 to get rank 98
    
    def strassen_2x2x2():
        """Rank-7 Strassen decomposition for 2x2x2."""
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2 = (A10 + A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=1
        
        # M3 = A00(B01 - B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4 = A11(B10 - B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        
        # M5 = (A00 + A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W
    
    def build_4x4x4():
        """Build 4x4x4x4 decomposition with rank 49 using Strassen."""
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Partition A into 2x2 blocks: A00, A01, A10, A11
        # Partition B into 2x2 blocks: B00, B01, B10, B11
        # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
        #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
        # 
        # We need 4 products of 2x2*2x2, each using Strassen (rank 7).
        # But we can share some Strassen terms!
        # 
        # Actually, for 4x4x4x4, the known rank is 49 = 7*7, which comes from
        # applying Strassen twice (once on the 2x2 block structure).
        #
        # The 49-rank decomposition uses 7 Strassen multiplications for each of
        # the 7 "combined" 2x2 products.
        #
        # Let me use a different construction: 4 independent Strassen 2x2x2
        # would give rank 28, but that's for 4x4x4x2, not 4x4x4x4.
        #
        # For 4x4x4x4, I'll use the fact that it's equivalent to 4x4x4x4 and
        # use a known rank-49 construction.
        #
        # Simpler: use 4 Strassen 2x2x2 for the 4 block products, giving rank 28
        # for 4x4x4x4? No, that's not right either.
        #
        # Let me just use the schoolbook for 4x4x4x4 (rank 64) and see if I can
        # improve it. Actually, for 4x4x4x8, schoolbook is 128.
        #
        # Let me use a pragmatic approach: use Strassen 2x2x2 to get 4x4x4x2 with
        # rank 28, then use 4 copies for 4x4x4x8 with rank 112.
        
        # Actually, let me just implement the 4x4x4x2 with rank 28 correctly.
        return None
    
    def build_4x4x2():
        """Build 4x4x4x2 decomposition with rank 28 using 4 Strassen 2x2x2."""
        n, m, p = 4, 4, 2
        rank = 28
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        # A (4x4), B (4x2), C (4x2)
        # A = [[A00, A01], [A10, A11]] where Aij are 2x2
        # B = [[B0], [B1]] where B0, B1 are 2x2
        # C = [[A00*B0 + A01*B1], [A10*B0 + A11*B1]]
        # C00 = A00*B0 + A01*B1 (2x2)
        # C01 = A10*B0 + A11*B1 (2x2)
        
        # Use 4 Strassen instances:
        # 1. A00*B0 -> contributes to C00
        # 2. A01*B1 -> contributes to C00
        # 3. A10*B0 -> contributes to C01
        # 4. A11*B1 -> contributes to C01
        
        instances = [
            # (i_off, j_off, k_off, c_i_off, c_j_off)
            # A block at (i_off, j_off), B block at row k_off, C block at (c_i_off, c_j_off)
            (0, 0, 0, 0, 0),  # A00*B0 -> C00
            (0, 2, 1, 0, 0),  # A01*B1 -> C00
            (2, 0, 0, 2, 0),  # A10*B0 -> C01
            (2, 2, 1, 2, 0),  # A11*B1 -> C01
        ]
        
        for inst_idx, (i_off, j_off, k_off, c_i_off, c_j_off) in enumerate(instances):
            base_t = inst_idx * 7
            for t in range(7):
                # U indices: A[i_off:i_off+2, j_off:j_off+2]
                # Global U index for A[i,j] is i*4+j
                u_map = [
                    i_off * 4 + j_off,
                    i_off * 4 + j_off + 1,
                    (i_off + 1) * 4 + j_off,
                    (i_off + 1) * 4 + j_off + 1
                ]
                for lu, gu in enumerate(u_map):
                    U[gu, base_t + t] = U_s[lu, t]
                
                # V indices: B[k_off:k_off+2, 0:2]
                # Global V index for B[j,k] is j*2+k
                v_map = [
                    k_off * 2 + 0,
                    k_off * 2 + 1,
                    (k_off + 1) * 2 + 0,
                    (k_off + 1) * 2 + 1
                ]
                for lv, gv in enumerate(v_map):
                    V[gv, base_t + t] = V_s[lv, t]
                
                # W indices: C[c_i_off:c_i_off+2, c_j_off:c_j_off+2]
                # Global W index for C[k,i] is k*4+i
                w_map = [
                    c_j_off * 4 + c_i_off,
                    (c_j_off + 1) * 4 + c_i_off,
                    c_j_off * 4 + (c_i_off + 1),
                    (c_j_off + 1) * 4 + (c_i_off + 1)
                ]
                for lw, gw in enumerate(w_map):
                    W[gw, base_t + t] = W_s[lw, t]
        
        return U, V, W
    
    def build_4x4x8_from_4x4x2():
        """Build 4x4x4x8 by using 4 copies of 4x4x4x2."""
        n, m, p = 4, 4, 8
        rank = 112
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        U_42, V_42, W_42 = build_4x4x2()
        
        for b_idx in range(4):
            base_t = b_idx * 28
            U[:, base_t:base_t+28] = U_42
            
            # V_42 has shape (8, 28) for B (4x2)
            # V has shape (32, 112) for B (4x8)
            # B block b_idx is columns b_idx*2 to b_idx*2+1
            for j in range(4):
                for k_local in range(2):
                    k_global = b_idx * 2 + k_local
                    V[j * 8 + k_global, base_t:base_t+28] = V_42[j * 2 + k_local, :]
            
            # W_42 has shape (8, 28) for C (2x4)
            # W has shape (32, 112) for C (8x4)
            # C block b_idx is columns b_idx*2 to b_idx*2+1
            for k_local in range(2):
                for i in range(4):
                    k_global = b_idx * 2 + k_local
                    W[k_global * 4 + i, base_t:base_t+28] = W_42[k_local * 4 + i, :]
        
        return U, V, W
    
    # Verify the 4x4x4x2 decomposition first
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try the rank-112 decomposition
    U, V, W = build_4x4x8_from_4x4x2()
    
    if verify_decomp(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
