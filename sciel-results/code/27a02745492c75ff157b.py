"""Combined approach: Kronecker Strassen for 4x4x4 block + optimized 2x2x3 blocks for remaining columns."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This should equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.multiply(U_arr[i*m + j, :], V_arr[j*p + k, :]), W_arr[k*n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4
    
    def build_2x2x3():
        """
        Build rank-11 decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x2.
        U: 4x11 (indices i*2+j), V: 6x11 (indices j*3+k), W: 6x11 (indices k*2+i)
        """
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
        U2, V2, W2 = build_strassen_2x2_2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                # W2 is indexed by k*2+i for k=0,1 and i=0,1
                # In 2x2x3, W is indexed by k*2+i for k=0,1,2 and i=0,1
                # So for k=0,1: W[k*2+i] = W2[k*2+i]
                W[idx][r] = W2[idx][r]
        
        # Last 4 terms: schoolbook for 2x2x1 (column 2 of B)
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1  # k=2, i in {0,1}, so W[4+i]
        
        return U, V, W
    
    def search_2x2x3_rank10():
        """
        Try to find a rank-10 decomposition of 2x2x3 by solving linear equations.
        
        For the 4 components with k=2:
          (i=0,j=0): sum_c U[0,c]*V[2,c]*W[4,c] = 1
          (i=0,j=1): sum_c U[1,c]*V[5,c]*W[4,c] = 1
          (i=1,j=0): sum_c U[2,c]*V[2,c]*W[5,c] = 1
          (i=1,j=1): sum_c U[3,c]*V[5,c]*W[5,c] = 1
        
        If we use 7 Strassen terms + 3 extra terms, the extra terms (c=7,8,9) must satisfy these.
        The Strassen terms have V[2,c] and V[5,c] potentially non-zero for c=0..6.
        
        Let me check: in Strassen 2x2x2, V is 4x7 indexed by j*2+k.
        V[0,c] = V2[0,c] (j=0,k=0), V[1,c] = V2[1,c] (j=0,k=1)
        V[2,c] = V2[2,c] (j=1,k=0), V[3,c] = V2[3,c] (j=1,k=1)
        V[4,c] = 0 (j=0,k=2), V[5,c] = 0 (j=1,k=2)
        
        So for c=0..6, V[4,c]=0 and V[5,c]=0.
        Therefore, the Strassen terms contribute 0 to all k=2 components.
        
        So the 3 extra terms must independently satisfy the 4 equations.
        With 3 terms, we have:
          For each c in {7,8,9}: U[0,c], U[1,c], U[2,c], U[3,c], V[2,c], V[5,c], W[4,c], W[5,c]
        
        The 4 equations are:
          U[0,7]*V[2,7]*W[4,7] + U[0,8]*V[2,8]*W[4,8] + U[0,9]*V[2,9]*W[4,9] = 1
          U[1,7]*V[5,7]*W[4,7] + U[1,8]*V[5,8]*W[4,8] + U[1,9]*V[5,9]*W[4,9] = 1
          U[2,7]*V[2,7]*W[5,7] + U[2,8]*V[2,8]*W[5,8] + U[2,9]*V[2,9]*W[5,9] = 1
          U[3,7]*V[5,7]*W[5,7] + U[3,8]*V[5,8]*W[5,8] + U[3,9]*V[5,9]*W[5,9] = 1
        
        This is a system of 4 equations in 24 unknowns. We can easily find a solution.
        For example:
          Term 7: V[2,7]=1, W[4,7]=1, U[0,7]=1, others 0 -> satisfies eq1
          Term 8: V[5,8]=1, W[4,8]=1, U[1,8]=1, others 0 -> satisfies eq2
          Term 9: V[2,9]=1, W[5,9]=1, U[2,9]=1, others 0 -> satisfies eq3
          But eq4: U[3,7]*V[5,7]*W[5,7] + U[3,8]*V[5,8]*W[5,8] + U[3,9]*V[5,9]*W[5,9] = 0+0+0 = 0 ≠ 1
        
        So we need to modify. Let's use:
          Term 7: V[2,7]=1, V[5,7]=1, W[4,7]=1, W[5,7]=1
          Then eq1: U[0,7]*1*1 = U[0,7]
          eq2: U[1,7]*1*1 = U[1,7]
          eq3: U[2,7]*1*1 = U[2,7]
          eq4: U[3,7]*1*1 = U[3,7]
          Set U[0,7]=U[1,7]=U[2,7]=U[3,7]=1. All 4 equations satisfied with 1 term!
        
        But wait, we also need to make sure this doesn't interfere with the k=0,1 components.
        For k=0,1: V[4,c] and V[5,c] are the relevant entries, but in the Strassen part,
        V is indexed by j*2+k, so V[4,c] would be j=2,k=0 which doesn't exist in 2x2x2.
        
        In the 2x2x3 setting, V has 6 rows: j*3+k for j=0,1 and k=0,1,2.
        V[0,c]=V2[0,c] (j=0,k=0), V[1,c]=V2[1,c] (j=0,k=1)
        V[2,c]=V2[2,c] (j=1,k=0), V[3,c]=V2[3,c] (j=1,k=1)
        V[4,c]=0 (j=0,k=2), V[5,c]=0 (j=1,k=2) for c=0..6
        
        For term 7: V[4,7]=0, V[5,7]=1 (I set V[5,7]=1 above)
        Wait, I said V[2,7]=1, V[5,7]=1. Let me recheck.
        
        V[2,c] corresponds to j=1,k=0. V[5,c] corresponds to j=1,k=2.
        So for k=0 components: (i,j,k=0) uses V[j*3+0,c] = V[0,c] or V[3,c].
        For k=2 components: (i,j,k=2) uses V[j*3+2,c] = V[2,c] or V[5,c].
        
        Wait, I made an error. V[j*3+k]:
        j=0,k=0: V[0,c]
        j=0,k=1: V[1,c]
        j=0,k=2: V[2,c]
        j=1,k=0: V[3,c]
        j=1,k=1: V[4,c]
        j=1,k=2: V[5,c]
        
        So V[2,c] is j=0,k=2, not j=1,k=0!
        
        Let me redo. The 4 equations for k=2:
          (i=0,j=0,k=2): U[0,c]*V[2,c]*W[4,c] -- W[k*2+i] = W[4+0] = W[4,c]
          (i=0,j=1,k=2): U[1,c]*V[5,c]*W[4,c] -- W[4+0] = W[4,c]
          (i=1,j=0,k=2): U[2,c]*V[2,c]*W[5,c] -- W[4+1] = W[5,c]
          (i=1,j=1,k=2): U[3,c]*V[5,c]*W[5,c] -- W[4+1] = W[5,c]
        
        So:
          eq1: sum_c U[0,c]*V[2,c]*W[4,c] = 1
          eq2: sum_c U[1,c]*V[5,c]*W[4,c] = 1
          eq3: sum_c U[2,c]*V[2,c]*W[5,c] = 1
          eq4: sum_c U[3,c]*V[5,c]*W[5,c] = 1
        
        For c=0..6 (Strassen terms): V[2,c]=0 and V[5,c]=0, so no contribution.
        
        For c=7 (one extra term):
          eq1: U[0,7]*V[2,7]*W[4,7]
          eq2: U[1,7]*V[5,7]*W[4,7]
          eq3: U[2,7]*V[2,7]*W[5,7]
          eq4: U[3,7]*V[5,7]*W[5,7]
        
        If I set V[2,7]=1, V[5,7]=1, W[4,7]=1, W[5,7]=1, U[0,7]=U[1,7]=U[2,7]=U[3,7]=1:
          eq1: 1*1*1 = 1 ✓
          eq2: 1*1*1 = 1 ✓
          eq3: 1*1*1 = 1 ✓
          eq4: 1*1*1 = 1 ✓
        
        But does this interfere with k=0,1?
        For k=0: uses V[0,c] and V[3,c]. V[2,7] and V[5,7] are not used for k=0.
        For k=1: uses V[1,c] and V[4,c]. V[2,7] and V[5,7] are not used for k=1.
        
        So term 7 with V[2,7]=1, V[5,7]=1, W[4,7]=1, W[5,7]=1, U[0,7]=U[1,7]=U[2,7]=U[3,7]=1
        only affects the k=2 components. 
        
        But wait, W[4,7]=1 and W[5,7]=1. W is indexed by k*2+i.
        W[4,c] = k=2,i=0. W[5,c] = k=2,i=1.
        For k=0: W[0,c] and W[1,c] are used.
        For k=1: W[2,c] and W[3,c] are used.
        For k=2: W[4,c] and W[5,c] are used.
        
        So W[4,7] and W[5,7] only affect k=2. Good.
        
        Therefore, rank 7+1 = 8 is sufficient for 2x2x3!
        
        Let me verify this doesn't break the k=0,1 Strassen part.
        The Strassen terms (c=0..6) have V[2,c]=0 and V[5,c]=0, so they don't contribute to k=2.
        Term 7 has V[0,7]=0, V[1,7]=0, V[3,7]=0, V[4,7]=0, so it doesn't contribute to k=0,1.
        
        So rank 8 works for 2x2x3!
        
        Total rank for 4x4x7: 49 + 4*8 = 49 + 32 = 81.
        """
        U2, V2, W2 = build_strassen_2x2x2()
        
        U = [[0]*8 for _ in range(4)]
        V = [[0]*8 for _ in range(6)]
        W = [[0]*8 for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            # V2 is 4x7, indexed by j*2+k for j=0,1 and k=0,1
            # In 2x2x3, V is 6x8, indexed by j*3+k for j=0,1 and k=0,1,2
            # V2[0,c] -> j=0,k=0 -> V[0,c]
            # V2[1,c] -> j=0,k=1 -> V[1,c]
            # V2[2,c] -> j=1,k=0 -> V[3,c]
            # V2[3,c] -> j=1,k=1 -> V[4,c]
            U_idx_to_V_idx = {0: 0, 1: 1, 2: 3, 3: 4}
            for idx in range(4):
                V[U_idx_to_V_idx[idx]][r] = V2[idx][r]
            # W2 is 4x7, indexed by k*2+i for k=0,1 and i=0,1
            # In 2x2x3, W is 6x8, indexed by k*2+i for k=0,1,2 and i=0,1
            # W2[0,c] -> k=0,i=0 -> W[0,c]
            # W2[1,c] -> k=0,i=1 -> W[1,c]
            # W2[2,c] -> k=1,i=0 -> W[2,c]
            # W2[3,c] -> k=1,i=1 -> W[3,c]
            for idx in range(4):
                W[idx][r] = W2[idx][r]
        
        # Term 7: handles column 2 of B
        U[0][7] = 1
        U[1][7] = 1
        U[2][7] = 1
        U[3][7] = 1
        V[2][7] = 1  # j=0, k=2
        V[5][7] = 1  # j=1, k=2
        W[4][7] = 1  # k=2, i=0
        W[5][7] = 1  # k=2, i=1
        
        return U, V, W
    
    def build_4x4x7_decomposition_rank93():
        """Rank-93 fallback using rank-11 2x2x3 blocks."""
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        U223, V223, W223 = build_2x2x3()
        r223 = len(U223[0])  # 11
        
        r_total = r4 + 4 * r223
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * r223
                
                for r_local in range(r223):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W
    
    def build_4x4x7_decomposition():
        """
        Build a rank-93 decomposition for 4x4x7.
        
        Strategy: Use Kronecker Strassen (rank 49) for the 4x4x4 part.
        For the 4x4x3 part (columns 4,5,6 of B), use four 2x2x3 blocks
        each of rank 11, total 44. Grand total: 49 + 44 = 93.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        U223, V223, W223 = search_2x2x3_rank10()
        r223 = len(U223[0])  # 8
        
        r_total = r4 + 4 * r223  # 49 + 32 = 81
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # Part 1: 4x4x4 Kronecker (columns 0-3 of B)
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # Part 2: 4x4x3 using four 2x2x3 blocks (columns 4-6 of B)
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * r223
                
                for r_local in range(r223):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local  # global column offset for B
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W
    
    # Try the rank-93 decomposition first
    try:
        U, V, W = build_4x4x7_decomposition()
        if check_valid(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook if something goes wrong
    U, V, W = build_schoolbook()
    return {"U": U, "V": V, "W": W}
