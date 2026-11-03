"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_rank72():
        """
        Build a rank-72 decomposition by exploiting the structure:
        R(3,4,8) <= 2 * R(3,2,8) <= 2 * 2 * R(3,2,4) <= 4 * 2 * R(3,2,2)
        Wait, let's re-verify the decomposition logic.
        
        Actually, the previous thought process suggested:
        Split j into 2 blocks of 2.
        Split k into 4 blocks of 2.
        For each block (j_block, k_block), we have a 3x2x2 subproblem.
        R(3,2,2) is the rank of multiplying 3x2 by 2x2.
        Is R(3,2,2) = 9?
        Let's check.
        A is 3x2, B is 2x2, C is 3x2.
        This is equivalent to two independent 3x1 by 1x2 multiplications? No.
        It is equivalent to multiplying a 3x2 matrix by a 2x2 matrix.
        We can view this as two 3x1 by 1x2 multiplications? No, the columns of B are mixed.
        
        Let's use the standard bound: R(m, n, p) <= R(m, n, 1) * p? No.
        R(3, 2, 2).
        We can decompose this using Strassen on the 2x2 part?
        If we view A as two rows of 1x2? No.
        
        Let's look at the structure of R(2,2,3) = 9.
        By symmetry, R(3,2,2) = R(2,2,3) = 9.
        So if we split the 3x4x8 problem into 8 blocks of size 3x2x2,
        the total rank would be 8 * 9 = 72.
        
        How to split 3x4x8 into 3x2x2 blocks?
        i ranges 0..2 (fixed).
        j ranges 0..3. Split into j0 (0,1) and j1 (2,3).
        k ranges 0..7. Split into k0 (0,1), k1 (2,3), k2 (4,5), k3 (6,7).
        So we have 1 * 2 * 4 = 8 blocks.
        Each block computes C[i, k_block] = A[i, j_block] * B[j_block, k_block].
        Specifically, for a fixed j_block and k_block, we compute:
        C_sub[i, k_local] = sum_{j_local} A[i, j_block_start + j_local] * B[j_block_start + j_local, k_block_start + k_local]
        This is exactly a 3x2 by 2x2 multiplication.
        
        So we need the rank-9 decomposition for 3x2x2.
        Since R(3,2,2) = R(2,2,3), we can use the known rank-9 decomposition for 2x2x3.
        
        Let's construct the rank-9 decomposition for 2x2x3.
        A (2x2), B (2x3), C (2x3).
        Split B into B1 (2x2) and B2 (2x1).
        C1 = A * B1 (2x2x2, Strassen, rank 7).
        C2 = A * B2 (2x2x1, Schoolbook, rank 2).
        Total rank 9.
        
        Factors for 2x2x2 (Strassen):
        U_s (4x7), V_s (4x7), W_s (4x7).
        Standard Strassen factors for 2x2x2:
        Indices for U: (i, j) -> i*m+j. m=2.
        Indices for V: (j, k) -> j*p+k. p=2.
        Indices for W: (k, i) -> k*n+i. n=2.
        
        Strassen factors (standard):
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B11)
        M4 = A22(B12-B22)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Let's write U, V, W for Strassen 2x2x2.
        U rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3.
        V rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3.
        W rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3.
        
        Col 0 (M1):
        U: A11, A22 -> U[0]=1, U[3]=1.
        V: B11, B22 -> V[0]=1, V[3]=1.
        W: C11, C22 -> W[0]=1, W[3]=1.
        
        Col 1 (M2):
        U: A21, A22 -> U[2]=1, U[3]=1.
        V: B11 -> V[0]=1.
        W: C21, C22 -> W[2]=1, W[3]=-1.
        
        Col 2 (M3):
        U: A11 -> U[0]=1.
        V: B21, B11 -> V[2]=1, V[0]=-1.
        W: C12, C22 -> W[1]=1, W[3]=1.
        
        Col 3 (M4):
        U: A22 -> U[3]=1.
        V: B12, B22 -> V[1]=1, V[3]=-1.
        W: C11, C21 -> W[0]=1, W[2]=1.
        
        Col 4 (M5):
        U: A11, A12 -> U[0]=1, U[1]=1.
        V: B22 -> V[3]=1.
        W: C11, C12 -> W[0]=-1, W[1]=1.
        
        Col 5 (M6):
        U: A21, A11 -> U[2]=1, U[0]=-1.
        V: B11, B12 -> V[0]=1, V[1]=1.
        W: C22 -> W[3]=1.
        
        Col 6 (M7):
        U: A12, A22 -> U[1]=1, U[3]=-1.
        V: B21, B22 -> V[2]=1, V[3]=1.
        W: C11 -> W[0]=1.
        
        Now for 2x2x1 (Schoolbook):
        A (2x2), B (2x1), C (2x1).
        C0 = A00*B0 + A10*B1
        C1 = A01*B0 + A11*B1
        Rank 2.
        Col 0:
        U: A00, A10 -> U[0]=1, U[2]=1.
        V: B0 -> V[0]=1.
        W: C0 -> W[0]=1.
        Col 1:
        U: A01, A11 -> U[1]=1, U[3]=1.
        V: B1 -> V[1]=1.
        W: C1 -> W[1]=1.
        
        Combining for 2x2x3 (rank 9):
        The "3" dimension is k.
        B is 2x3. B1 is 2x2 (k=0,1), B2 is 2x1 (k=2).
        So for the first 7 columns (Strassen), V only has non-zeros for k=0,1.
        For the last 2 columns (Schoolbook), V has non-zeros for k=2.
        
        Let's assemble the 9x(4,6,6) factors for 2x2x3.
        Wait, U shape (2*2, 9) = (4,9).
        V shape (2*3, 9) = (6,9).
        W shape (3*2, 9) = (6,9).
        
        U (4x9):
        Cols 0-6: Strassen U.
        Cols 7-8: Schoolbook U.
        
        V (6x9):
        Rows: (j,k). j in 0..1, k in 0..2.
        Index = j*3 + k.
        Cols 0-6: Strassen V mapped to k=0,1.
        Strassen V rows: (j,k) with k in 0..1.
        So V_strassen[j, k] goes to V_global[j*3 + k].
        Cols 7-8: Schoolbook V mapped to k=2.
        Schoolbook V rows: (j, k_local) with k_local=0 (which is global k=2).
        So V_schoolbook[j, 0] goes to V_global[j*3 + 2].
        
        W (6x9):
        Rows: (k,i). k in 0..2, i in 0..1.
        Index = k*2 + i.
        Cols 0-6: Strassen W mapped to k=0,1.
        Strassen W rows: (k,i) with k in 0..1.
        So W_strassen[k, i] goes to W_global[k*2 + i].
        Cols 7-8: Schoolbook W mapped to k=2.
        Schoolbook W rows: (k_local, i) with k_local=0 (global k=2).
        So W_schoolbook[0, i] goes to W_global[2*2 + i] = W_global[4+i].
        
        Now, we need to map this 2x2x3 decomposition to the 3x2x2 block.
        The 3x2x2 block has:
        A: 3x2 (i in 0..2, j in 0..1)
        B: 2x2 (j in 0..1, k in 0..1)
        C: 3x2 (i in 0..2, k in 0..1)
        
        This is NOT 2x2x3.
        R(3,2,2) = R(2,2,3) by permuting indices?
        T_{i,j,k} for 3x2x2.
        T_{j,k,i} for 2x2x3.
        Yes, they are isomorphic.
        So we can use the 2x2x3 factors, but we need to permute the indices.
        
        Let's define the mapping.
        In 2x2x3:
        U indices: (i', j') where i', j' in 0..1.
        V indices: (j', k') where j' in 0..1, k' in 0..2.
        W indices: (k', i') where k' in 0..2, i' in 0..1.
        
        In 3x2x2:
        U indices: (i, j) where i in 0..2, j in 0..1.
        V indices: (j, k) where j in 0..1, k in 0..1.
        W indices: (k, i) where k in 0..1, i in 0..2.
        
        Wait, the dimensions don't match directly for a simple permutation.
        2x2x3 has dimensions (2,2,3).
        3x2x2 has dimensions (3,2,2).
        They are permutations of each other.
        Specifically, (i', j', k') in 2x2x3 corresponds to (k', i', j') in 3x2x2?
        Let's check the tensor T.
        T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        
        If we take the 2x2x3 decomposition:
        T'_{i',j',k'} = sum_r U'[i',j',r] V'[j',k',r] W'[k',i',r].
        
        We want T_{i,j,k} for i in 0..2, j in 0..1, k in 0..1.
        Let i' = k, j' = i, k' = j.
        Then T'_{k,i,j} = sum_r U'[k,i,r] V'[i,j,r] W'[j,k,r].
        
        We want T_{i,j,k} = T'_{k,i,j}.
        So we set:
        U[i,j,r] = U'[k,i,r]? No, U depends on i,j.
        In T_{i,j,k}, U is associated with (i,j).
        In T'_{k,i,j}, U' is associated with (k,i).
        So U[i,j,r] should be U'[k,i,r]? But k is not in U's indices.
        
        Let's restart the mapping.
        We have a decomposition for 2x2x3.
        We want to use it for 3x2x2.
        The tensor for 3x2x2 is T_{i,j,k} with i in 0..2, j in 0..1, k in 0..1.
        The tensor for 2x2x3 is T'_{a,b,c} with a in 0..1, b in 0..1, c in 0..2.
        
        We can identify T_{i,j,k} with T'_{j,k,i}.
        Let's check:
        T'_{j,k,i} = sum_r U'[j,k,r] V'[k,i,r] W'[i,j,r].
        We want this to equal T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        
        So we need:
        U[i,j,r] = W'[i,j,r]
        V[j,k,r] = U'[j,k,r]
        W[k,i,r] = V'[k,i,r]
        
        Let's check the indices.
        U'[j,k,r]: j in 0..1, k in 0..1. But U' is for 2x2x3, so U' indices are (a,b) with a,b in 0..1.
        Wait, U' is associated with (a,b).
        In T'_{a,b,c}, U' is (a,b).
        So U'[j,k,r] means a=j, b=k.
        This matches V[j,k,r] in T_{i,j,k}.
        
        V'[k,i,r]: V' is associated with (b,c).
        So V'[k,i,r] means b=k, c=i.
        This matches W[k,i,r] in T_{i,j,k}.
        
        W'[i,j,r]: W' is associated with (c,a).
        So W'[i,j,r] means c=i, a=j.
        This matches U[i,j,r] in T_{i,j,k}.
        
        So the mapping is:
        U_block[i,j,r] = W'_global[i,j,r]
        V_block[j,k,r] = U'_global[j,k,r]
        W_block[k,i,r] = V'_global[k,i,r]
        
        Where U', V', W' are the factors for 2x2x3.
        
        Now, U' has shape (2*2, 9) = (4,9). Indices (a,b).
        V' has shape (2*3, 9) = (6,9). Indices (b,c).
        W' has shape (3*2, 9) = (6,9). Indices (c,a).
        
        For the 3x2x2 block:
        U_block has shape (3*2, 9) = (6,9). Indices (i,j).
        V_block has shape (2*2, 9) = (4,9). Indices (j,k).
        W_block has shape (2*3, 9) = (6,9). Indices (k,i).
        
        Let's map the indices.
        U_block[i,j,r] = W'[c=i, a=j, r].
        W' indices (c,a). c in 0..2, a in 0..1.
        So W' row index = c*2 + a = i*2 + j.
        So U_block[i,j,r] = W'[i*2+j, r].
        
        V_block[j,k,r] = U'[a=j, b=k, r].
        U' indices (a,b). a in 0..1, b in 0..1.
        So U' row index = a*2 + b = j*2 + k.
        So V_block[j,k,r] = U'[j*2+k, r].
        
        W_block[k,i,r] = V'[b=k, c=i, r].
        V' indices (b,c). b in 0..1, c in 0..2.
        So V' row index = b*3 + c = k*3 + i.
        So W_block[k,i,r] = V'[k*3+i, r].
        
        This looks correct.
        
        Now, we assemble the global 3x4x8 decomposition.
        Total rank = 8 * 9 = 72.
        
        Global U: shape (3*4, 72) = (12, 72).
        Global V: shape (4*8, 72) = (32, 72).
        Global W: shape (8*3, 72) = (24, 72).
        
        We have 8 blocks.
        j_block in 0..1 (j_start = 2*j_block).
        k_block in 0..3 (k_start = 2*k_block).
        
        For each block, we have 9 columns.
        col_start = (j_block * 4 + k_block) * 9.
        
        For i in 0..2:
        For j_local in 0..1:
        For k_local in 0..1:
        For r_local in 0..8:
            
            global_i = i
            global_j = j_start + j_local
            global_k = k_start + k_local
            
            global_r = col_start + r_local
            
            U_global[global_i * 4 + global_j][global_r] += U_block[i, j_local][r_local]
            V_global[global_j * 8 + global_k][global_r] += V_block[j_local, k_local][r_local]
            W_global[global_k * 3 + global_i][global_r] += W_block[k_local, i][r_local]
            
        We need to construct U_block, V_block, W_block from U', V', W'.
        
        U_block[i, j_local][r] = W'[i*2 + j_local, r]
        V_block[j_local, k_local][r] = U'[j_local*2 + k_local, r]
        W_block[k_local, i][r] = V'[k_local*3 + i, r]
        
        Let's implement this.
        """
        rank = 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Construct 2x2x3 factors (U', V', W')
        # U' (4x9), V' (6x9), W' (6x9)
        Up = [[0] * 9 for _ in range(4)]
        Vp = [[0] * 9 for _ in range(6)]
        Wp = [[0] * 9 for _ in range(6)]
        
        # Strassen 2x2x2 (7 cols)
        # Col 0
        Up[0][0] = 1; Up[3][0] = 1
        Vp[0][0] = 1; Vp[3][0] = 1
        Wp[0][0] = 1; Wp[3][0] = 1
        
        # Col 1
        Up[2][1] = 1; Up[3][1] = 1
        Vp[0][1] = 1
        Wp[2][1] = 1; Wp[3][1] = -1
        
        # Col 2
        Up[0][2] = 1
        Vp[2][2] = 1; Vp[0][2] = -1
        Wp[1][2] = 1; Wp[3][2] = 1
        
        # Col 3
        Up[3][3] = 1
        Vp[1][3] = 1; Vp[3][3] = -1
        Wp[0][3] = 1; Wp[2][3] = 1
        
        # Col 4
        Up[0][4] = 1; Up[1][4] = 1
        Vp[3][4] = 1
        Wp[0][4] = -1; Wp[1][4] = 1
        
        # Col 5
        Up[2][5] = 1; Up[0][5] = -1
        Vp[0][5] = 1; Vp[1][5] = 1
        Wp[3][5] = 1
        
        # Col 6
        Up[1][6] = 1; Up[3][6] = -1
        Vp[2][6] = 1; Vp[3][6] = 1
        Wp[0][6] = 1
        
        # Schoolbook 2x2x1 (2 cols)
        # Col 7
        Up[0][7] = 1; Up[2][7] = 1
        Vp[2][7] = 1  # j=0, k=2 -> index 0*3+2=2
        Wp[4][7] = 1  # k=2, i=0 -> index 2*2+0=4
        
        # Col 8
        Up[1][8] = 1; Up[3][8] = 1
        Vp[5][8] = 1  # j=1, k=2 -> index 1*3+2=5
        Wp[5][8] = 1  # k=2, i=1 -> index 2*2+1=5
        
        # Assemble global
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                col_start = (j_block * 4 + k_block) * 9
                for i in range(3):
                    for j_local in range(2):
                        global_j = j_start + j_local
                        # U_block[i, j_local][r] = Wp[i*2 + j_local, r]
                        u_row = i * 4 + global_j
                        w_row = i * 2 + j_local
                        for r_local in range(9):
                            if Wp[w_row][r_local] != 0:
                                U[u_row][col_start + r_local] += Wp[w_row][r_local]
                        
                        for k_local in range(2):
                            global_k = k_start + k_local
                            # V_block[j_local, k_local][r] = Up[j_local*2 + k_local, r]
                            v_row = global_j * 8 + global_k
                            u_row_p = j_local * 2 + k_local
                            for r_local in range(9):
                                if Up[u_row_p][r_local] != 0:
                                    V[v_row][col_start + r_local] += Up[u_row_p][r_local]
                            
                            # W_block[k_local, i][r] = Vp[k_local*3 + i, r]
                            w_row_global = global_k * 3 + i
                            v_row_p = k_local * 3 + i
                            for r_local in range(9):
                                if Vp[v_row_p][r_local] != 0:
                                    W[w_row_global][col_start + r_local] += Vp[v_row_p][r_local]
        
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try rank 72
    if time.time() < deadline - 10:
        res = build_rank72()
        if res and verify(res):
            return res
            
    # Fallback
    return build_schoolbook()
