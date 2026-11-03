import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen 2x2 rank-7 decomposition
    # U, V, W for 2x2 * 2x2 = 2x2
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # Indexing: A[i,j], B[i,j], C[i,j] for i,j in {0,1}
    # U[i*2+j, k], V[i*2+j, k], W[i*2+j, k]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_local[0,0]=1; U_local[3,0]=1
    V_local[0,0]=1; V_local[3,0]=1
    W_local[0,0]=1; W_local[3,0]=1
    
    # M2: (A10+A11)B00 -> C10, C11
    U_local[2,1]=1; U_local[3,1]=1
    V_local[0,1]=1
    W_local[2,1]=1; W_local[3,1]=-1
    
    # M3: A00(B01-B11) -> C01, C11
    U_local[0,2]=1
    V_local[1,2]=1; V_local[3,2]=-1
    W_local[1,2]=1; W_local[3,2]=1
    
    # M4: A11(B10-B00) -> C00, C10
    U_local[3,3]=1
    V_local[2,3]=1; V_local[0,3]=-1
    W_local[0,3]=1; W_local[2,3]=1
    
    # M5: (A00+A01)B11 -> C00, C01
    U_local[0,4]=1; U_local[1,4]=1
    V_local[3,4]=1
    W_local[0,4]=-1; W_local[1,4]=1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U_local[2,5]=1; U_local[0,5]=-1
    V_local[0,5]=1; V_local[1,5]=1
    W_local[3,5]=1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U_local[1,6]=1; U_local[3,6]=-1
    V_local[2,6]=1; V_local[3,6]=1
    W_local[0,6]=1
    
    # Try rank 49: 7 groups of rank 7
    # Combine the 8 independent 2x2*2x2 multiplications into 7
    # by exploiting that A has only 2 rows
    
    # A is 2x4: A = [A0 | A1] where A0, A1 are 2x2
    # B is 4x8: B = [B0; B1] where B0, B1 are 2x8
    # C = A0*B0 + A1*B1 (2x8)
    # B0 = [B00 B01 B02 B03], B1 = [B10 B11 B12 B13] (each 2x2)
    # C = [A0B00 + A1B10 | A0B01 + A1B11 | A0B02 + A1B12 | A0B03 + A1B13]
    
    # For each column block j: Cj = A0*B0j + A1*B1j
    # This is a 2x2 * 2x2 + 2x2 * 2x2 = 2x2 computation
    # We can compute this with rank 7 by treating it as a single bilinear form
    
    # Actually, A0*B0j + A1*B1j can be computed by:
    # Let X = [A0; A1] (4x2), Y_j = [B0j; B1j] (4x2)
    # Then Cj = X * Y_j (2x2)
    # This is a 4x2 * 2x2 = 2x2 multiplication, which has rank at most 7*... 
    # Actually this is just 2x2 * 2x2 after summing, so rank 7 suffices per column block.
    # Total: 4 column blocks * 7 = 28? No, that's not right.
    
    # Let me just use the straightforward 56-rank solution for now
    # and try to optimize
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # 2 row-groups for A (2x2 blocks), 4 col-groups for B (2x2 blocks)
    for j_group in range(2):
        for k_group in range(4):
            # A block: rows j_group*2 to j_group*2+1, cols 0 to 1 (2x2)
            # Wait, A is 2x4, so A[i, j_group*2 + col]
            # B block: rows j_group*2 to j_group*2+1, cols k_group*2 to k_group*2+1
            # C block: rows k_group*2 to k_group*2+1, col... 
            # C is 2x8: C[row, col] where row in {0,1}, col in {0..7}
            # C[row, k_group*2 + ccol] += sum over A[i, j_group*2 + acol] * B[j_group*2 + bcol, k_group*2 + ccol]
            
            base_r = (j_group * 4 + k_group) * 7
            
            # A indices: row i in {0,1}, col j_group*2 + acol
            a_map = [
                0 * 4 + j_group * 2,      # A[0, j_group*2+0]
                0 * 4 + j_group * 2 + 1,  # A[0, j_group*2+1]
                1 * 4 + j_group * 2,      # A[1, j_group*2+0]
                1 * 4 + j_group * 2 + 1   # A[1, j_group*2+1]
            ]
            
            # B indices: row j_group*2 + bcol, col k_group*2 + ccol
            b_map = [
                (j_group * 2) * 8 + k_group * 2,        # B[j_group*2+0, k_group*2+0]
                (j_group * 2) * 8 + k_group * 2 + 1,    # B[j_group*2+0, k_group*2+1]
                (j_group * 2 + 1) * 8 + k_group * 2,    # B[j_group*2+1, k_group*2+0]
                (j_group * 2 + 1) * 8 + k_group * 2 + 1 # B[j_group*2+1, k_group*2+1]
            ]
            
            # C indices: row ccol, col k_group*2 + ccol... 
            # C is 2x8: C[row, col], row in {0,1}, col in {0..7}
            # W index: k*n + i where k is col, i is row? Or k*n+i?
            # Task says W[k*n+i, r] where n=2, so W[col*2+row, r]
            # C[row, col] -> W[col*2 + row, r]
            c_map = [
                (k_group * 2) * 2 + 0,      # C[0, k_group*2+0]
                (k_group * 2 + 1) * 2 + 0,  # C[0, k_group*2+1]
                (k_group * 2) * 2 + 1,      # C[1, k_group*2+0]
                (k_group * 2 + 1) * 2 + 1   # C[1, k_group*2+1]
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
