import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 3, 4, 8

    def schoolbook():
        rank_sb = n * m * p  # 96
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, idx] = 1
                    V_sb[j * p + k, idx] = 1
                    W_sb[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    def verify(U, V, W, rank, tol=1e-9):
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
                return False
            U_3d = U_np.reshape(n, m, rank)
            V_3d = V_np.reshape(m, p, rank)
            W_3d = W_np.reshape(p, n, rank)
            rng = np.random.default_rng(12345)
            for _ in range(5):
                A = rng.standard_normal((n, m))
                B = rng.standard_normal((m, p))
                C = A @ B
                M = U_3d * A[:, :, np.newaxis]
                N = V_3d * B[:, :, np.newaxis]
                S = np.einsum('ijc,jkc->ikc', M, N)
                S_T = np.transpose(S, (1, 0, 2))
                C_dec = np.sum(S_T * W_3d, axis=2)
                if np.max(np.abs(C - C_dec)) > tol:
                    return False
            return True
        except:
            return False

    def to_clean_list(mat_np):
        result = []
        for row in mat_np:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(float(v))
            result.append(new_row)
        return result

    # Strategy: Deterministic Block Decomposition
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x8) into B1 (2x8) and B2 (2x8)
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (3x2 * 2x8):
    # Split A1 into A1_top (2x2) and A1_bot (1x2)
    # A1*B1 = A1_top*B1 + A1_bot*B1
    # A1_top*B1 (2x2 * 2x8) can be done with Strassen rank 7 * 4 (for the 4 columns of 2x2 blocks) = 28
    # A1_bot*B1 (1x2 * 2x8) is just a dot product for each of the 8 columns, rank 16 (schoolbook)
    # Total rank for A1*B1 = 28 + 16 = 44
    
    # For A2*B2 (3x2 * 2x8):
    # Same structure: 28 + 16 = 44
    
    # Total rank = 44 + 44 = 88
    
    # Let's construct the rank 88 decomposition explicitly.
    
    # Helper: Get Strassen decomposition for 2x2 * 2x2
    # Returns U (4x7), V (4x7), W (4x7) such that sum U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,k} (wait, W is for output)
    # Actually, the tensor is T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B.
    # Output C[i,k] = sum_j A[i,j] B[j,k].
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to map this to the U, V, W matrices.
    # U[i, j, c], V[j, k, c], W[k, i, c]
    # Note: In the problem statement, W has shape (p*n, rank), indexed by W[k*n+i, c].
    # So W_3d[k, i, c] corresponds to W_flat[k*n+i, c].
    
    def get_strassen_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7)
        # Indices for 2x2:
        # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7)) # W[k, i] where k is col of B, i is row of A
        
        # c=0: M1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22 (Note: B indices 0,1 are row 0, 2,3 are row 1? No.
        # Let's stick to standard matrix indexing:
        # A[i,j], B[j,k]
        # j=0,1 for 2x2.
        # B[j,k] where j is row of B (col of A), k is col of B.
        # So V[j, k, c].
        # B11 is B[0,0], B12 is B[0,1], B21 is B[1,0], B22 is B[1,1].
        
        V[0, 0, 0] = 1; V[1, 0, 0] = 1 # B[0,0], B[1,0] -> Wait, M1 uses (B11+B22).
        # M1 = (A11+A22)(B11+B22).
        # A11=A[0,0], A22=A[1,1].
        # B11=B[0,0], B22=B[1,1].
        # So U[0,0]=1, U[3,0]=1 (if we flatten A as 00,01,10,11).
        # V[0,0]=1, V[3,0]=1 (if we flatten B as 00,01,10,11).
        # Output C11 = M1...
        # C11 is C[0,0]. So W[0,0] gets +1.
        
        # Let's re-index carefully.
        # A flat: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B flat: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C flat: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        # C(0,0) += M1, C(1,1) += M1
        W[0,0,0]=1; W[3,1,0]=1 # W[k,i] -> W_flat[k*2+i]
        
        # M2: A(1,0)+A(1,1) * B(0,0)
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        # C(1,0) += M2
        W[0,1,1]=1
        
        # M3: A(0,0) * (B(1,0)-B(1,1))
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        # C(0,1) += M3
        W[1,0,2]=1
        
        # M4: A(1,1) * (B(0,1)-B(0,0))
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        # C(0,0) += M4, C(1,0) += M4
        W[0,0,3]=1; W[0,1,3]=1
        
        # M5: (A(0,0)+A(0,1)) * B(1,1)
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        # C(0,0) -= M5, C(0,1) += M5
        W[0,0,4]=-1; W[1,0,4]=1
        
        # M6: (A(1,0)-A(0,0)) * (B(0,0)+B(0,1))
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        # C(1,1) += M6
        W[3,1,5]=1
        
        # M7: (A(0,1)-A(1,1)) * (B(1,0)+B(1,1))
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        # C(0,0) += M7, C(1,1) -= M7
        W[0,0,6]=1; W[3,1,6]=-1
        
        return U, V, W

    def get_schoolbook_1x2x2():
        # A is 1x2, B is 2x2. Output 1x2.
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # Rank 4.
        U = np.zeros((2, 4)) # A flat: 0:(0,0), 1:(0,1)
        V = np.zeros((4, 4)) # B flat: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        W = np.zeros((2, 4)) # C flat: 0:(0,0), 1:(0,1). W[k,i] -> i is always 0.
        
        # Term 1: A[0,0]B[0,0] -> C[0,0]
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 2: A[0,1]B[1,0] -> C[0,0]
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        # Term 3: A[0,0]B[0,1] -> C[0,1]
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        # Term 4: A[0,1]B[1,1] -> C[0,1]
        U[1,3]=1; V[3,3]=1; W[1,3]=1
        
        return U, V, W

    # Construct 3x2 * 2x8 decomposition
    # A is 3x2. Split into A_top (2x2) and A_bot (1x2).
    # B is 2x8. Split into 4 blocks of 2x2: B_0, B_1, B_2, B_3.
    # C is 3x8.
    
    # Rank for A_top * B_block (2x2 * 2x2) is 7.
    # Rank for A_bot * B_block (1x2 * 2x2) is 4.
    # Total rank per 2x2 block of B is 7 + 4 = 11.
    # 4 blocks -> 44.
    
    def get_3x2x2_decomp():
        # Returns U (6x11), V (4x11), W (4x11)
        # A flat (3x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
        # B flat (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C flat (3x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
        # W flat (2x3): W[k*n+i] where n=3? No, for the subproblem, output is 3x2.
        # The global W is (p*n, rank). Here p_sub=2, n_sub=3.
        # So W_sub has shape (6, 11). Index k*3+i.
        
        U_sub = np.zeros((6, 11))
        V_sub = np.zeros((4, 11))
        W_sub = np.zeros((6, 11))
        
        # Top part (rows 0,1 of A)
        U_top, V_top, W_top = get_strassen_2x2x2()
        # U_top is 4x7 (indices 0..3 of A). Maps to 0..3 of U_sub.
        # V_top is 4x7. Maps to 0..3 of V_sub.
        # W_top is 4x7 (indices k*2+i for 2x2 output).
        # Output of top part is 2x2 (rows 0,1 of C).
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # W_sub indices for these: k*3+i.
        # k=0, i=0 -> 0
        # k=1, i=0 -> 3
        # k=0, i=1 -> 1
        # k=1, i=1 -> 4
        
        for c in range(7):
            U_sub[0:4, c] = U_top[:, c]
            V_sub[:, c] = V_top[:, c]
            # Map W_top[k, i] to W_sub[k*3+i]
            W_sub[0, c] += W_top[0, 0] # k=0,i=0
            W_sub[3, c] += W_top[0, 1] # k=0,i=1
            W_sub[1, c] += W_top[1, 0] # k=1,i=0
            W_sub[4, c] += W_top[1, 1] # k=1,i=1
            
        # Bottom part (row 2 of A)
        U_bot, V_bot, W_bot = get_schoolbook_1x2x2()
        # U_bot is 2x4 (indices 0,1 of A_bot). Maps to 4,5 of U_sub.
        # V_bot is 4x4. Maps to 0..3 of V_sub (offset 7).
        # W_bot is 2x4 (indices k*1+i? No, 1x2 output).
        # Output of bot part is 1x2 (row 2 of C).
        # C indices: 4:(2,0), 5:(2,1).
        # W_sub indices: k*3+2.
        # k=0 -> 2
        # k=1 -> 5
        
        for c in range(4):
            U_sub[4:6, 7+c] = U_bot[:, c]
            V_sub[:, 7+c] = V_bot[:, c]
            W_sub[2, 7+c] += W_bot[0, c] # k=0, i=2 (in sub-coord i=0) -> global i=2
            W_sub[5, 7+c] += W_bot[1, c] # k=1, i=2 (in sub-coord i=0) -> global i=2
            
        return U_sub, V_sub, W_sub

    # Now assemble 3x4 * 4x8
    # A (3x4) -> A1 (3x2, cols 0,1), A2 (3x2, cols 2,3)
    # B (4x8) -> B1 (2x8, rows 0,1), B2 (2x8, rows 2,3)
    # C (3x8)
    
    # We have a decomp for 3x2 * 2x8 (rank 44).
    # We need to apply it twice: once for A1*B1 and once for A2*B2.
    # A1*B1 uses cols 0,1 of A and rows 0,1 of B.
    # A2*B2 uses cols 2,3 of A and rows 2,3 of B.
    
    U_full = np.zeros((12, 88))
    V_full = np.zeros((32, 88))
    W_full = np.zeros((24, 88))
    
    U_sub, V_sub, W_sub = get_3x2x2_decomp()
    
    # Block 1: A1 * B1
    # A1 indices in A_full: 0..5 (i*4+j, j=0,1)
    # B1 indices in B_full: 0..15 (j*8+k, j=0,1)
    # C indices in C_full: 0..23 (k*3+i)
    
    for c in range(44):
        # Map U_sub (6x44) to U_full (12x88)
        # U_sub row i corresponds to A[i, 0..1]
        # A_full row i*4 + 0 and i*4 + 1
        for i in range(3):
            U_full[i*4 + 0, c] = U_sub[i*2 + 0, c]
            U_full[i*4 + 1, c] = U_sub[i*2 + 1, c]
            
        # Map V_sub (4x44) to V_full (32x88)
        # V_sub row j corresponds to B[j, 0..1] (j=0,1)
        # B_full row j*8 + k (k=0..7)
        # V_sub is for 2x2 B block.
        # Wait, my get_3x2x2_decomp was for 3x2 * 2x2.
        # I need 3x2 * 2x8.
        # Let's re-verify the logic.
        
    # The logic above was for a single 2x2 block of B.
    # B is 2x8, so it has 4 such blocks.
    # Let's construct the full 3x2 * 2x8 decomposition first.
    
    def get_3x2x8_decomp():
        # Returns U (6x44), V (16x44), W (24x44)
        # A (3x2), B (2x8), C (3x8)
        U_res = np.zeros((6, 44))
        V_res = np.zeros((16, 44))
        W_res = np.zeros((24, 44))
        
        for block_idx in range(4):
            # Offset in rank
            rank_offset = block_idx * 11
            # Offset in B columns (k)
            k_offset = block_idx * 2
            # Offset in C columns (k)
            c_k_offset = block_idx * 2
            
            U_sub, V_sub, W_sub = get_3x2x2_decomp()
            
            # Copy U
            U_res[:, rank_offset:rank_offset+11] = U_sub
            
            # Copy V
            # V_sub is 4x11 (rows 0,1 of B, cols 0,1 of B_block)
            # V_res is 16x44 (rows 0,1 of B, cols 0..7 of B)
            # V_sub row j, col k_sub maps to V_res row j, col k_offset + k_sub
            for j in range(2):
                for k_sub in range(2):
                    V_res[j*8 + k_offset + k_sub, rank_offset:rank_offset+11] = V_sub[j*2 + k_sub, :]
            
            # Copy W
            # W_sub is 6x11 (indices k*3+i for k=0,1 and i=0,1,2)
            # W_res is 24x44 (indices k*3+i for k=0..7 and i=0,1,2)
            # W_sub row k_sub*3+i maps to W_res row (k_offset + k_sub)*3 + i
            for k_sub in range(2):
                for i in range(3):
                    W_res[(k_offset + k_sub)*3 + i, rank_offset:rank_offset+11] = W_sub[k_sub*3 + i, :]
                    
        return U_res, V_res, W_res

    # Now assemble 3x4 * 4x8
    U_328, V_328, W_328 = get_3x2x8_decomp()
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # U_328 (6x44) -> U_full (12x88)
    # A1 rows i, cols 0,1 -> A_full rows i*4+0, i*4+1
    for i in range(3):
        U_full[i*4 + 0, 0:44] = U_328[i*2 + 0, :]
        U_full[i*4 + 1, 0:44] = U_328[i*2 + 1, :]
        
    # V_328 (16x44) -> V_full (32x88)
    # B1 rows 0,1, cols 0..7 -> B_full rows 0..15
    V_full[0:16, 0:44] = V_328
    
    # W_328 (24x44) -> W_full (24x88)
    # C cols 0..7 -> C_full cols 0..7
    W_full[0:24, 0:44] = W_328
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # Rank offset 44
    for i in range(3):
        U_full[i*4 + 2, 44:88] = U_328[i*2 + 0, :]
        U_full[i*4 + 3, 44:88] = U_328[i*2 + 1, :]
        
    # B2 rows 2,3, cols 0..7 -> B_full rows 16..31
    # V_328 row j (0,1) maps to B2 row j, which is global row 2+j
    # V_full row (2+j)*8 + k
    for j in range(2):
        for k in range(8):
            V_full[(2+j)*8 + k, 44:88] = V_328[j*8 + k, :]
            
    # W_328 -> W_full
    # C cols 0..7 -> C_full cols 0..7 (same indices)
    W_full[0:24, 44:88] = W_328
    
    # Verify
    U_list = to_clean_list(U_full)
    V_list = to_clean_list(V_full)
    W_list = to_clean_list(W_full)
    
    if verify(U_list, V_list, W_list, 88):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
