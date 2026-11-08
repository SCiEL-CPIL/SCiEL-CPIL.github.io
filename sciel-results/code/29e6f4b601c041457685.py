"""Matrix multiplication 3,3,8 with Strassen block decomposition."""
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_2x2x2_strassen():
    """
    Build Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where:
      U has shape (4, 7), indexed by [i*2+j, r]
      V has shape (4, 7), indexed by [j*2+k, r]
      W has shape (4, 7), indexed by [k*2+i, r]
    
    Strassen's 7 products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
      
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n = 2
    rank = 7
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # A indices: A00=(0,0)=idx0, A01=(0,1)=idx1, A10=(1,0)=idx2, A11=(1,1)=idx3
    # B indices: B00=(0,0)=idx0, B01=(0,1)=idx1, B10=(1,0)=idx2, B11=(1,1)=idx3
    # C indices: C00=(0,0)=idx0, C01=(0,1)=idx1, C10=(1,0)=idx2, C11=(1,1)=idx3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1 # B11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B01-B00)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[2][5] = 1  # B10
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[1][6] = 1  # B01
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    return U, V, W


def _build_schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m by m x p.
    Returns (U, V, W) with rank = n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def _combine_decompositions(decomp_list):
    """
    Combine multiple decompositions into one.
    Each decomp is (U, V, W) with same rank r.
    Returns combined (U, V, W) with rank = sum of ranks.
    """
    total_rank = sum(d[0][0].__len__() if len(d[0]) > 0 else 0 for d in decomp_list)
    # More robust: get rank from first row
    total_rank = 0
    for U, V, W in decomp_list:
        if len(U) > 0 and len(U[0]) > 0:
            total_rank += len(U[0])
    
    # Get dimensions from first decomposition
    if not decomp_list:
        return [], [], []
    
    U0, V0, W0 = decomp_list[0]
    dim_U = len(U0)
    dim_V = len(V0)
    dim_W = len(W0)
    
    U = [[0] * total_rank for _ in range(dim_U)]
    V = [[0] * total_rank for _ in range(dim_V)]
    W = [[0] * total_rank for _ in range(dim_W)]
    
    offset = 0
    for U_d, V_d, W_d in decomp_list:
        r = len(U_d[0]) if len(U_d) > 0 else 0
        for i in range(dim_U):
            for c in range(r):
                U[i][offset + c] = U_d[i][c]
        for j in range(dim_V):
            for c in range(r):
                V[j][offset + c] = V_d[j][c]
        for k in range(dim_W):
            for c in range(r):
                W[k][offset + c] = W_d[k][c]
        offset += r
    
    return U, V, W


def construct():
    """
    Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication.
    
    Uses block decomposition:
    - Split the 3x3 matrix into blocks:
      A = [[A_top, A_right], [A_bottom, ...]]
    
    Strategy: Decompose 3x3 x 3x8 as:
    - 2x2 x 3x8 part (using Strassen on the 2x2 block) + 1x3 x 3x8 part (schoolbook)
    
    More precisely, split the 3x3 matrix as:
    A = [[A_2x2, A_2x1], [A_1x2, A_1x1]]
    
    B is 3x8.
    
    C = A @ B
    
    We can write:
    C = [A_2x2 @ B_top2 + A_2x1 @ B_bottom1]
        [A_1x2 @ B_top2 + A_1x1 @ B_bottom1]
    
    where B_top2 is the first 2 rows of B (2x8) and B_bottom1 is the last row (1x8).
    
    So we need:
    1. 2x2 x 3x8: Strassen for the 2x2, then multiply result (2x3) by B_top2 (3x8)
       This is more complex. Let me think differently.
    
    Alternative: Split the 8 columns into 2+2+2+2 or 3+3+2.
    
    Let's split 8 as 3+3+2.
    B = [B1 (3x3), B2 (3x3), B3 (3x2)]
    
    C = [C1, C2, C3] where Ci = A @ Bi
    
    For C1 and C2 (3x3 x 3x3): Use rank-19 decomposition for 3x3x3.
    For C3 (3x3 x 3x2): Use a decomposition for 3x3 x 3x2.
    
    But we don't have a known rank-19 decomposition readily available in code.
    
    Simpler approach: Split 3x3 as 2x2 + 1x3 block.
    
    A = [[A22, A21], [A12, A11]]
    where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1.
    
    B is 3x8. Let B = [Btop (2x8), Bbot (1x8)]
    
    C = [[A22@Btop + A21@Bbot], [A12@Btop + A11@Bbot]]
    
    So we need to compute:
    1. A22 @ Btop: 2x2 @ 2x8 -> Use Strassen-style
    2. A21 @ Bbot: 2x1 @ 1x8 -> Schoolbook, rank 16
    3. A12 @ Btop: 1x2 @ 2x8 -> Schoolbook, rank 16  
    4. A11 @ Bbot: 1x1 @ 1x8 -> Schoolbook, rank 8
    
    For A22 @ Btop (2x2 @ 2x8):
    Split Btop into 4 blocks of 2x2: Btop = [B00, B01, B02, B03]
    Result = [C00, C01, C02, C03] where Ci = A22 @ Bi
    
    Each Ci is 2x2 = Strassen rank 7, so total for this part: 4 * 7 = 28
    
    Total rank = 28 + 16 + 16 + 8 = 68
    
    This matches the expected rank of 68.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Decompose 3x3 x 3x8 using block decomposition
    # A (3x3) split as [[A22 (2x2), A21 (2x1)], [A12 (1x2), A11 (1x1)]]
    # B (3x8) split as [Btop (2x8), Bbot (1x8)]
    # C (3x8) = [Ctop (2x8), Cbot (1x8)]
    # Ctop = A22 @ Btop + A21 @ Bbot
    # Cbot = A12 @ Btop + A11 @ Bbot
    
    decompositions = []
    
    # Part 1: A22 @ Btop where A22 is 2x2, Btop is 2x8
    # Split Btop into 4 blocks of size 2x2
    # For each 2x2 x 2x2, use Strassen (rank 7)
    for b_idx in range(4):
        U_s, V_s, W_s = _build_2x2x2_strassen()
        decompositions.append((U_s, V_s, W_s))
    
    # Part 2: A21 @ Bbot where A21 is 2x1, Bbot is 1x8
    # Schoolbook: rank = 2*1*8 = 16
    U2, V2, W2 = _build_schoolbook(2, 1, 8)
    decompositions.append((U2, V2, W2))
    
    # Part 3: A12 @ Btop where A12 is 1x2, Btop is 2x8
    # Schoolbook: rank = 1*2*8 = 16
    U3, V3, W3 = _build_schoolbook(1, 2, 8)
    decompositions.append((U3, V3, W3))
    
    # Part 4: A11 @ Bbot where A11 is 1x1, Bbot is 1x8
    # Schoolbook: rank = 1*1*8 = 8
    U4, V4, W4 = _build_schoolbook(1, 1, 8)
    decompositions.append((U4, V4, W4))
    
    # Now we need to map these decompositions to the full 3x3 x 3x8 problem
    # The decompositions above are for sub-problems. We need to construct
    # U, V, W for the full problem with the correct indexing.
    
    # Full problem:
    # U: (3*3, rank) = (9, rank), indexed by [i*3+j, r] where A[i,j]
    # V: (3*8, rank) = (24, rank), indexed by [j*8+k, r] where B[j,k]
    # W: (8*3, rank) = (24, rank), indexed by [k*3+i, r] where C[k,i]
    
    # Wait, the convention is W[k*n+i, r] where n=3, so W has 8*3=24 rows.
    
    # Let me reconsider the indexing.
    # A is 3x3: A[i,j] for i in 0..2, j in 0..2
    # B is 3x8: B[j,k] for j in 0..2, k in 0..7
    # C is 3x8: C[i,k] for i in 0..2, k in 0..7
    
    # U: (n*m, rank) = (9, rank), row index = i*m + j = i*3 + j
    # V: (m*p, rank) = (24, rank), row index = j*p + k = j*8 + k
    # W: (p*n, rank) = (24, rank), row index = k*n + i = k*3 + i
    
    # For the block decomposition:
    # A22: A[0:2, 0:2], so i in {0,1}, j in {0,1}
    # A21: A[0:2, 2], so i in {0,1}, j = 2
    # A12: A[2, 0:2], so i = 2, j in {0,1}
    # A11: A[2, 2], so i = 2, j = 2
    
    # Btop: B[0:2, :], so j in {0,1}, k in 0..7
    # Bbot: B[2, :], so j = 2, k in 0..7
    
    # Ctop: C[0:2, :], so i in {0,1}, k in 0..7
    # Cbot: C[2, :], so i = 2, k in 0..7
    
    # Part 1: A22 @ Btop
    # A22[i,j] for i,j in {0,1}
    # Btop[j,k] for j in {0,1}, k in 0..7
    # Result: Ctop[i,k] for i in {0,1}, k in 0..7
    
    # We split Btop into 4 blocks of 2x2: Btop[:, 0:2], Btop[:, 2:4], Btop[:, 4:6], Btop[:, 6:8]
    # For block b (0..3): Btop[:, b*2:(b+1)*2]
    
    # For each Strassen decomposition for 2x2 x 2x2:
    # U_s: (4, 7), row = i_sub * 2 + j_sub where i_sub, j_sub in {0,1}
    # V_s: (4, 7), row = j_sub * 2 + k_sub where j_sub, k_sub in {0,1}
    # W_s: (4, 7), row = k_sub * 2 + i_sub where k_sub, i_sub in {0,1}
    
    # Mapping to full indices:
    # U_s row (i_sub*2+j_sub) -> full U row (i_sub*3 + j_sub) since i_sub=i, j_sub=j for A22
    # V_s row (j_sub*2+k_sub) -> full V row (j_sub*8 + (b*2 + k_sub))
    # W_s row (k_sub*2+i_sub) -> full W row ((b*2 + k_sub)*3 + i_sub)
    
    # Part 2: A21 @ Bbot
    # A21[i] for i in {0,1} (this is A[i,2])
    # Bbot[k] for k in 0..7 (this is B[2,k])
    # Result: Ctop[i,k] for i in {0,1}, k in 0..7
    
    # Schoolbook for 2x1 x 1x8:
    # U2: (2, 16), row = i*1 + 0 = i (for A21[i])
    # V2: (8, 16), row = 0*8 + k = k (for Bbot[k])
    # W2: (16, 16), row = k*2 + i
    
    # Mapping:
    # U2 row i -> full U row (i*3 + 2)
    # V2 row k -> full V row (2*8 + k)
    # W2 row (k*2+i) -> full W row (k*3 + i)
    
    # Part 3: A12 @ Btop
    # A12[j] for j in {0,1} (this is A[2,j])
    # Btop[j,k] for j in {0,1}, k in 0..7
    # Result: Cbot[k] for k in 0..7 (this is C[2,k])
    
    # Schoolbook for 1x2 x 2x8:
    # U3: (2, 16), row = 0*2 + j = j (for A12[j])
    # V3: (16, 16), row = j*8 + k
    # W3: (16, 16), row = k*1 + 0 = k
    
    # Mapping:
    # U3 row j -> full U row (2*3 + j)
    # V3 row (j*8+k) -> full V row (j*8 + k)
    # W3 row k -> full W row (k*3 + 2)
    
    # Part 4: A11 @ Bbot
    # A11 (scalar A[2,2])
    # Bbot[k] for k in 0..7
    # Result: Cbot[k] for k in 0..7
    
    # Schoolbook for 1x1 x 1x8:
    # U4: (1, 8)
    # V4: (8, 8)
    # W4: (8, 8)
    
    # Mapping:
    # U4 row 0 -> full U row (2*3 + 2) = 8
    # V4 row k -> full V row (2*8 + k)
    # W4 row k -> full W row (k*3 + 2)
    
    total_rank = 4 * 7 + 16 + 16 + 8  # 28 + 16 + 16 + 8 = 68
    
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(24)]
    W = [[0] * total_rank for _ in range(24)]
    
    col = 0
    
    # Part 1: 4 Strassen decompositions
    for b_idx in range(4):
        U_s, V_s, W_s = _build_2x2x2_strassen()
        for r in range(7):
            for i_sub in range(2):
                for j_sub in range(2):
                    if U_s[i_sub * 2 + j_sub][r] != 0:
                        full_u_row = i_sub * 3 + j_sub
                        U[full_u_row][col] = U_s[i_sub * 2 + j_sub][r]
            for j_sub in range(2):
                for k_sub in range(2):
                    if V_s[j_sub * 2 + k_sub][r] != 0:
                        full_v_row = j_sub * 8 + (b_idx * 2 + k_sub)
                        V[full_v_row][col] = V_s[j_sub * 2 + k_sub][r]
            for k_sub in range(2):
                for i_sub in range(2):
                    if W_s[k_sub * 2 + i_sub][r] != 0:
                        full_w_row = (b_idx * 2 + k_sub) * 3 + i_sub
                        W[full_w_row][col] = W_s[k_sub * 2 + i_sub][r]
            col += 1
    
    # Part 2: Schoolbook 2x1 x 1x8
    U2, V2, W2 = _build_schoolbook(2, 1, 8)
    for r in range(16):
        for i in range(2):
            if U2[i][r] != 0:
                full_u_row = i * 3 + 2
                U[full_u_row][col] = U2[i][r]
        for k in range(8):
            if V2[k][r] != 0:
                full_v_row = 2 * 8 + k
                V[full_v_row][col] = V2[k][r]
        for idx in range(16):
            if W2[idx][r] != 0:
                # idx = k*2 + i
                k = idx // 2
                i = idx % 2
                full_w_row = k * 3 + i
                W[full_w_row][col] = W2[idx][r]
        col += 1
    
    # Part 3: Schoolbook 1x2 x 2x8
    U3, V3, W3 = _build_schoolbook(1, 2, 8)
    for r in range(16):
        for j in range(2):
            if U3[j][r] != 0:
                full_u_row = 2 * 3 + j
                U[full_u_row][col] = U3[j][r]
        for j in range(2):
            for k in range(8):
                if V3[j * 8 + k][r] != 0:
                    full_v_row = j * 8 + k
                    V[full_v_row][col] = V3[j * 8 + k][r]
        for k in range(8):
            if W3[k][r] != 0:
                full_w_row = k * 3 + 2
                W[full_w_row][col] = W3[k][r]
        col += 1
    
    # Part 4: Schoolbook 1x1 x 1x8
    U4, V4, W4 = _build_schoolbook(1, 1, 8)
    for r in range(8):
        if U4[0][r] != 0:
            U[8][col] = U4[0][r]
        for k in range(8):
            if V4[k][r] != 0:
                full_v_row = 2 * 8 + k
                V[full_v_row][col] = V4[k][r]
        for k in range(8):
            if W4[k][r] != 0:
                full_w_row = k * 3 + 2
                W[full_w_row][col] = W4[k][r]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
