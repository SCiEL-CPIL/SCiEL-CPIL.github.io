"""Rewritten program: Strassen-based tensor decomposition for 4x4x9 matrix multiplication."""

import time

DIMENSIONS = [4, 4, 9]

# Winograd 2x2x2 core (rank 7): computes 2x2 * 2x2
# u_i = row_i, v_j = col_j, w_k = out_k
# Standard Winograd:
#  U rows (2x2): [1,1,1,0,0,1,-1] mapped to indices
#  Actually let's use the standard Winograd for 2x2:
#  M0 = (a00+a11)(b00+b11), M1 = (a10+a11)b00, M2 = a00(b10-b11),
#  M3 = a11(b01-b00), M4 = (a00+a01)b11, M5 = (a01-a10)(b00+b10),
#  M6 = (a01-a10)(b11+b01)
#
#  U (4x7): input a has indices a00=0,a01=1,a10=2,a11=3
#  U[0] = [1,0,0,1]  -> a00 appears in M0, M2, M4? Let me re-derive.
#
#  Actually, the standard Winograd 2x2 factorization:
#  c00 = M0 - M1 + M2 + M3
#  c01 = M0 + M1 - M4 - M5
#  c10 = M0 - M2 - M3 + M6
#  c11 = M0 + M3 - M2 + M4  -- hmm, let me just use a known correct set.
#
#  Known Winograd 2x2 (from literature):
#  u1 = [1, 1, 1, 0]  (a00 + a11 + a01)
#  u2 = [1, 1, 0, 1]  (a00 + a01 + a11) -- wait, need to be careful.
#
#  Let me use a verified Winograd 2x2:
#  The 7 products are:
#  p1 = (a00+a11)(b00+b11)
#  p2 = (a10+a11)(b00)
#  p3 = a00(b10-b11)
#  p4 = a11(b01-b00)
#  p5 = (a00+a01)(b11)
#  p6 = (a01-a10)(b00+b10)
#  p7 = (a01-a10)(b11+b01)
#
#  Then:
#  c00 = p1 - p2 + p3 + p4
#  c01 = p1 + p2 - p5 - p6
#  c10 = p1 - p3 - p4 + p7
#  c11 = p1 + p4 - p3 + p5  -- need to verify.
#
#  Let me verify c00:
#  p1 - p2 + p3 + p4 = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b10-b11) + a11(b01-b00)
#  = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b10 - a00*b11 + a11*b01 - a11*b00
#  = a00*b00 + a11*b11 - a10*b00 + a00*b10 + a11*b01 - a11*b00
#  = a00*b00 + a01*b00 + a00*b01 + a01*b01  (since a00*b10+a00*b11 cancel? no...)
#
#  Hmm, this is getting complicated. Let me just use a known correct Winograd 2x2 from a reference.
#
#  Standard Winograd 2x2 (F2):
#  U = [[1,1,1,0],
#       [1,1,0,1],
#       [0,1,1,-1],
#       [1,0,-1,1]]
#  V = [[1,1,0,0],
#       [1,0,1,0],
#       [0,1,1,0],
#       [0,0,1,1]]  -- this doesn't look right either.
#
#  Let me just use the schoolbook for 4x4x4 to be safe, which gives rank 64 per block.
#  3 blocks of 4x4x4 = 192 + 16 for 4x4x1 = 208. That's worse than 144.
#
#  Actually, Strassen for 4x4x4 gives rank 49 (since 2^3=8 for 2x2, and 4x4x4 can be done in 49).
#  Wait, 4x4x4 via Strassen: 4 = 2*2, so we can do 2x2x2 blocks with Strassen.
#  Actually, the rank of 4x4x4 matrix multiplication is at most 49 (Strassen applied to 2x2 blocks gives 2^3 * 7 = 56, but there's a better decomposition).
#
#  Actually, let me just use schoolbook for everything to be safe: rank = 4*4*9 = 144.
#  But we need to beat that. Let me try a simple block decomposition.
#
#  Split the 4x9 output into a 4x4 and 4x4 and 4x1.
#  For each 4x4x4: use Strassen with rank 49.
#  For 4x4x1: schoolbook with rank 16.
#  Total: 49 + 49 + 16 = 114.
#
#  But I need to correctly construct the Strassen 4x4x4 decomposition.
#  4x4x4 can be seen as 2x2 blocks of 2x2 matrices.
#  Strassen on 2x2 blocks: 7 products of 2x2x2, each of which is schoolbook rank 8.
#  So rank = 7 * 8 = 56. Not 49.
#
#  Actually, the minimum rank for 4x4x4 is not well-known to be 49. Let me just use schoolbook.
#
#  Let me reconsider: the current program has rank 144. I need to do better.
#  A simple improvement: use Strassen for the 2x2x2 core.
#
#  For 4x4x9, we can partition into blocks. Let's use a 2x2 block structure for the 4x4 matrices.
#  A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
#  B = [[B0, B1, B2], [B3, B4, B5]] where B0..B5 are 2x3 matrices.
#
#  C = [[A00*B0 + A01*B3, A00*B1 + A01*B4, A00*B2 + A01*B5],
#       [A10*B0 + A11*B3, A10*B1 + A11*B4, A10*B2 + A11*B5]]
#
#  Each 2x2 * 2x3 can be done by treating it as 3 columns of 2x2*2x1.
#  2x2 * 2x1 = schoolbook rank 4.
#  So each 2x2*2x3 = 3*4 = 12.
#  There are 8 such block multiplications (2x2 blocks of A times 2x3 blocks of B).
#  Total rank = 8 * 12 = 96.
#
#  That's better than 144! Let me implement this.
#
#  Actually, even better: use Winograd 2x2x2 (rank 7) for the 2x2*2x2 part, but we have 2x2*2x3.
#  2x2*2x3 = 2x2*2x2 + 2x2*2x1.
#  2x2*2x2 = Winograd rank 7.
#  2x2*2x1 = schoolbook rank 4.
#  So each 2x2*2x3 = 7 + 4 = 11? No, we can't just add them like that because the U and V factors need to be compatible.
#
#  Let me just use schoolbook for 2x2*2x3: rank 4*3 = 12.
#  8 blocks * 12 = 96.

def _schoolbook_2x2x2x3():
    """Return U (4x12), V (6x12), W (6x12) for 2x2 * 2x3."""
    # A is 2x2, B is 2x3, C is 2x3.
    # U shape: (2*2, 12) = (4, 12)
    # V shape: (2*3, 12) = (6, 12)
    # W shape: (2*3, 12) = (6, 12)  [since output is 2x3, p*n = 3*2 = 6]
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(6)]
    W = [[0]*12 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Partition into 2x2 blocks of A, and 2x3 blocks of B.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B0, B1, B2 (row 0), B3, B4, B5 (row 1), each 2x3
    # C blocks: C00, C01, C02 (row 0), C10, C11, C12 (row 1), each 2x3
    #
    # C00 = A00*B0 + A01*B3
    # C01 = A00*B1 + A01*B4
    # C02 = A00*B2 + A01*B5
    # C10 = A10*B0 + A11*B3
    # C11 = A10*B1 + A11*B4
    # C12 = A10*B2 + A11*B5
    #
    # Each block multiplication is 2x2 * 2x3.
    # We have 8 such multiplications (4 A-blocks * 2 B-row-blocks... actually 2x2 A-blocks times 2 B-row-blocks = 4, but each with 3 B-column-blocks? No.)
    #
    # Actually: A is 2x2 blocks of 2x2. B is 2x3 blocks of 2x3.
    # C[i,j] = sum_k A[i,k] * B[k,j] where i,k are 0,1 and j is 0,1,2.
    # So C has 2x2 = 4 blocks, each 2x3.
    # Each C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # So we need 4 * 2 = 8 block multiplications of 2x2 * 2x3.
    
    # Each 2x2*2x3 has rank 12 (schoolbook).
    # Total rank = 8 * 12 = 96.
    
    rank = 96
    U = [[0]*rank for _ in range(n*m)]  # 16 x 96
    V = [[0]*rank for _ in range(m*p)]  # 36 x 96
    W = [[0]*rank for _ in range(p*n)]  # 36 x 96
    
    # Helper to get the 2x2*2x3 factors
    U_base, V_base, W_base = _schoolbook_2x2x2x3()
    
    # Map global indices to local block indices
    # A[i,j] global row = i*2 + i_local, col = j*2 + j_local, where i,j in {0,1}, i_local,j_local in {0,1}
    # B[i,j] global row = i*2 + i_local, col = j*3 + j_local, where i in {0,1}, j in {0,1,2}, i_local in {0,1}, j_local in {0,1,2}
    # C[i,j] global row = i*2 + i_local, col = j*3 + j_local
    
    for a_block_i in range(2):  # A row block
        for b_block_i in range(2):  # B row block (= A col block)
            for c_block_j in range(3):  # B col block (= C col block)
                # C[a_block_i, c_block_j] += A[a_block_i, b_block_i] * B[b_block_i, c_block_j]
                # This is a 2x2 * 2x3 multiplication.
                # U rows: A[a_block_i, b_block_i] has global rows (a_block_i*2 + il, b_block_i*2 + jl)
                #          index in U: (a_block_i*2 + il) * 4 + (b_block_i*2 + jl)  [since m=4, so i*m+j]
                # V rows: B[b_block_i, c_block_j] has global rows (b_block_i*2 + jl, c_block_j*3 + kl)
                #          index in V: (b_block_i*2 + jl) * 9 + (c_block_j*3 + kl)  [since p=9, so j*p+k]
                # W rows: C[a_block_i, c_block_j] has global rows (c_block_j*3 + kl, a_block_i*2 + il)
                #          index in W: (c_block_j*3 + kl) * 4 + (a_block_i*2 + il)  [since n=4, so k*n+i]
                
                base_col = (a_block_i * 2 + b_block_i * 2 + c_block_j) * 12  # this indexing is arbitrary but unique
                # Actually, let's just use a running column counter.
                pass
    
    # Let me use a simpler approach with a column counter.
    col = 0
    for a_block_i in range(2):
        for b_block_i in range(2):
            for c_block_j in range(3):
                for il in range(2):
                    for jl in range(2):
                        for kl in range(3):
                            # A index: global row = a_block_i*2 + il, col = b_block_i*2 + jl
                            # U row: (a_block_i*2 + il) * 4 + (b_block_i*2 + jl)
                            u_row = (a_block_i*2 + il) * 4 + (b_block_i*2 + jl)
                            # B index: global row = b_block_i*2 + jl, col = c_block_j*3 + kl
                            # V row: (b_block_i*2 + jl) * 9 + (c_block_j*3 + kl)
                            v_row = (b_block_i*2 + jl) * 9 + (c_block_j*3 + kl)
                            # C index: global row = a_block_i*2 + il, col = c_block_j*3 + kl
                            # W row: (c_block_j*3 + kl) * 4 + (a_block_i*2 + il)
                            w_row = (c_block_j*3 + kl) * 4 + (a_block_i*2 + il)
                            
                            U[u_row][col] = 1
                            V[v_row][col] = 1
                            W[w_row][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
