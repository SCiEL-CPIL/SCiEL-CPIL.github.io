"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[a*2+b, r], V[b*2+c, r], W[c*2+a, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 1: M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Product 3: M3 = A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Product 4: M4 = A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Product 5: M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Product 6: M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Product 7: M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _laderman_3x3():
    """Return U, V, W for 3x3 matrix multiplication using Laderman's algorithm.
    U has shape (9, 15), V has shape (9, 15), W has shape (9, 15).
    Indexing: U[a*3+b, r], V[b*3+c, r], W[c*3+a, r]
    """
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    
    # Product 0: A[0,0] * B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Product 1: (A[0,0]+A[0,1]) * (B[0,1]-B[1,1]) -> C[0,1]
    U[0][1] = 1; U[1][1] = 1
    V[1][1] = 1; V[4][1] = -1
    W[1][1] = 1
    
    # Product 2: (A[0,0]+A[0,2]) * (B[0,2]-B[2,2]) -> C[0,2]
    U[0][2] = 1; U[2][2] = 1
    V[2][2] = 1; V[8][2] = -1
    W[2][2] = 1
    
    # Product 3: (A[0,1]+A[0,2]) * (B[1,2]-B[2,1]) -> C[0,2] (additional)
    U[1][3] = 1; U[2][3] = 1
    V[5][3] = 1; V[3][3] = -1
    W[2][3] = 1
    
    # Product 4: A[1,0] * B[0,0] -> C[1,0]
    U[3][4] = 1; V[0][4] = 1; W[3][4] = 1
    
    # Product 5: (A[1,0]+A[1,1]) * (B[0,1]-B[1,1]) -> C[1,1]
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1; V[4][5] = -1
    W[4][5] = 1
    
    # Product 6: (A[1,0]+A[1,2]) * (B[0,2]-B[2,2]) -> C[1,2]
    U[3][6] = 1; U[5][6] = 1
    V[2][6] = 1; V[8][6] = -1
    W[5][6] = 1
    
    # Product 7: (A[1,1]+A[1,2]) * (B[1,2]-B[2,1]) -> C[1,2] (additional)
    U[4][7] = 1; U[5][7] = 1
    V[5][7] = 1; V[3][7] = -1
    W[5][7] = 1
    
    # Product 8: A[2,0] * B[0,0] -> C[2,0]
    U[6][8] = 1; V[0][8] = 1; W[6][8] = 1
    
    # Product 9: (A[2,0]+A[2,1]) * (B[0,1]-B[1,1]) -> C[2,1]
    U[6][9] = 1; U[7][9] = 1
    V[1][9] = 1; V[4][9] = -1
    W[7][9] = 1
    
    # Product 10: (A[2,0]+A[2,2]) * (B[0,2]-B[2,2]) -> C[2,2]
    U[6][10] = 1; U[8][10] = 1
    V[2][10] = 1; V[8][10] = -1
    W[8][10] = 1
    
    # Product 11: (A[2,1]+A[2,2]) * (B[1,2]-B[2,1]) -> C[2,2] (additional)
    U[7][11] = 1; U[8][11] = 1
    V[5][11] = 1; V[3][11] = -1
    W[8][11] = 1
    
    # Product 12: (A[0,0]+A[1,0]+A[2,0]) * (B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
    U[0][12] = 1; U[3][12] = 1; U[6][12] = 1
    V[3][12] = 1; V[6][12] = 1
    W[0][12] = 1; W[3][12] = 1; W[6][12] = 1
    
    # Product 13: (A[0,1]+A[1,1]+A[2,1]) * (B[1,0]+B[2,0]) -> C[0,1]+C[1,1]+C[2,1]
    U[1][13] = 1; U[4][13] = 1; U[7][13] = 1
    V[3][13] = 1; V[6][13] = 1
    W[1][13] = 1; W[4][13] = 1; W[7][13] = 1
    
    # Product 14: (A[0,2]+A[1,2]+A[2,2]) * (B[1,0]+B[2,0]) -> C[0,2]+C[1,2]+C[2,2]
    U[2][14] = 1; U[5][14] = 1; U[8][14] = 1
    V[3][14] = 1; V[6][14] = 1
    W[2][14] = 1; W[5][14] = 1; W[8][14] = 1
    
    return U, V, W

def _schoolbook_rect(n, m, p):
    """Return schoolbook decomposition for n×m times m×p multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_5x5_decomposition():
    """
    Build a 5x5 matrix multiplication decomposition using block structure.
    Partition indices into S={0,1,2} and T={3,4}.
    The tensor decomposes into 8 independent sub-tensors based on which
    partition a, b, c belong to.
    
    Total rank = sum of ranks of all 8 sub-tensors.
    """
    n = 5
    S = [0, 1, 2]  # size 3
    T = [3, 4]     # size 2
    
    # Sub-tensor specifications: (group_a, group_b, group_c) -> (n_a, n_b, n_c)
    # where n_a = len(group_a), n_b = len(group_b), n_c = len(group_c)
    # U dim = n_a * n_b, V dim = n_b * n_c, W dim = n_c * n_a
    
    blocks = [
        (S, S, S),  # (3,3,3) -> Laderman rank 15
        (S, S, T),  # (3,2,2) -> schoolbook rank 12
        (S, T, S),  # (2,3,3) -> schoolbook rank 18
        (S, T, T),  # (2,2,2) -> Strassen rank 7
        (T, S, S),  # (2,3,2) -> schoolbook rank 12
        (T, S, T),  # (2,2,2) -> Strassen rank 7
        (T, T, S),  # (2,2,3) -> schoolbook rank 12
        (T, T, T),  # (2,2,2) -> Strassen rank 7
    ]
    
    # Get decompositions for each block type
    def get_decomp(na, nb, nc):
        if na == 3 and nb == 3 and nc == 3:
            return _laderman_3x3()
        elif na == 2 and nb == 2 and nc == 2:
            return _strassen_2x2()
        else:
            return _schoolbook_rect(na, nb, nc)
    
    # Compute total rank
    total_rank = 0
    block_decomps = []
    for ga, gb, gc in blocks:
        na, nb, nc = len(ga), len(gb), len(gc)
        U_b, V_b, W_b = get_decomp(na, nb, nc)
        rank_b = len(U_b[0])
        total_rank += rank_b
        block_decomps.append((ga, gb, gc, na, nb, nc, U_b, V_b, W_b, rank_b))
    
    # Build full U, V, W matrices
    U = [[0]*total_rank for _ in range(n*n)]  # 25 rows
    V = [[0]*total_rank for _ in range(n*n)]  # 25 rows
    W = [[0]*total_rank for _ in range(n*n)]  # 25 rows
    
    col_offset = 0
    for ga, gb, gc, na, nb, nc, U_b, V_b, W_b, rank_b in block_decomps:
        # Map local indices to global indices
        # U row: a*5+b where a in ga, b in gb
        # V row: b*5+c where b in gb, c in gc
        # W row: c*5+a where c in gc, a in ga
        
        for i_local in range(na * nb):
            a_local = i_local // nb
            b_local = i_local % nb
            a_global = ga[a_local]
            b_global = gb[b_local]
            U_global_row = a_global * n + b_global
            
            for r in range(rank_b):
                U[U_global_row][col_offset + r] = U_b[i_local][r]
        
        for i_local in range(nb * nc):
            b_local = i_local // nc
            c_local = i_local % nc
            b_global = gb[b_local]
            c_global = gc[c_local]
            V_global_row = b_global * n + c_global
            
            for r in range(rank_b):
                V[V_global_row][col_offset + r] = V_b[i_local][r]
        
        for i_local in range(nc * na):
            c_local = i_local // na
            a_local = i_local % na
            c_global = gc[c_local]
            a_global = ga[a_local]
            W_global_row = c_global * n + a_global
            
            for r in range(rank_b):
                W[W_global_row][col_offset + r] = W_b[i_local][r]
        
        col_offset += rank_b
    
    return U, V, W

def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try the block decomposition
    if time.time() < deadline - 5:
        U_bd, V_bd, W_bd = _build_5x5_decomposition()
        rank_bd = len(U_bd[0])
        
        if _verify(U_bd, V_bd, W_bd, n, m, p, rank_bd):
            print(f"Block decomposition verified, rank {rank_bd}")
            if rank_bd < best_rank:
                best_rank = rank_bd
                best_result = {"U": U_bd, "V": V_bd, "W": W_bd}
        else:
            print(f"Block decomposition FAILED verification, rank {rank_bd}")
    
    return best_result
# EVOLVE-BLOCK-END
