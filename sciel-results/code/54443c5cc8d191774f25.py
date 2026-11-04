"""Tensor decomposition for 2x3 * 3x10 matrix multiplication using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """
    Return (U, V, W) for 2x2 * 2x2 multiplication using Strassen's rank-7 algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # U indices: a=0, b=1, c=2, d=3
    # V indices: e=0, f=1, g=2, h=3
    # W indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # p1 = (a+d)(e+h): U contributions to a,d; V to e,h
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    
    # p2 = (c+d)e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    
    # p3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1 # h
    
    # p4 = d(g-e)
    U[3][3] = 1  # d
    V[0][3] = -1 # e
    V[2][3] = 1  # g
    
    # p5 = (a+b)h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    
    # p6 = (c-a)(e+f)
    U[0][5] = -1 # a
    U[2][5] = 1  # c
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    
    # p7 = (b-d)(g+h)
    U[1][6] = 1  # b
    U[3][6] = -1 # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    
    # W: C[ki] where k=0,1 and i=0,1
    # C[0,0] = p1 + p4 - p5 + p7
    W[0][0] = 1   # p1
    W[0][3] = 1   # p4
    W[0][4] = -1  # p5
    W[0][6] = 1   # p7
    
    # C[0,1] = p3 + p5
    W[1][2] = 1   # p3
    W[1][4] = 1   # p5
    
    # C[1,0] = p2 + p4
    W[2][1] = 1   # p2
    W[2][3] = 1   # p4
    
    # C[1,1] = p1 - p2 + p3 + p6
    W[3][0] = 1   # p1
    W[3][1] = -1  # p2
    W[3][2] = 1   # p3
    W[3][5] = 1   # p6
    
    return U, V, W, rank


def _build_schoolbook_212():
    """
    Return (U, V, W) for 2x1 * 1x2 multiplication (rank 4).
    A is 2x1: [a; c], B is 1x2: [e, f]
    C = [[a*e, a*f], [c*e, c*f]]
    U shape: (2, 4), V shape: (2, 4), W shape: (2, 4)
    U indices: a=0, c=1
    V indices: e=0, f=1
    W indices: C[0,0]=0, C[0,1]=1
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    # p1 = a*e -> C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # p2 = a*f -> C[0,1]
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1
    
    # p3 = c*e -> C[1,0]
    U[1][2] = 1
    V[0][2] = 1
    W[0][2] = 1
    
    # p4 = c*f -> C[1,1]
    U[1][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W, rank


def _build_block_232():
    """
    Build decomposition for 2x3 * 3x2 multiplication.
    Split m=3 into m1=2, m2=1.
    2x2*2x2: Strassen rank 7
    2x1*1x2: schoolbook rank 4
    Total rank: 11
    
    A is 2x3: indices a00,a01,a02,a10,a11,a12 (row-major: i*3+j)
    B is 3x2: indices b00,b01,b10,b11,b20,b21 (row-major: j*2+k)
    C is 2x2: indices c00,c01,c10,c11 (row-major: i*2+k)
    
    W convention: W[k*n+i, r] where n=2, so W[k*2+i]
    C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    """
    rank_total = 11
    U = [[0]*rank_total for _ in range(6)]  # 2*3
    V = [[0]*rank_total for _ in range(6)]  # 3*2
    W = [[0]*rank_total for _ in range(4)]  # 2*2
    
    # Block 1: rows 0,1 of A (2x2) * rows 0,1 of B (2x2) -> Strassen
    sU, sV, sW, sr = _build_strassen_222()
    for i in range(4):
        for r in range(sr):
            U[i][r] += sU[i][r]
            V[i][r] += sV[i][r]
    for i in range(4):
        for r in range(sr):
            W[i][r] += sW[i][r]
    
    # Block 2: row 2 of A (2x1) * row 2 of B (1x2) -> schoolbook
    # A row 2: [a02, a12] -> indices 2, 5 in U
    # B row 2: [b20, b21] -> indices 4, 5 in V
    # C: 2x2
    # p1 = a02*b20 -> C[0,0]
    # p2 = a02*b21 -> C[0,1]
    # p3 = a12*b20 -> C[1,0]
    # p4 = a12*b21 -> C[1,1]
    base = 7
    U[2][base+0] = 1   # a02
    V[4][base+0] = 1   # b20
    W[0][base+0] = 1   # C[0,0]
    
    U[2][base+1] = 1   # a02
    V[5][base+1] = 1   # b21
    W[1][base+1] = 1   # C[0,1]
    
    U[5][base+2] = 1   # a12
    V[4][base+2] = 1   # b20
    W[2][base+2] = 1   # C[1,0]
    
    U[5][base+3] = 1   # a12
    V[5][base+3] = 1   # b21
    W[3][base+3] = 1   # C[1,1]
    
    return U, V, W, rank_total


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r u_{ij}^{(r)} v_{jk}^{(r)} = w_{ki}^{(r)} for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U_arr[i * m + j] @ V_arr[j * p + k]
                rhs = W_arr[k * n + i]
                if not np.allclose(lhs, rhs, atol=1e-9):
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Split p=10 into five blocks of size 2.
    # For each 2x3 * 3x2 block, use Strassen (rank 7) + schoolbook (rank 4) = rank 11.
    # Total rank: 5 * 11 = 55.
    
    block_rank = 11
    num_blocks = 5  # p / 2 = 10 / 2
    total_rank = block_rank * num_blocks  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    for block in range(num_blocks):
        block_base = block * block_rank
        
        # Get the 2x3 * 3x2 decomposition
        bU, bV, bW, br = _build_block_232()
        
        # Place into global matrices
        # U: same for all blocks (A is 2x3)
        for i in range(n * m):
            for r in range(br):
                U[i][block_base + r] += bU[i][r]
        
        # V: rows j=0,1,2 of B, columns k=block*2, block*2+1
        for j in range(m):
            for k_local in range(2):
                k_global = block * 2 + k_local
                v_idx = j * p + k_global
                bV_idx = j * 2 + k_local
                for r in range(br):
                    V[v_idx][block_base + r] += bV[bV_idx][r]
        
        # W: C[i,k] where k = block*2, block*2+1
        # W index: k*n + i
        for i in range(n):
            for k_local in range(2):
                k_global = block * 2 + k_local
                w_idx = k_global * n + i
                bW_idx = k_local * n + i
                for r in range(br):
                    W[w_idx][block_base + r] += bW[bW_idx][r]
    
    # Quick self-check
    if not _validate(n, m, p, U, V, W):
        # Fall back to trivial if validation fails
        U, V, W, _ = _build_trivial(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
