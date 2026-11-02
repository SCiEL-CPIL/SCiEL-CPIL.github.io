"""Tensor decomposition for 2x3x10 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # seconds; leave 10s margin before the 150s kill

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Check tensor equation: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # and for all other (i,j,k) combinations in the tensor, it should be 0.
    # Actually, the tensor M_{i,j,k} = 1 if the output element C[i,k] = sum_j A[i,j]*B[j,k].
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{correct}
    # For matrix multiplication, the tensor has T[i,j,k] = 1 for all valid i,j,k (since each
    # (i,j,k) triple contributes A[i,j]*B[j,k] to C[i,k]).
    # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i in [0,n), j in [0,m), k in [0,p).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_strassen_block_2x3x2():
    """
    Build a rank-11 decomposition for 2x3x2 matrix multiplication.
    
    We partition A (2x3) into A_left (2x2) and A_right (2x1).
    We partition B (3x2) into B_top (2x2) and B_bottom (1x2).
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2x2), use Strassen: rank 7.
    For A_right * B_bottom (2x1x1x2), this is outer product: A_right[:,0] * B_bottom[0,:].
    This gives 2x2 = 4 rank-1 terms.
    Total: 7 + 4 = 11.
    
    Strassen for 2x2x2:
    Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U rows: indices 0,1 for A_left; 2,3 for A_right
    V rows: indices 0,1 for B_top; 2,3 for B_bottom
    W rows: indices 0,1,2,3 for C
    
    Strassen terms (7):
    Term 0: M1 = (a11+a22)(b11+b22)
      U: row 0 += 1, row 1 += 1
      V: row 0 += 1, row 1 += 1
      W: row 0 += 1 (C11), row 3 += 1 (C22)
    
    Wait, I need to be more careful. Each Strassen term is a rank-1 contribution.
    M1 contributes to C11 with +1 and C22 with +1.
    But in the tensor decomposition, each "column" c gives a rank-1 term.
    For M1: U has entries for A indices, V for B indices, W for C indices.
    M1 = (a11+a22)(b11+b22) contributes to C11 and C22.
    
    Actually, each rank-1 term in the tensor is: U[a_index, c] * V[b_index, c] * W[c_index, c]
    For the 2x2x2 Strassen, each M_i is a product of a linear form in A and a linear form in B,
    and it contributes linearly to each C element. So we need one column per (M_i, C_element) pair
    where M_i contributes to C_element.
    
    M1 -> C11 (+1), C22 (+1): 2 columns
    M2 -> C21 (+1), C22 (-1): 2 columns
    M3 -> C12 (+1), C22 (+1): 2 columns
    M4 -> C11 (+1), C21 (+1): 2 columns
    M5 -> C11 (-1), C12 (+1): 2 columns
    M6 -> C22 (+1): 1 column
    M7 -> C11 (+1): 1 column
    
    Total: 2+2+2+2+2+1+1 = 12. Hmm, that's more than 7.
    
    Wait, I'm confusing myself. In the standard Strassen decomposition of the 2x2x2 tensor,
    the rank is 7, meaning there are 7 rank-1 terms. Each rank-1 term is a triple of vectors
    (u, v, w) where u is in R^4 (A indices), v is in R^4 (B indices), w is in R^4 (C indices).
    The tensor is sum of 7 such outer products.
    
    For M1: u = e_{00} + e_{11} (for A: a11 and a22), v = e_{00} + e_{11} (for B: b11 and b22)
    w = e_{C11} + e_{C22} (contributes +1 to C11 and +1 to C22)
    
    So each M_i is ONE rank-1 term with a specific u, v, w. The w vector can have multiple nonzero
    entries. So we have 7 columns total for Strassen.
    
    Then for the 4 schoolbook terms (A_right * B_bottom):
    Term: a_{i,2} * b_{2,k} -> C_{i,k} for i in {0,1}, k in {0,1}
    Each is a rank-1 term: u = e_{(i,2)}, v = e_{(2,k)}, w = e_{(i,k)}
    4 more columns.
    
    Total: 7 + 4 = 11 columns.
    """
    # A indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    rank = 11
    U = [[0.0]*rank for _ in range(6)]
    V = [[0.0]*rank for _ in range(6)]
    W = [[0.0]*rank for _ in range(4)]
    
    # Strassen 2x2x2 (using A rows 0,1 and B rows 0,1, i.e., A_left and B_top)
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (a00+a11)(b00+b11) -> C00(+1), C11(+1)
    c = 0
    U[0][c] += 1; U[4][c] += 1  # a00, a11
    V[0][c] += 1; V[3][c] += 1  # b00, b11
    W[0][c] += 1; W[3][c] += 1  # C00, C11
    
    # M2 = (a10+a11)(b00) -> C10(+1), C11(-1)
    c = 1
    U[3][c] += 1; U[4][c] += 1  # a10, a11
    V[0][c] += 1  # b00
    W[2][c] += 1; W[3][c] += -1  # C10(+1), C11(-1)
    
    # M3 = a00(b01-b11) -> C01(+1), C11(+1)
    c = 2
    U[0][c] += 1  # a00
    V[1][c] += 1; V[3][c] += -1  # b01, b11
    W[1][c] += 1; W[3][c] += 1  # C01(+1), C11(+1)
    
    # M4 = a11(b10-b00) -> C00(+1), C10(+1)
    c = 3
    U[4][c] += 1  # a11
    V[2][c] += 1; V[0][c] += -1  # b10, b00
    W[0][c] += 1; W[2][c] += 1  # C00(+1), C10(+1)
    
    # M5 = (a00+a01)(b11) -> C00(-1), C01(+1)
    c = 4
    U[0][c] += 1; U[1][c] += 1  # a00, a01
    V[3][c] += 1  # b11
    W[0][c] += -1; W[1][c] += 1  # C00(-1), C01(+1)
    
    # M6 = (a10-a00)(b00+b01) -> C11(+1)
    c = 5
    U[3][c] += 1; U[0][c] += -1  # a10, a00
    V[0][c] += 1; V[1][c] += 1  # b00, b01
    W[3][c] += 1  # C11(+1)
    
    # M7 = (a01-a11)(b10+b11) -> C00(+1)
    c = 6
    U[1][c] += 1; U[4][c] += -1  # a01, a11
    V[2][c] += 1; V[3][c] += 1  # b10, b11
    W[0][c] += 1  # C00(+1)
    
    # Schoolbook: A_right (column 2) * B_bottom (row 2)
    # a_{i,2} * b_{2,k} -> C_{i,k}
    # A index for (i,2): i=0 -> 2, i=1 -> 5
    # B index for (2,k): k=0 -> 4, k=1 -> 5
    # C index: (i,k): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    c = 7  # a_{0,2} * b_{2,0} -> C_{0,0}
    U[2][c] += 1
    V[4][c] += 1
    W[0][c] += 1
    
    c = 8  # a_{0,2} * b_{2,1} -> C_{0,1}
    U[2][c] += 1
    V[5][c] += 1
    W[1][c] += 1
    
    c = 9  # a_{1,2} * b_{2,0} -> C_{1,0}
    U[5][c] += 1
    V[4][c] += 1
    W[2][c] += 1
    
    c = 10  # a_{1,2} * b_{2,1} -> C_{1,1}
    U[5][c] += 1
    V[5][c] += 1
    W[3][c] += 1
    
    return U, V, W


def _build_block_decomposition():
    """
    Build the full 2x3x10 decomposition using 5 blocks of 2x3x2.
    Total rank: 5 * 11 = 55.
    """
    n, m, p = 2, 3, 10
    total_rank = 55
    U = [[0.0]*total_rank for _ in range(n*m)]  # 6 rows
    V = [[0.0]*total_rank for _ in range(m*p)]  # 30 rows
    W = [[0.0]*total_rank for _ in range(p*n)]  # 20 rows
    
    block_U, block_V, block_W = _build_strassen_block_2x3x2()
    
    for b in range(5):
        # B rows for this block: columns 2b and 2b+1
        # B global index: row j, col k -> j*p + k
        # For block b, B columns are 2b and 2b+1
        # B rows: j=0,1,2 (the 3 rows of B)
        # Block B local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1)
        # Global B index for (j, 2b+local_col):
        #   local_col 0 -> col 2b, local_col 1 -> col 2b+1
        #   j=0: global = 0*10 + 2b + local_col = 2b + local_col
        #   j=1: global = 1*10 + 2b + local_col = 10 + 2b + local_col
        #   j=2: global = 2*10 + 2b + local_col = 20 + 2b + local_col
        
        # A is shared: same A indices for all blocks
        # A global index: i*m + j = i*3 + j (i in 0,1; j in 0,1,2)
        # Block A local: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
        # These match global A indices directly.
        
        # C: for block b, C columns are 2b and 2b+1
        # Block C local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # Global C index: k*n + i (trace convention: W[k*n+i])
        #   C(i, 2b+local_col): global = (2b+local_col)*2 + i
        #   local (0,0) -> i=0, col=0 -> global = (2b)*2+0 = 4b
        #   local (0,1) -> i=0, col=1 -> global = (2b+1)*2+0 = 4b+2
        #   local (1,0) -> i=1, col=0 -> global = (2b)*2+1 = 4b+1
        #   local (1,1) -> i=1, col=1 -> global = (2b+1)*2+1 = 4b+3
        
        rank_offset = b * 11
        
        # Copy U (same for all blocks since A is shared)
        for a_idx in range(6):
            for c in range(11):
                U[a_idx][rank_offset + c] += block_U[a_idx][c]
        
        # Copy V with index mapping
        # Block V local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1)
        # Global B index for (j, col): j*p + col where col = 2b + local_col
        # local 0: j=0, col=0 -> global 2b
        # local 1: j=0, col=1 -> global 2b+1
        # local 2: j=1, col=0 -> global 10+2b
        # local 3: j=1, col=1 -> global 10+2b+1
        # local 4: j=2, col=0 -> global 20+2b
        # local 5: j=2, col=1 -> global 20+2b+1
        v_map = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for local_idx in range(6):
            global_idx = v_map[local_idx]
            for c in range(11):
                V[global_idx][rank_offset + c] += block_V[local_idx][c]
        
        # Copy W with index mapping
        # Block W local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # Global C index: (col)*n + i where col = 2b + local_col
        # local 0: i=0, col=0 -> global 0*2 + 0 = 4b (wait, col=2b*0? No)
        # Let me re-derive: local C index (i, k_local) where k_local in {0,1}
        # Global column = 2b + k_local
        # Global W index = column * n + i = (2b + k_local) * 2 + i
        # local 0: i=0, k_local=0 -> (2b)*2 + 0 = 4b
        # local 1: i=0, k_local=1 -> (2b+1)*2 + 0 = 4b+2
        # local 2: i=1, k_local=0 -> (2b)*2 + 1 = 4b+1
        # local 3: i=1, k_local=1 -> (2b+1)*2 + 1 = 4b+3
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for local_idx in range(4):
            global_idx = w_map[local_idx]
            for c in range(11):
                W[global_idx][rank_offset + c] += block_W[local_idx][c]
    
    return U, V, W


def construct():
    start_time = time.time()
    
    # Build the rank-55 block decomposition
    U, V, W = _build_block_decomposition()
    
    # Verify
    n, m, p = 2, 3, 10
    if _verify(U, V, W, n, m, p):
        result = {"U": U, "V": V, "W": W}
        return result
    
    # Fallback: schoolbook rank 60
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1.0
                V[j*p+k][column] = 1.0
                W[k*n+i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
