import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    if time.time() > deadline:
        return True
    return False

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7) - maps A entries [a00,a01,a10,a11] to 7 intermediate values
    V: (4, 7) - maps B entries [b00,b01,b10,b11] to 7 intermediate values
    W: (4, 7) - maps 7 products to C entries [c00,c01,c10,c11]
    
    Strassen's 7 multiplications:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)(b00)
    M3 = (a00)(b01 - b11)
    M4 = (a11)(b10 - b00)
    M5 = (a00 + a01)(b11)
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    Output:
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    """
    # U: row index is A entry (0=a00, 1=a01, 2=a10, 3=a11), col is which M
    U = [
        [1.0,  0.0,  1.0,  0.0,  1.0, -1.0,  0.0],   # a00
        [0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0],   # a01
        [0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0],   # a10
        [1.0,  1.0,  0.0,  1.0,  0.0,  0.0, -1.0],   # a11
    ]
    
    # V: row index is B entry (0=b00, 1=b01, 2=b10, 3=b11), col is which M
    V = [
        [1.0,  1.0,  0.0, -1.0,  0.0,  1.0,  0.0],   # b00
        [0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0],   # b01
        [0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0],   # b10
        [1.0,  0.0, -1.0,  0.0,  1.0,  0.0,  1.0],   # b11
    ]
    
    # W: row index is C entry (0=c00, 1=c01, 2=c10, 3=c11), col is which M
    W = [
        [1.0,  0.0,  0.0,  1.0, -1.0,  0.0,  1.0],   # c00
        [0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0],   # c01
        [0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0],   # c10
        [1.0, -1.0, 1.0,  0.0,  0.0,  1.0,  0.0],    # c11
    ]
    
    return U, V, W

def _build_hybrid_decomposition(n, m, p):
    """
    Build a hybrid decomposition using Strassen for a 2x2x2 block
    and schoolbook for the remaining terms.
    
    For 3x4x5: Use Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    This replaces 8 schoolbook terms (2*2*2) with 7 Strassen terms.
    Remaining terms: 60 - 8 = 52 schoolbook terms.
    Total rank: 7 + 52 = 59.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank_strassen = 7
    # Schoolbook terms for all (i,j,k) not in {0,1}x{0,1}x{0,1}
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not (i < 2 and j < 2 and k < 2):
                    schoolbook_terms.append((i, j, k))
    
    total_rank = rank_strassen + len(schoolbook_terms)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Fill in Strassen block (columns 0..6)
    # Strassen handles i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*m+j, r] for i in {0,1}, j in {0,1}
    # V[j*p+k, r] for j in {0,1}, k in {0,1}
    # W[k*n+i, r] for k in {0,1}, i in {0,1}
    for r in range(rank_strassen):
        for i in range(2):
            for j in range(2):
                U[i * m + j][r] = U_s[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + k][r] = V_s[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][r] = W_s[k * 2 + i][r]
    
    # Fill in schoolbook terms (columns 7..)
    col = rank_strassen
    for (i, j, k) in schoolbook_terms:
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}

def _build_hybrid_two_strassen(n, m, p):
    """
    Build a hybrid decomposition using two Strassen blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    
    Each Strassen block replaces 8 schoolbook terms with 7.
    Total: 60 - 8 - 8 + 7 + 7 = 58.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank_per_block = 7
    num_blocks = 2
    total_strassen_rank = rank_per_block * num_blocks
    
    # Schoolbook terms for all (i,j,k) not covered by Strassen blocks
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
                in_block1 = (i < 2 and j < 2 and k < 2)
                # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
                in_block2 = (i < 2 and 2 <= j < 4 and k < 2)
                if not (in_block1 or in_block2):
                    schoolbook_terms.append((i, j, k))
    
    total_rank = total_strassen_rank + len(schoolbook_terms)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}, columns 0..6
    for r in range(rank_per_block):
        c = r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_s[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + k][c] = V_s[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][c] = W_s[k * 2 + i][r]
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}, columns 7..13
    for r in range(rank_per_block):
        c = rank_per_block + r
        for i in range(2):
            for j in range(2, 4):
                U[i * m + j][c] = U_s[i * 2 + (j - 2)][r]
        for j in range(2, 4):
            for k in range(2):
                V[j * p + k][c] = V_s[(j - 2) * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][c] = W_s[k * 2 + i][r]
    
    # Fill in schoolbook terms
    col = total_strassen_rank
    for (i, j, k) in schoolbook_terms:
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}

def _build_hybrid_three_strassen(n, m, p):
    """
    Build a hybrid decomposition using three Strassen blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    Block 3: i in {2}, j in {0,1}, k in {0,1} -- but this is only 1x2x2, not 2x2x2
    
    Actually, let's do:
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    
    Each block is 2x2x2, replacing 8 schoolbook terms with 7.
    Total: 60 - 8 - 8 - 8 + 7 + 7 + 7 = 57.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank_per_block = 7
    num_blocks = 3
    total_strassen_rank = rank_per_block * num_blocks
    
    # Define the blocks: (i_range, j_range, k_range)
    blocks = [
        (range(0, 2), range(0, 2), range(0, 2)),   # Block 1
        (range(0, 2), range(2, 4), range(0, 2)),   # Block 2
        (range(0, 2), range(0, 2), range(2, 4)),   # Block 3
    ]
    
    # Schoolbook terms for all (i,j,k) not covered by Strassen blocks
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (ir, jr, kr) in blocks:
                    if i in ir and j in jr and k in kr:
                        in_block = True
                        break
                if not in_block:
                    schoolbook_terms.append((i, j, k))
    
    total_rank = total_strassen_rank + len(schoolbook_terms)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block_idx, (ir, jr, kr) in enumerate(blocks):
        i_list = list(ir)
        j_list = list(jr)
        k_list = list(kr)
        
        for r in range(rank_per_block):
            c = col + r
            for i in i_list:
                for j in j_list:
                    U[i * m + j][c] = U_s[i_list.index(i) * 2 + j_list.index(j)][r]
            for j in j_list:
                for k in k_list:
                    V[j * p + k][c] = V_s[j_list.index(j) * 2 + k_list.index(k)][r]
            for k in k_list:
                for i in i_list:
                    W[k * n + i][c] = W_s[k_list.index(k) * 2 + i_list.index(i)][r]
        col += rank_per_block
    
    # Fill in schoolbook terms
    for (i, j, k) in schoolbook_terms:
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}

def _build_hybrid_four_strassen(n, m, p):
    """
    Build a hybrid decomposition using four Strassen blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    Each block is 2x2x2, replacing 8 schoolbook terms with 7.
    Total: 60 - 8 - 8 - 8 - 8 + 7 + 7 + 7 + 7 = 56.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank_per_block = 7
    num_blocks = 4
    total_strassen_rank = rank_per_block * num_blocks
    
    # Define the blocks: (i_range, j_range, k_range)
    blocks = [
        (range(0, 2), range(0, 2), range(0, 2)),   # Block 1
        (range(0, 2), range(2, 4), range(0, 2)),   # Block 2
        (range(0, 2), range(0, 2), range(2, 4)),   # Block 3
        (range(0, 2), range(2, 4), range(2, 4)),   # Block 4
    ]
    
    # Schoolbook terms for all (i,j,k) not covered by Strassen blocks
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (ir, jr, kr) in blocks:
                    if i in ir and j in jr and k in kr:
                        in_block = True
                        break
                if not in_block:
                    schoolbook_terms.append((i, j, k))
    
    total_rank = total_strassen_rank + len(schoolbook_terms)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block_idx, (ir, jr, kr) in enumerate(blocks):
        i_list = list(ir)
        j_list = list(jr)
        k_list = list(kr)
        
        for r in range(rank_per_block):
            c = col + r
            for i in i_list:
                for j in j_list:
                    U[i * m + j][c] = U_s[i_list.index(i) * 2 + j_list.index(j)][r]
            for j in j_list:
                for k in k_list:
                    V[j * p + k][c] = V_s[j_list.index(j) * 2 + k_list.index(k)][r]
            for k in k_list:
                for i in i_list:
                    W[k * n + i][c] = W_s[k_list.index(k) * 2 + i_list.index(i)][r]
        col += rank_per_block
    
    # Fill in schoolbook terms
    for (i, j, k) in schoolbook_terms:
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    """
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    rank = U_arr.shape[1]
    C_computed = np.zeros((n, p), dtype=float)
    
    for r in range(rank):
        term_U = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(m):
                term_U[i] += U_arr[i * m + j, r] * A[i, j]
        
        term_V = np.zeros(p, dtype=float)
        for k in range(p):
            for j in range(m):
                term_V[k] += V_arr[j * p + k, r] * B[j, k]
        
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k * n + i, r] * term_U[i] * term_V[k]
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 110  # Leave 10s safety margin
    
    # Start with schoolbook (rank 60)
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try hybrid decompositions with increasing number of Strassen blocks
    hybrid_builders = [
        _build_hybrid_decomposition,       # 1 Strassen block, rank 59
        _build_hybrid_two_strassen,        # 2 Strassen blocks, rank 58
        _build_hybrid_three_strassen,      # 3 Strassen blocks, rank 57
        _build_hybrid_four_strassen,       # 4 Strassen blocks, rank 56
    ]
    
    for builder in hybrid_builders:
        if _check_time(deadline):
            break
        try:
            hybrid = builder(n, m, p)
            hybrid_rank = hybrid["rank"]
            if hybrid_rank < best_rank:
                valid = _verify_decomposition(hybrid['U'], hybrid['V'], hybrid['W'], n, m, p)
                if valid:
                    best = hybrid
                    best_rank = hybrid_rank
        except Exception:
            pass
    
    # Verify the best decomposition
    if not _check_time(deadline):
        valid = _verify_decomposition(best['U'], best['V'], best['W'], n, m, p)
        if valid:
            return best
    
    # Fallback: return schoolbook anyway
    return _build_schoolbook(n, m, p)
