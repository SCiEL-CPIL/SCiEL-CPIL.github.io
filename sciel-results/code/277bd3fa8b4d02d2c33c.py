import numpy as np

def _strassen_2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    where i,k in {0,1} (output row/col), j in {0,1} (inner dim).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[3][3] = 1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _add_strassen_block(U, V, W, term_idx, i_off, j_off, k_off, n, m, p):
    """
    Add a Strassen 2x2x2 block to the global matrices.
    i_off: offset for i (0 or 1)
    j_off: offset for j (0 or 2)
    k_off: offset for k (0, 2, or 4)
    """
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        for idx in range(4):
            if sU[idx][c] != 0:
                # idx = i'*2 + j'
                i_prime = idx // 2
                j_prime = idx % 2
                global_i = i_prime + i_off
                global_j = j_prime + j_off
                U[global_i * m + global_j][term_idx] = sU[idx][c]
                
            if sV[idx][c] != 0:
                # idx = j'*2 + k'
                j_prime = idx // 2
                k_prime = idx % 2
                global_j = j_prime + j_off
                global_k = k_prime + k_off
                V[global_j * p + global_k][term_idx] = sV[idx][c]
                
            if sW[idx][c] != 0:
                # idx = k'*2 + i'
                k_prime = idx // 2
                i_prime = idx % 2
                global_k = k_prime + k_off
                global_i = i_prime + i_off
                W[global_k * n + global_i][term_idx] = sW[idx][c]

def _add_rank1_block(U, V, W, term_idx, i_off, j_off, k_val, n, m, p):
    """
    Add a rank-1 block for J_{2,2,1} at k=k_val.
    i_off: offset for i (0 or 1)
    j_off: offset for j (0 or 2)
    k_val: the k index (0, 2, or 4)
    """
    for i_prime in range(2):
        for j_prime in range(2):
            global_i = i_prime + i_off
            global_j = j_prime + j_off
            U[global_i * m + global_j][term_idx] = 1
            V[global_j * p + k_val][term_idx] = 1
            W[k_val * n + global_i][term_idx] = 1

def construct():
    n, m, p = 2, 4, 5
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    term_idx = 0
    
    # Block 1: j in {0, 1}
    # Sub-block 1.1: k in {0, 1}
    _add_strassen_block(U, V, W, term_idx, i_off=0, j_off=0, k_off=0, n=n, m=m, p=p)
    term_idx += 7
    
    # Sub-block 1.2: k in {2, 3}
    _add_strassen_block(U, V, W, term_idx, i_off=0, j_off=0, k_off=2, n=n, m=m, p=p)
    term_idx += 7
    
    # Sub-block 1.3: k = 4
    _add_rank1_block(U, V, W, term_idx, i_off=0, j_off=0, k_val=4, n=n, m=m, p=p)
    term_idx += 1
    
    # Block 2: j in {2, 3}
    # Sub-block 2.1: k in {0, 1}
    _add_strassen_block(U, V, W, term_idx, i_off=0, j_off=2, k_off=0, n=n, m=m, p=p)
    term_idx += 7
    
    # Sub-block 2.2: k in {2, 3}
    _add_strassen_block(U, V, W, term_idx, i_off=0, j_off=2, k_off=2, n=n, m=m, p=p)
    term_idx += 7
    
    # Sub-block 2.3: k = 4
    _add_rank1_block(U, V, W, term_idx, i_off=0, j_off=2, k_val=4, n=n, m=m, p=p)
    term_idx += 1
    
    return {"U": U, "V": V, "W": W}
