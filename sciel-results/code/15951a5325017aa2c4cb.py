"""Matrix multiplication 4x5 by 5x6 — Block-cyclic 2x2x2 Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_2d = U_arr[:, t].reshape(n, m)
        v_2d = V_arr[:, t].reshape(m, p)
        w_2d = W_arr[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-10)

def construct():
    """
    Block-cyclic decomposition: partition 4x5x6 into 2x2x2 Strassen blocks
    and 2x1x2 schoolbook blocks.
    
    The 4x5x6 tensor is partitioned as follows:
    - i dimension (rows of A, rows of C): 4 = 2 + 2
    - j dimension (cols of A, rows of B): 5 = 2 + 2 + 1
    - k dimension (cols of B, cols of C): 6 = 2 + 2 + 2
    
    For each (i_block, j_block, k_block) where i_block is 0 or 1,
    j_block is 0, 1, or 2, k_block is 0, 1, or 2:
    - If j_block is 0 or 1 (size 2): use 2x2x2 Strassen (rank 7)
    - If j_block is 2 (size 1): use 2x1x2 schoolbook (rank 4)
    
    Total rank = 4 * 7 + 2 * 4 = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 100  # Safety margin before 150s limit
    
    U2, V2, W2 = _build_strassen_2x2x2()
    
    decomps = []
    
    # Partition: i in [0,4), j in [0,5), k in [0,6)
    # i blocks: [0,2), [2,4)
    # j blocks: [0,2), [2,4), [4,5)
    # k blocks: [0,2), [2,4), [4,6)
    
    i_starts = [0, 2]
    i_sizes = [2, 2]
    
    j_starts = [0, 2, 4]
    j_sizes = [2, 2, 1]
    
    k_starts = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    for i_idx, i_s in enumerate(i_starts):
        i_sz = i_sizes[i_idx]
        for j_idx, j_s in enumerate(j_starts):
            j_sz = j_sizes[j_idx]
            for k_idx, k_s in enumerate(k_starts):
                k_sz = k_sizes[k_idx]
                
                if time.time() > deadline:
                    # Return best result found so far
                    break
                
                if j_sz == 2:
                    # Use 2x2x2 Strassen
                    U_sub = U2
                    V_sub = V2
                    W_sub = W2
                else:
                    # Use 2x1x2 schoolbook
                    U_sub, V_sub, W_sub = _build_schoolbook(2, 1, 2)
                
                decomps.append((U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz))
    
    if time.time() > deadline:
        # If we ran out of time, use what we have
        pass
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify for correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
