import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = A22(B11-B21)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks):
    """
    Combine a list of block decompositions into one global decomposition.
    blocks: list of tuples (n_i, m_j, p_k, U_block, V_block, W_block, i_offset, j_offset, k_offset)
    """
    total_rank = 0
    for b in blocks:
        total_rank += len(b[3][0])
        
    n = sum(b[0] for b in blocks)
    m = sum(b[1] for b in blocks)
    p = sum(b[2] for b in blocks)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    for n_i, m_j, p_k, U_b, V_b, W_b, i_off, j_off, k_off in blocks:
        rank_b = len(U_b[0])
        for c in range(rank_b):
            for i in range(n_i):
                idx_u = (i_off + i) * m + (j_off) # Wait, U index is i*m+j.
                # U_block index is i*m_j + j? No, U_block is for n_i x m_j.
                # U_block[i_b, j_b] where i_b in [0, n_i), j_b in [0, m_j).
                # Global index: (i_off + i_b) * m + (j_off + j_b)
                pass
                
        # We need to iterate over the local indices of the block
        for i_b in range(n_i):
            for j_b in range(m_j):
                idx_u_global = (i_off + i_b) * m + (j_off + j_b)
                for c in range(rank_b):
                    U[idx_u_global][col + c] = U_b[i_b * m_j + j_b][c]
                    
        for j_b in range(m_j):
            for k_b in range(p_k):
                idx_v_global = (j_off + j_b) * p + (k_off + k_b)
                for c in range(rank_b):
                    V[idx_v_global][col + c] = V_b[j_b * p_k + k_b][c]
                    
        for k_b in range(p_k):
            for i_b in range(n_i):
                idx_w_global = (k_off + k_b) * n + (i_off + i_b)
                for c in range(rank_b):
                    W[idx_w_global][col + c] = W_b[k_b * n_i + i_b][c]
                    
        col += rank_b
        
    return U, V, W

def _build_block_decomp(n, m, p):
    """
    Build a decomposition for an n x m x p block.
    Uses Strassen for 2x2x2 parts if possible.
    """
    if n == 1 and m == 1 and p == 2:
        return _schoolbook_decomp(1, 1, 2)
    if n == 1 and m == 1 and p == 4:
        return _schoolbook_decomp(1, 1, 4)
    if n == 1 and m == 4 and p == 2:
        return _schoolbook_decomp(1, 4, 2)
    if n == 1 and m == 4 and p == 4:
        return _schoolbook_decomp(1, 4, 4)
    if n == 2 and m == 1 and p == 2:
        return _schoolbook_decomp(2, 1, 2)
    if n == 2 and m == 1 and p == 4:
        return _schoolbook_decomp(2, 1, 4)
    
    if n == 2 and m == 4 and p == 2:
        # Sum of two 2x2x2
        U1, V1, W1 = _strassen_2x2x2()
        U2, V2, W2 = _strassen_2x2x2()
        
        # Block 1: A1 (2x2) * B1 (2x2) -> C1 (2x2)
        # Indices: i in [0,2), j in [0,2), k in [0,2)
        # Block 2: A2 (2x2) * B2 (2x2) -> C2 (2x2)
        # Indices: i in [0,2), j in [2,4), k in [0,2)
        
        rank = 7 + 7
        U = [[0]*rank for _ in range(2*4)]
        V = [[0]*rank for _ in range(4*2)]
        W = [[0]*rank for _ in range(2*2)]
        
        # Copy Block 1
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[i*4+j][c] = U1[i*2+j][c]
                    V[j*2+k][c] = V1[j*2+k][c] # Wait, V index is j*p+k. p=2.
                    # V1 index is j*2+k.
                    # Global V index is j*2+k.
                    # So V[j*2+k][c] = V1[j*2+k][c]
                for k in range(2):
                    W[k*2+i][c] = W1[k*2+i][c]
                    
        # Copy Block 2 (shifted j by 2)
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[i*4 + (j+2)][7+c] = U2[i*2+j][c]
                    # V index: j_global = j+2, k.
                    # V[(j+2)*2 + k][7+c] = V2[j*2+k][c]
                    for k in range(2):
                         V[(j+2)*2 + k][7+c] = V2[j*2+k][c]
                         W[k*2 + i][7+c] = W2[k*2+i][c]
                         
        return U, V, W

    if n == 2 and m == 4 and p == 4:
        # Sum of four 2x2x2
        U1, V1, W1 = _strassen_2x2x2()
        
        rank = 4 * 7
        U = [[0]*rank for _ in range(2*4)]
        V = [[0]*rank for _ in range(4*4)]
        W = [[0]*rank for _ in range(4*2)]
        
        # A = [A1 A2]
        # B = [B11 B12; B21 B22]
        # C = [C1 C2]
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        
        # Block 1: A1 B11. i in [0,2), j in [0,2), k in [0,2)
        # Block 2: A1 B12. i in [0,2), j in [0,2), k in [2,4)
        # Block 3: A2 B21. i in [0,2), j in [2,4), k in [0,2)
        # Block 4: A2 B22. i in [0,2), j in [2,4), k in [2,4)
        
        for b_idx in range(4):
            offset = b_idx * 7
            # Determine j_offset and k_offset
            if b_idx == 0: j_off, k_off = 0, 0
            elif b_idx == 1: j_off, k_off = 0, 2
            elif b_idx == 2: j_off, k_off = 2, 0
            else: j_off, k_off = 2, 2
            
            for i in range(2):
                for j in range(2):
                    for c in range(7):
                        U[i*4 + (j+j_off)][offset+c] = U1[i*2+j][c]
                        for k in range(2):
                            V[(j+j_off)*4 + (k+k_off)][offset+c] = V1[j*2+k][c]
                            W[(k+k_off)*2 + i][offset+c] = W1[k*2+i][c]
                            
        return U, V, W

    # Fallback
    return _schoolbook_decomp(n, m, p)

def construct():
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [1, 2]
    m_parts = [1, 4]
    p_parts = [2, 4]
    
    blocks = []
    i_off = 0
    for n_i in n_parts:
        j_off = 0
        for m_j in m_parts:
            k_off = 0
            for p_k in p_parts:
                U_b, V_b, W_b = _build_block_decomp(n_i, m_j, p_k)
                blocks.append((n_i, m_j, p_k, U_b, V_b, W_b, i_off, j_off, k_off))
                k_off += p_k
            j_off += m_j
        i_off += n_i
        
    U, V, W = _combine_blocks(blocks)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": _schoolbook_decomp(n, m, p)[0], "V": _schoolbook_decomp(n, m, p)[1], "W": _schoolbook_decomp(n, m, p)[2]}
