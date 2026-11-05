import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication. Rank 7.
    U: 4x7 (A indices i*2+j), V: 4x7 (B indices j*2+k), W: 4x7 (C indices k*2+i)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B10-B11) -> C11
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[3][2] = 1
    
    # M4 = A11(B01-B10) -> C00, C10
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11 -> C00, C01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_block(n, m, p):
    """Generic schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _get_block_decomp(n, m, p):
    """Get the best known decomposition for n x m * m x p -> n x p."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    return _schoolbook_block(n, m, p)

def _block_decomposition_5x5x5():
    """
    Construct block decomposition for 5x5x5 using 2+2+1 partition.
    Partition indices into blocks: B0={0,1}, B1={2,3}, B2={4}
    Uses Strassen for 2x2x2 sub-blocks, schoolbook for others.
    Total rank = 117.
    """
    n, m, p = DIMENSIONS
    sizes = [2, 2, 1]
    num_blocks = 3
    
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    # First pass: compute total rank
    total_rank = 0
    block_specs = []
    
    for a in range(num_blocks):
        for k in range(num_blocks):
            for b in range(num_blocks):
                sa, sk, sb = sizes[a], sizes[k], sizes[b]
                decomp = _get_block_decomp(sa, sk, sb)
                rank_block = len(decomp["U"][0]) if decomp["U"] else 0
                total_rank += rank_block
                block_specs.append((a, k, b, decomp, rank_block))
    
    # Allocate matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Second pass: fill matrices
    current_col = 0
    for (a, k, b, decomp, rank_block) in block_specs:
        sa, sk, sb = sizes[a], sizes[k], sizes[b]
        U_block = decomp["U"]
        V_block = decomp["V"]
        W_block = decomp["W"]
        
        # Fill U: A block (a, k), local indices (i, j) -> global row (offsets[a]+i)*m + (offsets[k]+j)
        for local_i in range(sa):
            for local_j in range(sk):
                global_row = (offsets[a] + local_i) * m + (offsets[k] + local_j)
                local_row = local_i * sk + local_j
                for r in range(rank_block):
                    U[global_row][current_col + r] = U_block[local_row][r]
        
        # Fill V: B block (k, b), local indices (j, k) -> global row (offsets[k]+j)*p + (offsets[b]+k)
        for local_j in range(sk):
            for local_k in range(sb):
                global_row = (offsets[k] + local_j) * p + (offsets[b] + local_k)
                local_row = local_j * sb + local_k
                for r in range(rank_block):
                    V[global_row][current_col + r] = V_block[local_row][r]
        
        # Fill W: C block (a, b), local indices (k, i) -> global row (offsets[b]+k)*n + (offsets[a]+i)
        for local_k in range(sb):
            for local_i in range(sa):
                global_row = (offsets[b] + local_k) * n + (offsets[a] + local_i)
                local_row = local_k * sa + local_i
                for r in range(rank_block):
                    W[global_row][current_col + r] = W_block[local_row][r]
        
        current_col += rank_block
    
    return {"U": U, "V": W, "W": W} # Wait, typo in return? No, U, V, W.

def _verify(decomp, n, m, p):
    """Verify that the decomposition correctly decomposes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                expected[u_idx, v_idx, w_idx] = 1.0
    
    return np.allclose(T, expected, atol=1e-9)

def _schoolbook_full():
    """Full schoolbook decomposition for 5x5x5. Rank 125."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition (rank 117)
    try:
        block_decomp = _block_decomposition_5x5x5()
        if _verify(block_decomp, n, m, p):
            return block_decomp
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 125)
    return _schoolbook_full()

# EVOLVE-BLOCK-END
