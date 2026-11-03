"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2x2():
    """Return the 7 rank-1 components for 2x2x2 matrix multiplication (Strassen)."""
    # Each component: (u, v, w) where u,v,w are 2x2 matrices
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U matrix: maps A (2x2) to the 7 intermediate products' A-side
    # V matrix: maps B (2x2) to the 7 intermediate products' B-side  
    # W matrix: maps the 7 intermediates to C (2x2)
    
    components = []
    
    # M1: (A11+A22) * (B11+B22)
    components.append({
        'U': [[1, 0], [0, 1]],   # A11 + A22
        'V': [[1, 0], [0, 1]],   # B11 + B22
        'W': [[1, 0], [0, 1]]    # contributes +1 to C11, +1 to C22
    })
    
    # M2: (A21+A22) * B11
    components.append({
        'U': [[0, 0], [1, 1]],   # A21 + A22
        'V': [[1, 0], [0, 0]],   # B11
        'W': [[0, 0], [1, 0]]    # contributes +1 to C21
    })
    
    # M3: A11 * (B12-B22)
    components.append({
        'U': [[1, 0], [0, 0]],   # A11
        'V': [[0, 1], [0, -1]],  # B12 - B22
        'W': [[0, 1], [0, 0]]    # contributes +1 to C12
    })
    
    # M4: A22 * (B21-B11)
    components.append({
        'U': [[0, 0], [0, 1]],   # A22
        'V': [[-1, 0], [0, 1]],  # B21 - B11
        'W': [[1, 0], [0, 1]]    # contributes +1 to C11, +1 to C22
    })
    
    # M5: (A11+A12) * B22
    components.append({
        'U': [[1, 1], [0, 0]],   # A11 + A12
        'V': [[0, 0], [0, 1]],   # B22
        'W': [[-1, 0], [0, 1]]   # contributes -1 to C11, +1 to C22
    })
    
    # M6: (A21-A11) * (B11+B12)
    components.append({
        'U': [[-1, 0], [1, 0]],  # A21 - A11
        'V': [[1, 1], [0, 0]],   # B11 + B12
        'W': [[0, 0], [0, 1]]    # contributes +1 to C22
    })
    
    # M7: (A12-A22) * (B21+B22)
    components.append({
        'U': [[0, 1], [0, -1]],  # A12 - A22
        'V': [[0, 0], [1, 1]],   # B21 + B22
        'W': [[1, 0], [0, 0]]    # contributes +1 to C11
    })
    
    return components

def _kron_4x4x4(strassen_components):
    """Build rank-49 decomposition for 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    # Kronecker product: for each pair of 2x2x2 components, create a 4x4x4 component
    n_comp = len(strassen_components)
    total_rank = n_comp * n_comp  # 7*7 = 49
    
    U = [[0] * total_rank for _ in range(16)]  # 4x4 = 16 entries
    V = [[0] * total_rank for _ in range(16)]  # 4x4 = 16 entries
    W = [[0] * total_rank for _ in range(16)]  # 4x4 = 16 entries
    
    col = 0
    for i in range(n_comp):
        for j in range(n_comp):
            u1, v1, w1 = strassen_components[i]['U'], strassen_components[i]['V'], strassen_components[i]['W']
            u2, v2, w2 = strassen_components[j]['U'], strassen_components[j]['V'], strassen_components[j]['W']
            
            # Kronecker product of 2x2 matrices
            uk = [[u1[0][0]*u2[0][0], u1[0][0]*u2[0][1], u1[0][1]*u2[0][0], u1[0][1]*u2[0][1]],
                  [u1[0][0]*u2[1][0], u1[0][0]*u2[1][1], u1[0][1]*u2[1][0], u1[0][1]*u2[1][1]],
                  [u1[1][0]*u2[0][0], u1[1][0]*u2[0][1], u1[1][1]*u2[0][0], u1[1][1]*u2[0][1]],
                  [u1[1][0]*u2[1][0], u1[1][0]*u2[1][1], u1[1][1]*u2[1][0], u1[1][1]*u2[1][1]]]
            
            vk = [[v1[0][0]*v2[0][0], v1[0][0]*v2[0][1], v1[0][1]*v2[0][0], v1[0][1]*v2[0][1]],
                  [v1[0][0]*v2[1][0], v1[0][0]*v2[1][1], v1[0][1]*v2[1][0], v1[0][1]*v2[1][1]],
                  [v1[1][0]*v2[0][0], v1[1][0]*v2[0][1], v1[1][1]*v2[0][0], v1[1][1]*v2[0][1]],
                  [v1[1][0]*v2[1][0], v1[1][0]*v2[1][1], v1[1][1]*v2[1][0], v1[1][1]*v2[1][1]]]
            
            wk = [[w1[0][0]*w2[0][0], w1[0][0]*w2[0][1], w1[0][1]*w2[0][0], w1[0][1]*w2[0][1]],
                  [w1[0][0]*w2[1][0], w1[0][0]*w2[1][1], w1[0][1]*w2[1][0], w1[0][1]*w2[1][1]],
                  [w1[1][0]*w2[0][0], w1[1][0]*w2[0][1], w1[1][1]*w2[0][0], w1[1][1]*w2[0][1]],
                  [w1[1][0]*w2[1][0], w1[1][0]*w2[1][1], w1[1][1]*w2[1][0], w1[1][1]*w2[1][1]]]
            
            for r in range(4):
                for c in range(4):
                    idx = r * 4 + c
                    U[idx][col] = uk[r][c]
                    V[idx][col] = vk[r][c]
                    W[idx][col] = wk[r][c]
            col += 1
    
    return U, V, W

def _block_4x4x1():
    """Build rank-4 decomposition for 4x4x1 (A(4x4) * B(4x1) -> C(4x1))."""
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # This is a rank-4 tensor: one component per j
    rank = 4
    U = [[0] * rank for _ in range(16)]  # 4x4
    V = [[0] * rank for _ in range(4)]   # 4x1
    W = [[0] * rank for _ in range(4)]   # 1x4
    
    for j in range(4):
        U[j][j] = 1  # Wait, need to be careful with indexing
        # A is 4x4, indexed as A[i,j] -> flat index i*4+j
        # B is 4x1, indexed as B[j,0] -> flat index j
        # C is 1x4, indexed as C[0,i] -> flat index i (since p*n = 1*4 = 4, W[k*n+i] = W[0*4+i] = W[i])
        
        # For component j: U picks A[:,j], V picks B[j,0], W puts result in C[0,:]
        for i in range(4):
            U[i * 4 + j][j] = 1
        V[j][j] = 1
        for i in range(4):
            W[i][j] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Get Strassen components for 2x2x2
    strassen = _strassen_2x2x2()
    
    # Build 4x4x4 decomposition (rank 49)
    U_444, V_444, W_444 = _kron_4x4x4(strassen)
    
    # Build 4x4x1 decomposition (rank 4)
    U_441, V_441, W_441 = _block_4x4x1()
    
    # Now assemble the full 4x4x9 decomposition
    # A is 4x4 (16 entries), B is 4x9 (36 entries), C is 9x4 (36 entries)
    # U: (n*m, rank) = (16, total_rank)
    # V: (m*p, rank) = (36, total_rank) 
    # W: (p*n, rank) = (36, total_rank)
    
    rank_444 = 49
    rank_441 = 4
    total_rank = 2 * rank_444 + rank_441  # 49 + 49 + 4 = 102
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Block 1: columns 0-3 of B (4x4 sub-block)
    # B columns 0-3: flat indices 0-15 (j*9+k for j in 0-3, k in 0-3)
    # C columns 0-3: flat indices 0-15 (k*4+i for k in 0-3, i in 0-3)
    for c in range(rank_444):
        for idx in range(16):
            U[idx][c] = U_444[idx][c]
        for j in range(4):
            for k in range(4):
                b_idx = j * 9 + k  # B[j,k] for k in 0-3
                v_idx = j * 4 + k  # index in 4x4 V matrix
                V[b_idx][c] = V_444[v_idx][c]
        for k in range(4):
            for i in range(4):
                w_idx = k * 4 + i  # C[k,i] for k in 0-3
                W[w_idx][c] = W_444[w_idx][c]
    
    # Block 2: columns 4-7 of B (4x4 sub-block)
    # B columns 4-7: flat indices j*9+k for j in 0-3, k in 4-7
    # C columns 4-7: flat indices k*4+i for k in 4-7, i in 0-3
    col_offset = rank_444
    for c in range(rank_444):
        for idx in range(16):
            U[idx][col_offset + c] = U_444[idx][c]
        for j in range(4):
            for k in range(4):
                b_idx = j * 9 + (k + 4)  # B[j,k+4]
                v_idx = j * 4 + k  # index in 4x4 V matrix
                V[b_idx][col_offset + c] = V_444[v_idx][c]
        for k in range(4):
            for i in range(4):
                w_idx = (k + 4) * 4 + i  # C[k+4,i]
                W[w_idx][col_offset + c] = W_444[w_idx][c]
    
    # Block 3: column 8 of B (4x1 sub-block)
    col_offset2 = 2 * rank_444
    for c in range(rank_441):
        for idx in range(16):
            U[idx][col_offset2 + c] = U_441[idx][c]
        for j in range(4):
            b_idx = j * 9 + 8  # B[j,8]
            V[b_idx][col_offset2 + c] = V_441[j][c]
        for i in range(4):
            w_idx = 8 * 4 + i  # C[8,i]
            W[w_idx][col_offset2 + c] = W_441[i][c]
    
    return {"U": U, "V": V, "W": W}
