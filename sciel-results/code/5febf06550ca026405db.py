"""Improved program for the suite task; uses Strassen for 2x2 blocks where possible."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists (or numpy arrays, but let's stick to lists for compatibility and simplicity, 
    # then convert to lists at the end if needed, though the prompt says "factor matrices of shapes... with integer/half-integer entries").
    # The seed uses lists of lists. I will use lists of lists to be safe.
    
    U = [[0] * (n*m*p) for _ in range(n*m)]  # Max rank is schoolbook
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    rank = 0
    
    # We need to cover all (i, j, k) indices.
    # A standard block decomposition approach:
    # We can partition the dimensions into 2s and 1s.
    # For each block of size h_i x h_j x h_k, if h_i=2, h_j=2, h_k=2, use Strassen (rank 7).
    # Otherwise, use Schoolbook (rank h_i*h_j*h_k).
    
    # Let's partition n=3 into [2, 1], m=5 into [2, 2, 1], p=7 into [2, 2, 2, 1].
    # This is just one possible partition. We want to maximize the number of 2x2x2 blocks.
    
    # Let's define the blocks.
    # n blocks: [2, 1] -> offsets [0, 2]
    # m blocks: [2, 2, 1] -> offsets [0, 2, 4]
    # p blocks: [2, 2, 2, 1] -> offsets [0, 2, 4, 6]
    
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2, 1]
    
    # Calculate offsets
    n_offsets = []
    cur = 0
    for b in n_blocks:
        n_offsets.append(cur)
        cur += b
    m_offsets = []
    cur = 0
    for b in m_blocks:
        m_offsets.append(cur)
        cur += b
    p_offsets = []
    cur = 0
    for b in p_blocks:
        p_offsets.append(cur)
        cur += b
        
    # Iterate over all block combinations
    for i_idx, (i_off, i_size) in enumerate(zip(n_offsets, n_blocks)):
        for j_idx, (j_off, j_size) in enumerate(zip(m_offsets, m_blocks)):
            for k_idx, (k_off, k_size) in enumerate(zip(p_offsets, p_blocks)):
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen
                    # Local indices u,v,w in {0,1}
                    # Global i = i_off + u, j = j_off + v, k = k_off + w
                    
                    # P1 = (A00 + A11)(B00 + B11)
                    # U: A00, A11
                    # V: B00, B11
                    # W: C00, C11 (with signs +1, +1)
                    
                    # Let's define the 7 Strassen products explicitly with their U, V, W contributions.
                    # U entries are added to U[i*m + j, rank]
                    # V entries are added to V[j*p + k, rank]
                    # W entries are added to W[k*n + i, rank]
                    
                    # Product 1: (A00 + A11)(B00 + B11)
                    # U: (i_off, j_off): +1, (i_off+1, j_off+1): +1
                    # V: (j_off, k_off): +1, (j_off+1, k_off+1): +1
                    # W: (k_off, i_off): +1, (k_off+1, i_off+1): +1
                    U[i_off*m + j_off][rank] += 1
                    U[(i_off+1)*m + (j_off+1)][rank] += 1
                    V[j_off*p + k_off][rank] += 1
                    V[(j_off+1)*p + (k_off+1)][rank] += 1
                    W[k_off*n + i_off][rank] += 1
                    W[(k_off+1)*n + (i_off+1)][rank] += 1
                    rank += 1
                    
                    # Product 2: (A10 + A11)B00
                    # U: (i_off+1, j_off): +1, (i_off+1, j_off+1): +1
                    # V: (j_off, k_off): +1
                    # W: (k_off, i_off+1): +1
                    U[(i_off+1)*m + j_off][rank] += 1
                    U[(i_off+1)*m + (j_off+1)][rank] += 1
                    V[j_off*p + k_off][rank] += 1
                    W[k_off*n + (i_off+1)][rank] += 1
                    rank += 1
                    
                    # Product 3: A00(B01 - B11)
                    # U: (i_off, j_off): +1
                    # V: (j_off, k_off+1): +1, (j_off+1, k_off+1): -1
                    # W: (k_off+1, i_off): +1
                    U[i_off*m + j_off][rank] += 1
                    V[j_off*p + (k_off+1)][rank] += 1
                    V[(j_off+1)*p + (k_off+1)][rank] -= 1
                    W[(k_off+1)*n + i_off][rank] += 1
                    rank += 1
                    
                    # Product 4: A11(B10 - B00)
                    # U: (i_off+1, j_off+1): +1
                    # V: (j_off+1, k_off): +1, (j_off, k_off): -1
                    # W: (k_off, i_off+1): +1
                    U[(i_off+1)*m + (j_off+1)][rank] += 1
                    V[(j_off+1)*p + k_off][rank] += 1
                    V[j_off*p + k_off][rank] -= 1
                    W[k_off*n + (i_off+1)][rank] += 1
                    rank += 1
                    
                    # Product 5: (A00 + A01)B11
                    # U: (i_off, j_off): +1, (i_off, j_off+1): +1
                    # V: (j_off+1, k_off+1): +1
                    # W: (k_off+1, i_off): +1
                    U[i_off*m + j_off][rank] += 1
                    U[i_off*m + (j_off+1)][rank] += 1
                    V[(j_off+1)*p + (k_off+1)][rank] += 1
                    W[(k_off+1)*n + i_off][rank] += 1
                    rank += 1
                    
                    # Product 6: (A10 - A00)(B00 + B01)
                    # U: (i_off+1, j_off): +1, (i_off, j_off): -1
                    # V: (j_off, k_off): +1, (j_off, k_off+1): +1
                    # W: (k_off, i_off+1): +1
                    U[(i_off+1)*m + j_off][rank] += 1
                    U[i_off*m + j_off][rank] -= 1
                    V[j_off*p + k_off][rank] += 1
                    V[j_off*p + (k_off+1)][rank] += 1
                    W[k_off*n + (i_off+1)][rank] += 1
                    rank += 1
                    
                    # Product 7: (A01 - A11)(B10 + B11)
                    # U: (i_off, j_off+1): +1, (i_off+1, j_off+1): -1
                    # V: (j_off+1, k_off): +1, (j_off+1, k_off+1): +1
                    # W: (k_off+1, i_off): +1
                    U[i_off*m + (j_off+1)][rank] += 1
                    U[(i_off+1)*m + (j_off+1)][rank] -= 1
                    V[(j_off+1)*p + k_off][rank] += 1
                    V[(j_off+1)*p + (k_off+1)][rank] += 1
                    W[(k_off+1)*n + i_off][rank] += 1
                    rank += 1
                    
                else:
                    # Schoolbook
                    for u in range(i_size):
                        for v in range(j_size):
                            for w in range(k_size):
                                i = i_off + u
                                j = j_off + v
                                k = k_off + w
                                U[i*m + j][rank] += 1
                                V[j*p + k][rank] += 1
                                W[k*n + i][rank] += 1
                                rank += 1
                                
    # Trim U, V, W to the actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

