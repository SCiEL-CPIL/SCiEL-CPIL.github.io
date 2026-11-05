"""
Program for 6x6x6 Matrix Multiplication Tensor Decomposition.
Uses the tensor product of the schoolbook 3x3x3 decomposition (rank 27) 
and the Strassen 2x2x2 decomposition (rank 7) to achieve rank 189.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the rank-189 decomposition 
    of the 6x6 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # Indices for 2x2: 0,1,2,3 correspond to [[0,1],[2,3]]
    # u: A indices, v: B indices, w: C indices
    strassen_triples = [
        ([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Schoolbook 3x3x3 decomposition (Rank 27)
    # Indices for 3x3: 0..8 correspond to row-major 3x3
    schoolbook_triples = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0] * 9
                v = [0] * 9
                w = [0] * 9
                u[i * 3 + j] = 1
                v[j * 3 + k] = 1
                w[k * 3 + i] = 1
                schoolbook_triples.append((u, v, w))
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for u3, v3, w3 in schoolbook_triples:
        for u2, v2, w2 in strassen_triples:
            # Combine indices
            # U index: i3*3+j3 (from u3) and i2*2+j2 (from u2)
            # Full U index: (i3*2+i2)*6 + (j3*2+j2)
            # We iterate over all non-zero entries in u3 and u2
            # Since u3 and u2 are sparse (only one non-zero for schoolbook, 
            # but multiple for Strassen), we just multiply the vectors.
            
            # u3 has one 1 at idx3. u2 has multiple non-zeros.
            # The combined U vector will have non-zeros where u3 is non-zero AND u2 is non-zero.
            # Actually, the tensor product logic:
            # U6[I, J, col] = U3[i3, j3, r3] * U2[i2, j2, r2]
            # I = i3*2 + i2, J = j3*2 + j2
            
            # u3 is a list of 9. u2 is a list of 4.
            # We need to map (i3, j3) from u3 index and (i2, j2) from u2 index.
            
            # Iterate through all possible (i3, j3) for u3
            for idx3 in range(9):
                if u3[idx3] == 0: continue
                i3 = idx3 // 3
                j3 = idx3 % 3
                val3 = u3[idx3]
                
                # Iterate through all possible (i2, j2) for u2
                for idx2 in range(4):
                    if u2[idx2] == 0: continue
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    val2 = u2[idx2]
                    
                    # Full indices
                    I = i3 * 2 + i2
                    J = j3 * 2 + j2
                    U_idx = I * 6 + J
                    
                    U[U_idx][col] += val3 * val2
                    
            # Same for V
            for idx3 in range(9):
                if v3[idx3] == 0: continue
                j3 = idx3 // 3
                k3 = idx3 % 3
                val3 = v3[idx3]
                
                for idx2 in range(4):
                    if v2[idx2] == 0: continue
                    j2 = idx2 // 2
                    k2 = idx2 % 2
                    val2 = v2[idx2]
                    
                    # Full indices for V: J, K
                    J = j3 * 2 + j2
                    K = k3 * 2 + k2
                    V_idx = J * 6 + K
                    
                    V[V_idx][col] += val3 * val2
                    
            # Same for W
            for idx3 in range(9):
                if w3[idx3] == 0: continue
                k3 = idx3 // 3
                i3 = idx3 % 3
                val3 = w3[idx3]
                
                for idx2 in range(4):
                    if w2[idx2] == 0: continue
                    k2 = idx2 // 2
                    i2 = idx2 % 2
                    val2 = w2[idx2]
                    
                    # Full indices for W: K, I
                    K = k3 * 2 + k2
                    I = i3 * 2 + i2
                    W_idx = K * 6 + I
                    
                    W[W_idx][col] += val3 * val2
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
