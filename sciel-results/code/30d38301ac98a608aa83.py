import numpy as np

def construct():
    n, m, p = 2, 4, 8
    # Calculate flattening rank as per directive
    # Tensor T has shape (n*m, m*p, p*n) ? 
    # Actually, the tensor indices are (i, j) for A, (j, k) for B, (k, i) for C?
    # Let's define the tensor T_{alpha, beta, gamma}
    # alpha in 0..n*m-1 (A)
    # beta in 0..m*p-1 (B)
    # gamma in 0..p*n-1 (C)
    
    # The tensor is 1 if the indices are compatible, 0 otherwise.
    # Compatibility: A_{ij}, B_{jk}, C_{ik} (or C_{ki} depending on W index)
    # W index is k*n + i. So C_{ik} is at index k*n+i.
    # A index is i*m + j.
    # B index is j*p + k.
    
    # Flattening over A and B modes:
    # Matrix M of size (n*m) x (m*p * p*n)
    # M[alpha, (beta, gamma)] = T_{alpha, beta, gamma}
    
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    T = np.zeros((size_a, size_b, size_c))
    
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            for k in range(p):
                b_idx = j * p + k
                c_idx = k * n + i # W[k*n+i]
                T[a_idx, b_idx, c_idx] = 1.0
                
    # Flatten over last two modes
    M_flat = T.reshape(size_a, size_b * size_c)
    rank_flattening = np.linalg.matrix_rank(M_flat)
    
    # The directive suggests this rank is the minimal rank.
    # However, for matrix multiplication tensors, CP rank > Matrix Rank usually.
    # We will use the known valid Strassen decomposition with rank 56.
    # If rank_flattening < 56, we might try to use it, but it's risky.
    # We stick to 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_terms = [
        # P1
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    for k_block in range(4):
        for term in strassen_terms:
            for (i, jl, val) in term["U"]:
                jg = jl
                U[i*4 + jg][col] += val
            for (jl, kl, val) in term["V"]:
                jg = jl
                kg = 2*k_block + kl
                V[jg*8 + kg][col] += val
            for (i, kl, val) in term["W"]:
                kg = 2*k_block + kl
                W[kg*2 + i][col] += val
            col += 1
            
    for k_block in range(4):
        for term in strassen_terms:
            for (i, jl, val) in term["U"]:
                jg = 2 + jl
                U[i*4 + jg][col] += val
            for (jl, kl, val) in term["V"]:
                jg = 2 + jl
                kg = 2*k_block + kl
                V[jg*8 + kg][col] += val
            for (i, kl, val) in term["W"]:
                kg = 2*k_block + kl
                W[kg*2 + i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
