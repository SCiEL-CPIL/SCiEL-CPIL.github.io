"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """Strassen-based rank-35 decomposition for 2x2 times 2x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    # 5 blocks of 2 columns each, Strassen rank 7 per block
    num_blocks = p // 2  # 5
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # shape (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # shape (20, 35)
    
    # Strassen coefficients for 2x2 * 2x2
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  (i*m+j)
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  (j*p_local+k_local)
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  (k_local*n+i)
    
    # Each Strassen term: list of (index, coeff) for U, V, W
    # U index = i*m + j, V index = j*2 + k_local, W index = k_local*n + i
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {
            'U': [(0, 1), (3, 1)],       # a, d
            'V': [(0, 1), (3, 1)],       # e, h (j=0,k=0 and j=1,k=1)
            'W': [(0, 1), (3, 1)],       # C[0,0], C[1,1] (k=0,i=0 and k=1,i=1)
        },
        # M2 = (c+d)e
        {
            'U': [(2, 1), (3, 1)],       # c, d
            'V': [(0, 1)],               # e (j=0,k=0)
            'W': [(2, 1), (3, -1)],      # C[1,0], C[1,1]
        },
        # M3 = a(f-h)
        {
            'U': [(0, 1)],               # a
            'V': [(1, 1), (3, -1)],      # f, h (j=0,k=1 and j=1,k=1)
            'W': [(0, -1), (1, 1)],      # C[0,0], C[0,1]
        },
        # M4 = d(g-e)
        {
            'U': [(3, 1)],               # d
            'V': [(2, 1), (0, -1)],      # g, e (j=1,k=0 and j=0,k=0)
            'W': [(0, 1), (2, 1)],       # C[0,0], C[1,0]
        },
        # M5 = (a+b)h
        {
            'U': [(0, 1), (1, 1)],       # a, b
            'V': [(3, 1)],               # h (j=1,k=1)
            'W': [(0, -1), (1, 1)],      # C[0,0], C[0,1]
        },
        # M6 = (c-a)(e+f)
        {
            'U': [(2, 1), (0, -1)],      # c, a
            'V': [(0, 1), (1, 1)],       # e, f (j=0,k=0 and j=0,k=1)
            'W': [(3, 1)],               # C[1,1]
        },
        # M7 = (b-d)(g+h)
        {
            'U': [(1, 1), (3, -1)],      # b, d
            'V': [(2, 1), (3, 1)],       # g, h (j=1,k=0 and j=1,k=1)
            'W': [(0, 1), (3, -1)],      # C[0,0], C[1,1]
        },
    ]
    
    for block in range(num_blocks):
        col_offset = block * 2  # starting column in B and C
        for s in range(7):
            col = block * 7 + s
            term = strassen_terms[s]
            
            for idx, coeff in term['U']:
                U[idx][col] += coeff
            
            for idx, coeff in term['V']:
                # V index: j*2 + k_local, but globally j*p + (col_offset + k_local)
                # idx = j*2 + k_local, so j = idx // 2, k_local = idx % 2
                j = idx // 2
                k_local = idx % 2
                v_idx = j * p + col_offset + k_local
                V[v_idx][col] += coeff
            
            for idx, coeff in term['W']:
                # W index: k_local*n + i, globally (col_offset + k_local)*n + i
                # idx = k_local*n + i, so k_local = idx // n, i = idx % n
                k_local = idx // n
                i = idx % n
                w_idx = (col_offset + k_local) * n + i
                W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
