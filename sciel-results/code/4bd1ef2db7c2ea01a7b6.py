"""Rewritten program using Strassen's algorithm for rank 56."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # We split p=16 into 8 blocks of size 2.
    # Each block uses Strassen's rank 7 algorithm for 2x2 * 2x2.
    # Total rank = 8 * 7 = 56.
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 4 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Coefficients for Strassen's 7 products
    # u_s[i,j] where i in 0..1, j in 0..1
    # Index in U is i*m + j = i*2 + j
    u_coeffs = [
        [1, 0, 0, 1],  # M0: (A00+A11)
        [0, 0, 1, 1],  # M1: (A10+A11)
        [1, 0, 0, 0],  # M2: A00
        [0, 0, 0, 1],  # M3: A11
        [1, 1, 0, 0],  # M4: (A00+A01)
        [-1, 0, 1, 0], # M5: (A10-A00)
        [0, 1, 0, -1]  # M6: (A01-A11)
    ]
    
    # v_coeffs[j, k'] where j in 0..1, k' in 0..1
    # Index in V is j*p + k. For block t, k = 2*t + k'.
    # So index is j*16 + 2*t + k'.
    v_coeffs = [
        [(1, 0), (0, 1)], # M0: B00=1, B11=1. (j=0,k'=0)=1, (j=1,k'=1)=1
        [(1, 0), (0, 0)], # M1: B00=1
        [(0, 1), (0, -1)],# M2: B01=1, B11=-1
        [(-1, 0), (1, 0)],# M3: B00=-1, B10=1
        [(0, 0), (0, 1)], # M4: B11=1
        [(1, 1), (0, 0)], # M5: B00=1, B01=1
        [(0, 0), (1, 1)]  # M6: B10=1, B11=1
    ]
    # Note: v_coeffs[s] is a list of 2 tuples (for j=0 and j=1), each tuple is (val_k0, val_k1)
    
    # w_coeffs[i, k'] where i in 0..1, k' in 0..1
    # Index in W is k*n + i = k*2 + i. For block t, k = 2*t + k'.
    # So index is (2*t + k')*2 + i.
    # We define w_coeffs[s][i][k']
    w_coeffs = [
        [[1, 0], [0, 1]], # M0: C00=1, C11=1
        [[0, 0], [1, -1]],# M1: C10=1, C11=-1
        [[0, 1], [-1, 0]],# M2: C01=1, C10=-1
        [[-1, 0], [1, 0]],# M3: C00=-1, C01=1
        [[1, 0], [1, 0]], # M4: C00=1, C10=1
        [[0, 0], [-1, 1]],# M5: C10=-1, C11=1
        [[-1, 0], [0, -1]] # M6: C00=-1, C11=-1
    ]

    num_blocks = p // 2 # 8 blocks
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            
            # Fill U
            for idx in range(4):
                U[idx][col] = u_coeffs[s][idx]
                
            # Fill V
            for j in range(2):
                for kp in range(2):
                    val = v_coeffs[s][j][kp]
                    if val != 0:
                        v_idx = j * p + (2 * t + kp)
                        V[v_idx][col] = val
                        
            # Fill W
            for i in range(2):
                for kp in range(2):
                    val = w_coeffs[s][i][kp]
                    if val != 0:
                        k = 2 * t + kp
                        w_idx = k * n + i
                        W[w_idx][col] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
