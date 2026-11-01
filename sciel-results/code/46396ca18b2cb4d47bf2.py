"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is a list of (local_idx, coeff) for U, V, W
    # U local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # M2
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        # M3
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        # M4
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(2, 1), (0, 1)]
        },
        # M5
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # M6
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # M7
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]
    
    col = 0
    for block in range(2): # 0: cols 0,1; 1: cols 2,3
        for sub in range(3): # 0: k 0,1; 1: k 2,3; 2: k 4,5
            for s_idx in range(7):
                term = strassen_terms[s_idx]
                
                # Update U
                for local_u, coeff in term['U']:
                    a = local_u // 2
                    l = local_u % 2
                    global_u = a * m + (2 * block + l)
                    U[global_u][col] += coeff
                
                # Update V
                for local_v, coeff in term['V']:
                    l = local_v // 2
                    c = local_v % 2
                    global_v = (2 * block + l) * p + (2 * sub + c)
                    V[global_v][col] += coeff
                
                # Update W
                for local_w, coeff in term['W']:
                    a = local_w // 2
                    c = local_w % 2
                    global_w = (2 * sub + c) * n + a
                    W[global_w][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
