DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for Strassen 2x2x2
    # Each entry is a list of (local_index, value)
    # Local indices: a,b for A (2x2), b,c for B (2x2), a,c for C (2x2)
    
    coeffs = [
        # r=0
        {
            'U': [(0,0,1), (1,1,1)], # (a, b, val)
            'V': [(0,0,1), (1,1,1)], # (b, c, val)
            'W': [(0,0,1), (1,1,1)]  # (a, c, val)
        },
        # r=1
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(1,0,1), (1,1,-1)]
        },
        # r=2
        {
            'U': [(0,0,1)],
            'V': [(1,0,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)]
        },
        # r=3
        {
            'U': [(1,1,1)],
            'V': [(0,1,1), (0,0,-1)],
            'W': [(0,0,1), (1,0,1)]
        },
        # r=4
        {
            'U': [(0,0,1), (1,0,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]
        },
        # r=5
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # r=6
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

    col = 0
    for u in range(2):
        for v in range(4):
            for r in range(7):
                c = coeffs[r]
                for a, b, val in c['U']:
                    i = a
                    j = 2 * u + b
                    U[i * m + j][col] = val
                for b, c_idx, val in c['V']:
                    j = 2 * u + b
                    k = 2 * v + c_idx
                    V[j * p + k][col] = val
                for a, c_idx, val in c['W']:
                    i = a
                    k = 2 * v + c_idx
                    W[k * n + i][col] = val
                col += 1

    return {"U": U, "V": V, "W": W}
