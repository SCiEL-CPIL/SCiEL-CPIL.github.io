import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the decomposition dynamically to determine the rank
    # But we know the rank is 46.
    # Let's just use lists of tuples to define the components, then build the matrices.
    
    components = []
    
    # Strassen block helper
    def add_strassen(k0):
        # k0 is the start column index for the 2x2 block
        # A entries: A00 (i=0,j=0), A01 (i=0,j=1), A10 (i=1,j=0), A11 (i=1,j=1)
        # B entries: B0k0 (j=0,k=k0), B0k1 (j=0,k=k0+1), B1k0 (j=1,k=k0), B1k1 (j=1,k=k0+1)
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        # alpha: A00, A11
        # beta: B0k0, B1k1
        # gamma: C0k0 (+1), C1k1 (+1)
        components.append({
            'U': [(0, 1), (3, 1)], # i*m+j: 0*2+0=0, 1*2+1=3
            'V': [(0, 1), (13 + k0 + 1, 1)], # j*p+k: 0*13+k0=k0, 1*13+k0+1=14+k0
            'W': [(k0 * 2 + 0, 1), ((k0 + 1) * 2 + 1, 1)] # k*n+i: k0*2+0, (k0+1)*2+1
        })
        
        # M2 = (A10 + A11) * B0k0
        # alpha: A10, A11
        # beta: B0k0
        # gamma: C1k0 (+1)
        components.append({
            'U': [(2, 1), (3, 1)], # 1*2+0=2, 1*2+1=3
            'V': [(k0, 1)], # 0*13+k0=k0
            'W': [(k0 * 2 + 1, 1)] # k0*2+1
        })
        
        # M3 = A00 * (B0k1 - B1k1)
        # alpha: A00
        # beta: B0k1, -B1k1
        # gamma: C0k1 (+1)
        components.append({
            'U': [(0, 1)],
            'V': [(k0 + 1, 1), (13 + k0 + 1, -1)], # 0*13+k0+1, 1*13+k0+1
            'W': [((k0 + 1) * 2 + 0, 1)] # (k0+1)*2+0
        })
        
        # M4 = A11 * (B1k0 - B0k0)
        # alpha: A11
        # beta: B1k0, -B0k0
        # gamma: C0k0 (+1), C1k0 (+1)
        components.append({
            'U': [(3, 1)],
            'V': [(13 + k0, 1), (k0, -1)], # 1*13+k0, 0*13+k0
            'W': [(k0 * 2 + 0, 1), (k0 * 2 + 1, 1)] # k0*2+0, k0*2+1
        })
        
        # M5 = (A00 + A01) * B1k1
        # alpha: A00, A01
        # beta: B1k1
        # gamma: C0k0 (-1), C0k1 (+1)
        components.append({
            'U': [(0, 1), (1, 1)], # 0*2+0, 0*2+1
            'V': [(13 + k0 + 1, 1)], # 1*13+k0+1
            'W': [(k0 * 2 + 0, -1), ((k0 + 1) * 2 + 0, 1)] # k0*2+0, (k0+1)*2+0
        })
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        # alpha: A10, -A00
        # beta: B0k0, B0k1
        # gamma: C1k1 (+1)
        components.append({
            'U': [(2, 1), (0, -1)], # 1*2+0, 0*2+0
            'V': [(k0, 1), (k0 + 1, 1)], # 0*13+k0, 0*13+k0+1
            'W': [((k0 + 1) * 2 + 1, 1)] # (k0+1)*2+1
        })
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        # alpha: A01, -A11
        # beta: B1k0, B1k1
        # gamma: C0k0 (+1)
        components.append({
            'U': [(1, 1), (3, -1)], # 0*2+1, 1*2+1
            'V': [(13 + k0, 1), (13 + k0 + 1, 1)], # 1*13+k0, 1*13+k0+1
            'W': [(k0 * 2 + 0, 1)] # k0*2+0
        })

    # Add 6 Strassen blocks for columns 0-11
    for b in range(6):
        add_strassen(2 * b)
        
    # Last column k=12 (single column, schoolbook)
    k = 12
    # C0k = A00*B0k + A01*B1k
    # Term 1: A00 * B0k
    components.append({
        'U': [(0, 1)],
        'V': [(k, 1)],
        'W': [(k * 2 + 0, 1)]
    })
    # Term 2: A01 * B1k
    components.append({
        'U': [(1, 1)],
        'V': [(13 + k, 1)],
        'W': [(k * 2 + 0, 1)]
    })
    # C1k = A10*B0k + A11*B1k
    # Term 3: A10 * B0k
    components.append({
        'U': [(2, 1)],
        'V': [(k, 1)],
        'W': [(k * 2 + 1, 1)]
    })
    # Term 4: A11 * B1k
    components.append({
        'U': [(3, 1)],
        'V': [(13 + k, 1)],
        'W': [(k * 2 + 1, 1)]
    })
    
    rank = len(components)
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, comp in enumerate(components):
        for idx, val in comp['U']:
            U[idx][r] += val
        for idx, val in comp['V']:
            V[idx][r] += val
        for idx, val in comp['W']:
            W[idx][r] += val
            
    return {"U": U, "V": V, "W": W}
