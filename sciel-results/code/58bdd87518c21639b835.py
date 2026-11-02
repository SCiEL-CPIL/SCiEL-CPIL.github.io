import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indices for A (U):
    # a=0, b=1, c=2, d=3
    
    # Indices for B (V):
    # Block 1 (cols 0,1): e1=0, f1=1, g1=4, h1=5
    # Block 2 (cols 2,3): e2=2, f2=3, g2=6, h2=7
    
    # Indices for C (W):
    # Block 1 (cols 0,1): C00=0, C10=1, C01=2, C11=3
    # Block 2 (cols 2,3): C02=4, C12=5, C03=6, C13=7
    
    # Strassen products definitions
    # Each product is a tuple: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {idx: val}
    # V_coeffs: dict {idx: val}
    # W_coeffs: dict {idx: val}
    
    products = []
    
    # Block 1
    # P1
    products.append({
        'U': {0: 1, 3: 1},
        'V': {0: 1, 5: 1},
        'W': {0: 1, 3: 1}
    })
    # P2
    products.append({
        'U': {2: 1, 3: 1},
        'V': {0: 1},
        'W': {1: 1, 3: -1}
    })
    # P3
    products.append({
        'U': {0: 1},
        'V': {1: 1, 5: -1},
        'W': {2: 1, 3: 1}
    })
    # P4
    products.append({
        'U': {3: 1},
        'V': {4: 1, 0: -1},
        'W': {0: 1, 1: 1}
    })
    # P5
    products.append({
        'U': {0: 1, 1: 1},
        'V': {5: 1},
        'W': {0: -1, 2: 1}
    })
    # P6
    products.append({
        'U': {2: 1, 0: -1},
        'V': {0: 1, 1: 1},
        'W': {3: 1}
    })
    # P7
    products.append({
        'U': {1: 1, 3: -1},
        'V': {4: 1, 5: 1},
        'W': {0: 1}
    })
    
    # Block 2
    # P1
    products.append({
        'U': {0: 1, 3: 1},
        'V': {2: 1, 7: 1},
        'W': {4: 1, 7: 1}
    })
    # P2
    products.append({
        'U': {2: 1, 3: 1},
        'V': {2: 1},
        'W': {5: 1, 7: -1}
    })
    # P3
    products.append({
        'U': {0: 1},
        'V': {3: 1, 7: -1},
        'W': {6: 1, 7: 1}
    })
    # P4
    products.append({
        'U': {3: 1},
        'V': {6: 1, 2: -1},
        'W': {4: 1, 5: 1}
    })
    # P5
    products.append({
        'U': {0: 1, 1: 1},
        'V': {7: 1},
        'W': {4: -1, 6: 1}
    })
    # P6
    products.append({
        'U': {2: 1, 0: -1},
        'V': {2: 1, 3: 1},
        'W': {7: 1}
    })
    # P7
    products.append({
        'U': {1: 1, 3: -1},
        'V': {6: 1, 7: 1},
        'W': {4: 1}
    })
    
    for r, prod in enumerate(products):
        for idx, val in prod['U'].items():
            U[idx][r] = val
        for idx, val in prod['V'].items():
            V[idx][r] = val
        for idx, val in prod['W'].items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
