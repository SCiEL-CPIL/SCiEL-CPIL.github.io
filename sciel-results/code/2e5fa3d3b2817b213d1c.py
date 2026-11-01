import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    # We use 8 blocks of 2x2 multiplication
    num_blocks = p // 2
    rank = num_blocks * 7
    R = rank
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Strassen coefficients for A (indices 0:00, 1:01, 2:10, 3:11)
    # Strassen coefficients for B (indices 0:00, 1:01, 2:10, 3:11)
    # Strassen coefficients for C (indices 0:00, 1:01, 2:10, 3:11)
    
    # P1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # P2: (A10+A11)B00      -> C10+=1, C11-=1
    # P3: A00(B01-B11)      -> C01+=1, C11+=1
    # P4: A11(B10-B00)      -> C00+=1, C10+=1
    # P5: (A00+A01)B11      -> C00-=1, C01+=1
    # P6: (A10-A00)(B00+B01)-> C11+=1
    # P7: (A01-A11)(B10+B11)-> C00+=1
    
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_coeffs = [
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(num_blocks):
        # Block b uses columns 2*b and 2*b+1 of B and C
        # B indices:
        # B[0, 2*b] -> 0 * p + 2*b = 2*b
        # B[0, 2*b+1] -> 2*b+1
        # B[1, 2*b] -> 1 * p + 2*b = p + 2*b
        # B[1, 2*b+1] -> p + 2*b + 1
        
        # C indices:
        # C[0, 2*b] -> 2*b
        # C[0, 2*b+1] -> 2*b+1
        # C[1, 2*b] -> p + 2*b
        # C[1, 2*b+1] -> p + 2*b + 1
        
        b_idx = [2*b, 2*b+1, p + 2*b, p + 2*b + 1]
        c_idx = [2*b, 2*b+1, p + 2*b, p + 2*b + 1]
        
        for s in range(7):
            col = b * 7 + s
            
            # U
            for i in range(4):
                U[i, col] = U_coeffs[s][i]
                
            # V
            for i in range(4):
                V[b_idx[i], col] = V_coeffs[s][i]
                
            # W
            for i in range(4):
                W[c_idx[i], col] = W_coeffs[s][i]
                
    return {"U": U, "V": V, "W": W}
