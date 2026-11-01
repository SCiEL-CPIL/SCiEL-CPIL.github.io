import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 16. Strassen-like rank is 14.
    rank = 14
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2 multiplication
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
    ]
    
    V_S = [
        [1, 1, 0, 0, 0, 1, 0], # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0], # j=0, k=1 (B01)
        [0, 0, 0, -1, 0, 0, 1],# j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1] # j=1, k=1 (B11)
    ]
    
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1 (C01)
        [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0 (C10)
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1 (C11)
    ]
    
    # Fill U
    # Columns 0-6 and 7-13 are identical (U_S)
    for r in range(7):
        for i in range(4):
            U[i][r] = U_S[i][r]
            U[i][r + 7] = U_S[i][r]
            
    # Fill V
    # Block 1 (cols 0-6) uses rows 0-3 of V_S
    # Block 2 (cols 7-13) uses rows 4-7 of V_S (which are V_S shifted)
    for r in range(7):
        for j in range(2):
            for k in range(2):
                # Block 1 index: j*2+k
                V[j*2+k][r] = V_S[j*2+k][r]
                # Block 2 index: j*4 + (k+2) = j*2*2 + k + 2
                # Wait, V index is j*p+k. p=4.
                # Block 1 k in {0,1}. Block 2 k in {2,3}.
                # V_S index is j*2+k_local.
                # V global index for block 1: j*4 + k_local.
                # V global index for block 2: j*4 + (k_local + 2).
                
                # Let's re-verify V_S indices.
                # V_S is 4x7. Indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1).
                # V_S[idx] = V_S[j*2+k_local].
                
                # For block 1 (r < 7):
                # V[j*4 + k_local][r] = V_S[j*2+k_local][r]
                
                # For block 2 (r >= 7):
                # V[j*4 + (k_local+2)][r] = V_S[j*2+k_local][r]
                
    # Let's rewrite the V filling loop clearly
    for r in range(7):
        for j in range(2):
            for k_loc in range(2):
                val = V_S[j*2+k_loc][r]
                # Block 1
                V[j*4 + k_loc][r] = val
                # Block 2
                V[j*4 + k_loc + 2][r + 7] = val
                
    # Fill W
    # Block 1 (cols 0-6) uses rows 0-3 of W_S
    # Block 2 (cols 7-13) uses rows 4-7 of W_S
    for r in range(7):
        for k_loc in range(2):
            for i in range(2):
                val = W_S[k_loc*2+i][r]
                # Block 1
                W[k_loc*2 + i][r] = val
                # Block 2
                W[(k_loc+2)*2 + i][r + 7] = val
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
