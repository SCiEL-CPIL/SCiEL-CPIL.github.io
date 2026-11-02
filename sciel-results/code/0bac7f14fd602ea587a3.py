"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
by a 3 by 5 matrix.

Strategy:
Decompose the multiplication into blocks:
A (2x3) = [A1 (2x2) | A2 (2x1)]
B (3x5) = [B11 (2x2) B12 (2x2) B13 (2x1); B2 (1x5)]

C = A1 B11 + A1 B12 + A1 B13 + A2 B2

Ranks:
1. A1 B11 (2x2 * 2x2): Strassen rank 7.
2. A1 B12 (2x2 * 2x2): Strassen rank 7.
3. A1 B13 (2x2 * 2x1): Schoolbook rank 4.
4. A2 B2 (2x1 * 1x5): Schoolbook rank 10.

Total Rank = 7 + 7 + 4 + 10 = 28.
"""

def construct():
    n, m, p = 2, 3, 5
    
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen 7 products for 2x2x2
        # A indices: (i0+di, j0+dj)
        # B indices: (j0+dj, k0+dk)
        # C indices: (i0+di, k0+dk)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We define the coefficients for U, V, W for each of the 7 products.
        # U entries: A[i,j]
        # V entries: B[j,k]
        # W entries: C[i,k]
        
        # Local indices 0,1 for i,j,k offsets.
        # Global i = i0 + di, j = j0 + dj, k = k0 + dk.
        
        products = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: {(di,dj): val}, V: {(dj,dk): val}, W: {(di,dk): val}
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1, (0,1):1, (1,0):1}), # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):1}), # M2
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (0,1):1}), # M3
            ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}), # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,0):1}), # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):-1}), # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,0):-1, (1,1):1}), # M7
        ]
        
        for u_c, v_c, w_c in products:
            for (di, dj), val_u in u_c.items():
                U[(i0+di)*m + (j0+dj)][col] += val_u
            for (dj, dk), val_v in v_c.items():
                V[(j0+dj)*p + (k0+dk)][col] += val_v
            for (di, dk), val_w in w_c.items():
                W[(k0+dk)*n + (i0+di)][col] += val_w
            col += 1

    # Block 1: A1 B11 (i in 0,1; j in 0,1; k in 0,1)
    add_strassen(0, 0, 0)
    
    # Block 2: A1 B12 (i in 0,1; j in 0,1; k in 2,3)
    add_strassen(0, 0, 2)
    
    # Block 3: A1 B13 (i in 0,1; j in 0,1; k in 4)
    # Schoolbook: 4 terms
    # C[i,4] = A[i,0]B[0,4] + A[i,1]B[1,4]
    for i in range(2):
        for j in range(2):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    # Block 4: A2 B2 (i in 0,1; j in 2; k in 0,1,2,3,4)
    # Schoolbook: 10 terms
    # C[i,k] = A[i,2]B[2,k]
    for i in range(2):
        for k in range(5):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    assert col == rank
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    import numpy as np
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 2, 3, 5
    # Check rank-1 sums
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    print(f"Error at {i},{j},{k}: {s}")
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        print("Verification passed.")
