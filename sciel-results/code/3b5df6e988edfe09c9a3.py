import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strategy: Use non-overlapping 2x2x2 Strassen blocks to reduce rank.
    # Each 2x2x2 block: 8 schoolbook terms -> 7 Strassen terms (save 1)
    # We can place up to 4 such blocks in the 3x5x5 tensor
    # Blocks: (i0,i1,j0,j1,k0,k1) with i0=0,i1=1
    # j in {0,1}x{0,1} or {2,3}x{0,1} or {0,1}x{2,3} or {2,3}x{2,3}
    # Total: 4 blocks * 7 + (75 - 4*8) = 28 + 43 = 71
    
    total_rank = 4 * 7 + (n * m * p - 4 * 8)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Define the 4 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 2, 3),
    ]
    
    # Strassen formulas for a 2x2 block:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # M1 = (a00 + a11)(b00 + b11)
        c = col
        U[i0*m + j0, c] += 1
        U[i1*m + j1, c] += 1
        V[j0*p + k0, c] += 1
        V[j1*p + k1, c] += 1
        W[k0*n + i0, c] += 1
        W[k1*n + i1, c] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        c = col
        U[i1*m + j0, c] += 1
        U[i1*m + j1, c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + i1, c] += 1
        W[k1*n + i1, c] += -1
        col += 1
        
        # M3 = a00(b01 - b11)
        c = col
        U[i0*m + j0, c] += 1
        V[j0*p + k1, c] += 1
        V[j1*p + k1, c] += -1
        W[k1*n + i0, c] += 1
        W[k1*n + i1, c] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        c = col
        U[i1*m + j1, c] += 1
        V[j1*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + i1, c] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        c = col
        U[i0*m + j0, c] += 1
        U[i0*m + j1, c] += 1
        V[j1*p + k1, c] += 1
        W[k0*n + i0, c] += -1
        W[k1*n + i0, c] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        c = col
        U[i1*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + k1, c] += 1
        W[k1*n + i1, c] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        c = col
        U[i0*m + j1, c] += 1
        U[i1*m + j1, c] += -1
        V[j1*p + k0, c] += 1
        V[j1*p + k1, c] += 1
        W[k0*n + i0, c] += 1
        col += 1
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (i0, i1, j0, j1, k0, k1) in blocks:
                    if i0 <= i <= i1 and j0 <= j <= j1 and k0 <= k <= k1:
                        in_block = True
                        break
                if not in_block:
                    U[i*m + j, col] += 1
                    V[j*p + k, col] += 1
                    W[k*n + i, col] += 1
                    col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
    
    T_expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max error: {max_err}")
    
    if max_err < 1e-10:
        print("Decomposition is correct!")
    else:
        print("Decomposition has errors!")
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_reconstructed[i,j,k] - 1.0) > 1e-10:
                        print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]} (expected 1.0)")
