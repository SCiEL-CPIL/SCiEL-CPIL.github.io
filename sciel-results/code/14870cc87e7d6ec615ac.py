import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target rank for hybrid decomposition
    # Row 0: Schoolbook 1x4x8 = 32
    # Rows 1-2: Strassen on 2x4x8 = 56
    # Total = 88
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook
    for j in range(m):
        for k in range(p):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
            
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) - Strassen
    # We need to add 8 blocks of 2x2x2 Strassen
    # Block indices:
    # i0 is 1 (rows 1,2 of A)
    # jA0 is 0 or 2 (cols of A)
    # jB0 is 0 or 2 (rows of B)
    # k0 is 0, 2, 4, 6 (cols of B)
    
    def add_strassen(U, V, W, col, i0, jA0, jB0, k0, n, m, p):
        # M1 = (A00+A11)(B00+B11)
        U[i0*m + jA0, col] += 1
        U[(i0+1)*m + (jA0+1), col] += 1
        V[jB0*p + k0, col] += 1
        V[(jB0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + jA0, col] += 1
        U[(i0+1)*m + (jA0+1), col] += 1
        V[jB0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i0*m + jA0, col] += 1
        V[jB0*p + (k0+1), col] += 1
        V[(jB0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i0+1)*m + (jA0+1), col] += 1
        V[(jB0+1)*p + k0, col] += 1
        V[jB0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + jA0, col] += 1
        U[i0*m + (jA0+1), col] += 1
        V[(jB0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + jA0, col] += 1
        U[i0*m + jA0, col] -= 1
        V[jB0*p + k0, col] += 1
        V[jB0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m + (jA0+1), col] += 1
        U[(i0+1)*m + (jA0+1), col] -= 1
        V[(jB0+1)*p + k0, col] += 1
        V[(jB0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1
        
        return col

    # Iterate over the 4 blocks in j and 2 blocks in k? 
    # jA0 in {0, 2}
    # jB0 in {0, 2}
    # k0 in {0, 2, 4, 6}
    # This covers 2 * 2 * 4 = 16 blocks?
    # No.
    # A is 2x4. Rows 1,2. Cols 0,1,2,3.
    # B is 4x8. Rows 0,1,2,3. Cols 0..7.
    # We split A cols into {0,1} and {2,3}.
    # We split B rows into {0,1} and {2,3}.
    # We split B cols into {0,1}, {2,3}, {4,5}, {6,7}.
    # So we have 2 (A col pairs) * 2 (B row pairs) * 4 (B col pairs) = 16 blocks?
    # Wait.
    # A_1 (cols 0,1) multiplies B_top (rows 0,1). This gives 4 blocks of 2x2x2.
    # A_1 (cols 0,1) multiplies B_bot (rows 2,3). This gives 4 blocks of 2x2x2.
    # A_2 (cols 2,3) multiplies B_top (rows 0,1). This gives 4 blocks of 2x2x2.
    # A_2 (cols 2,3) multiplies B_bot (rows 2,3). This gives 4 blocks of 2x2x2.
    # Total 16 blocks of 2x2x2.
    # Rank 16 * 7 = 112.
    # This is NOT 56.
    
    # My previous calculation of 56 was wrong.
    # Let's re-evaluate R(2,4,8).
    # R(2,4,8) = R(2,2,8) + R(2,2,8).
    # R(2,2,8) = 4 * R(2,2,2) = 28.
    # So R(2,4,8) = 56.
    # Why the discrepancy in block count?
    # A_1 (2x2) * B_top (2x8).
    # B_top is 2x8. It consists of 4 blocks of 2x2.
    # So A_1 * B_top involves 4 Strassen blocks.
    # A_1 * B_bot involves 4 Strassen blocks.
    # A_2 * B_top involves 4 Strassen blocks.
    # A_2 * B_bot involves 4 Strassen blocks.
    # Total 16 blocks.
    # 16 * 7 = 112.
    
    # So where did 56 come from?
    # Ah. R(2,2,8) is the rank of multiplying 2x2 by 2x8.
    # This is 4 blocks of 2x2x2. Rank 28.
    # R(2,4,8) splits j into 2+2.
    # Term 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x8). Rank 28.
    # Term 2: A[:, 2:4] * B[2:4, :] (2x2 * 2x8). Rank 28.
    # Total 56.
    
    # So we only need 8 Strassen blocks, not 16.
    # The blocks are:
    # 1. A[1:3, 0:2] * B[0:2, 0:2]
    # 2. A[1:3, 0:2] * B[0:2, 2:4]
    # 3. A[1:3, 0:2] * B[0:2, 4:6]
    # 4. A[1:3, 0:2] * B[0:2, 6:8]
    # 5. A[1:3, 2:4] * B[2:4, 0:2]
    # 6. A[1:3, 2:4] * B[2:4, 2:4]
    # 7. A[1:3, 2:4] * B[2:4, 4:6]
    # 8. A[1:3, 2:4] * B[2:4, 6:8]
    
    # So the loop should be:
    # for jA0 in [0, 2]:
    #   for jB0 in [0, 2]:
    #     for k0 in [0, 2, 4, 6]:
    #       add_strassen(...)
    
    # This is 2 * 2 * 4 = 16 iterations?
    # No.
    # If jA0=0, jB0 MUST be 0?
    # A[:, 0:2] multiplies B[0:2, :]. So jB0=0.
    # A[:, 2:4] multiplies B[2:4, :]. So jB0=2.
    # So jB0 is determined by jA0.
    # jA0=0 -> jB0=0.
    # jA0=2 -> jB0=2.
    # So we have 2 * 4 = 8 blocks.
    
    # Correct.
    
    for jA0 in [0, 2]:
        jB0 = jA0  # Since B rows correspond to A cols
        for k0 in [0, 2, 4, 6]:
            col = add_strassen(U, V, W, col, i0=1, jA0=jA0, jB0=jB0, k0=k0, n=n, m=m, p=p)
            
    assert col == 88, f"Expected 88, got {col}"
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
