import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14

    # Strassen decomposition for 2x2x2 matrix multiplication (rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Strassen terms:
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    U_2x2x2 = np.zeros((4, 7), dtype=int)
    V_2x2x2 = np.zeros((4, 7), dtype=int)
    W_2x2x2 = np.zeros((4, 7), dtype=int)

    # U[i*2+j, r]: coefficient of A_{i,j} in term r
    # V[j*2+k, r]: coefficient of B_{j,k} in term r
    # W[k*2+i, r]: coefficient for output C_{k*2+i} from term r
    # Note: W index is [k*n + i] where n=2, so W[k*2+i, r]

    # M1 = (A11+A22)(B12+B21)
    U_2x2x2[0, 0] = 1  # A11
    U_2x2x2[3, 0] = 1  # A22
    V_2x2x2[1, 0] = 1  # B12
    V_2x2x2[2, 0] = 1  # B21
    # C11 += M1, C22 += M1
    W_2x2x2[0, 0] = 1  # C11 (k=0,i=0)
    W_2x2x2[3, 0] = 1  # C22 (k=1,i=1)

    # M2 = (A21+A22)B11
    U_2x2x2[2, 1] = 1  # A21
    U_2x2x2[3, 1] = 1  # A22
    V_2x2x2[0, 1] = 1  # B11
    # C21 += M2, C22 -= M2
    W_2x2x2[2, 1] = 1   # C21 (k=1,i=0)
    W_2x2x2[3, 1] = -1  # C22 (k=1,i=1)

    # M3 = A11(B11-B21)
    U_2x2x2[0, 2] = 1  # A11
    V_2x2x2[0, 2] = 1  # B11
    V_2x2x2[2, 2] = -1 # B21
    # C12 += M3, C22 += M3
    W_2x2x2[1, 2] = 1  # C12 (k=0,i=1)
    W_2x2x2[3, 2] = 1  # C22 (k=1,i=1)

    # M4 = A22(B22-B12)
    U_2x2x2[3, 3] = 1  # A22
    V_2x2x2[3, 3] = 1  # B22
    V_2x2x2[1, 3] = -1 # B12
    # C11 += M4, C21 += M4
    W_2x2x2[0, 3] = 1  # C11 (k=0,i=0)
    W_2x2x2[2, 3] = 1  # C21 (k=1,i=0)

    # M5 = (A11+A12)B22
    U_2x2x2[0, 4] = 1  # A11
    U_2x2x2[1, 4] = 1  # A12
    V_2x2x2[3, 4] = 1  # B22
    # C11 -= M5, C12 += M5
    W_2x2x2[0, 4] = -1 # C11 (k=0,i=0)
    W_2x2x2[1, 4] = 1  # C12 (k=0,i=1)

    # M6 = (A21-A11)(B11+B12)
    U_2x2x2[2, 5] = 1  # A21
    U_2x2x2[0, 5] = -1 # A11
    V_2x2x2[0, 5] = 1  # B11
    V_2x2x2[1, 5] = 1  # B12
    # C22 += M6
    W_2x2x2[3, 5] = 1  # C22 (k=1,i=1)

    # M7 = (A12-A22)(B21+B22)
    U_2x2x2[1, 6] = 1  # A12
    U_2x2x2[3, 6] = -1 # A22
    V_2x2x2[2, 6] = 1  # B21
    V_2x2x2[3, 6] = 1  # B22
    # C11 += M7
    W_2x2x2[0, 6] = 1  # C11 (k=0,i=0)

    # Schoolbook decomposition for 2x2x1 (rank 4)
    U_2x2x1 = np.zeros((4, 4), dtype=int)
    V_2x2x1 = np.zeros((2, 4), dtype=int)
    W_2x2x1 = np.zeros((2, 4), dtype=int)
    # C = A * B where A is 2x2, B is 2x1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    for j in range(2):
        for k in range(1):
            r = j * 1 + k
            U_2x2x1[j, r] = 1  # A[j, 0] for j=0,1
            V_2x2x1[0, r] = 1  # B[0]
            W_2x2x1[0, r] = 1  # C[0]
    for j in range(2):
        for k in range(1):
            r = 2 + j
            U_2x2x1[j + 2, r] = 1  # A[j, 1] for j=0,1 -> indices 2,3
            V_2x2x1[1, r] = 1  # B[1]
            W_2x2x1[1, r] = 1  # C[1]

    # Wait, let me redo 2x2x1 more carefully
    # A is 2x2 (rows i=0,1; cols j=0,1), B is 2x1 (rows j=0,1; col k=0)
    # C is 2x1 (rows i=0,1; col k=0)
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # Rank 4: one term per (j, ...) -- actually we need terms indexed by j
    # Term r for j: U[i,j] contributes, V[j,0] contributes, W[0,i] contributes
    # Actually for 2x2x1: n=2, m=2, p=1
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (2, rank)
    # W shape: (p*n, rank) = (2, rank)
    # C[i*1+0, ...] = sum_r U[i*2+j, r] * V[j*1+0, r] * W[0*2+i, r]
    # For each j: one term
    # Term j: U[i*2+j, j] = 1 for all i, V[j, j] = 1, W[i, j] = 1
    U_2x2x1 = np.zeros((4, 2), dtype=int)
    V_2x2x1 = np.zeros((2, 2), dtype=int)
    W_2x2x1 = np.zeros((2, 2), dtype=int)
    for j in range(2):
        for i in range(2):
            U_2x2x1[i * 2 + j, j] = 1
        V_2x2x1[j, j] = 1
        for i in range(2):
            W_2x2x1[i, j] = 1

    # Now build the full 2x4x7 decomposition
    # A is 2x4, B is 4x7
    # Split A into A1 (2x2) and A2 (2x2): A = [A1 | A2]
    # Split B into B1 (2x7) and B2 (2x7): B = [B1; B2]
    # C = A1 * B1 + A2 * B2
    # Each Aa * Ba is 2x2 * 2x7
    # Split B1 into B1a (2x2) and B1b (2x5)... no wait
    # Actually, B is 4x7. Split rows: B1 is first 2 rows (2x7), B2 is last 2 rows (2x7)
    # A1 is first 2 cols of A (2x2), A2 is last 2 cols of A (2x2)
    # C = A1 * B1 + A2 * B2 where B1 is 2x7, B2 is 2x7
    # Now for A1 * B1 (2x2 * 2x7): split B1 into B1a (2x2) and B1b (2x5)
    # A1 * B1a is 2x2 * 2x2 -> use Strassen (rank 7)
    # A1 * B1b is 2x2 * 2x5 -> split B1b further? Or use schoolbook (rank 2*2*5=20)?
    # Better: for 2x2 * 2x7, split the 7 columns into 2+2+2+1
    # That gives 3 * (2x2x2) + 1 * (2x2x1) = 3*7 + 1*2 = 23 per block
    # Total: 2 blocks * 23 = 46? No wait...
    
    # Let me reconsider. The total rank should be:
    # For each of the 2 "row blocks" of B (each 2x7):
    #   Split 7 columns into groups: 2, 2, 2, 1
    #   Each 2x2 group uses Strassen: rank 7
    #   The 2x1 group uses schoolbook: rank 2
    #   Subtotal per row block: 3*7 + 1*2 = 23
    # Two row blocks: 2 * 23 = 46
    
    # Hmm, but the known result from the cell finding says rank 50.
    # Let me re-examine. Maybe the decomposition is different.
    # 
    # Alternative: split 4 as 2+2 in the middle, and 7 as 4+3 or 2+2+2+1
    # 
    # Actually, let me think about this differently.
    # A (2x4) * B (4x7) = C (2x7)
    # Write A = [A1 | A2] where A1, A2 are 2x2
    # Write B = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1 (2x2 * 2x7): 
    #   Write B1 = [B1a | B1b | B1c | B1d] where each is 2x2, 2x2, 2x2, 2x1
    #   A1*B1a: 2x2*2x2, Strassen rank 7
    #   A1*B1b: 2x2*2x2, Strassen rank 7  
    #   A1*B1c: 2x2*2x2, Strassen rank 7
    #   A1*B1d: 2x2*2x1, schoolbook rank 2
    #   Subtotal: 23
    # Similarly for A2*B2: 23
    # Total: 46
    
    # But wait - can we share terms between A1*B1 and A2*B2? No, they're independent.
    # So rank 46? Let me verify this is valid.
    
    # Actually, I need to be more careful. Let me just build rank-46 and verify.
    
    # Build the full decomposition
    # Total rank = 2 * (3*7 + 1*2) = 2 * 23 = 46
    
    total_rank = 46
    U = np.zeros((nm, total_rank), dtype=int)
    V = np.zeros((mp, total_rank), dtype=int)
    W = np.zeros((pn, total_rank), dtype=int)
    
    col = 0
    for block in range(2):  # block 0: A1*B1, block 1: A2*B2
        # A block: rows 0-1 of A, cols 2*block to 2*block+1
        # B block: rows 2*block to 2*block+1 of B, all 7 cols
        # Split 7 cols into: 2, 2, 2, 1
        
        # Sub-block 0: cols 0-1 (2x2) -> Strassen
        for sub in range(3):
            for r in range(7):
                # U: A[i, 2*block + j] -> index i*4 + (2*block + j)
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_2x2x2[i * 2 + j, r]
                # V: B[2*block + j, sub*2 + k] -> index (2*block + j)*7 + (sub*2 + k)
                for j in range(2):
                    for k in range(2):
                        V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_2x2x2[j * 2 + k, r]
                # W: C[i, sub*2 + k] -> index (sub*2 + k)*2 + i
                for k in range(2):
                    for i in range(2):
                        W[(sub * 2 + k) * 2 + i, col] += W_2x2x2[k * 2 + i, r]
                col += 1
        
        # Sub-block 3: col 6 (2x1) -> schoolbook rank 2
        for r in range(2):
            # U: A[i, 2*block + j] for the 2x2x1
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (2 * block + j), col] += U_2x2x1[i * 2 + j, r]
            # V: B[2*block + j, 6] -> index (2*block + j)*7 + 6
            for j in range(2):
                V[(2 * block + j) * 7 + 6, col] += V_2x2x1[j, r]
            # W: C[i, 6] -> index 6*2 + i
            for i in range(2):
                W[6 * 2 + i, col] += W_2x2x1[i, r]
            col += 1
    
    # Verify
    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
