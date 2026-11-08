"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We use a hybrid approach:
    # 1. Use Strassen decomposition on 2x2x2 sub-blocks where possible.
    # 2. Use schoolbook for the remaining elements.
    # 3. The key is to cover the 3x5x5 tensor with minimal rank.
    
    # The tensor is T[i,j,k] for i in 0..2, j in 0..4, k in 0..4.
    # We can decompose this into blocks.
    
    # Strategy:
    # - The first 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}) can use Strassen (rank 7)
    # - The second 2x2x2 block (i in {0,1}, j in {2,3}, k in {0,1}) can use Strassen (rank 7)
    # - The third 2x2x2 block (i in {0,1}, j in {0,1}, k in {2,3}) can use Strassen (rank 7)
    # - The fourth 2x2x2 block (i in {0,1}, j in {2,3}, k in {2,3}) can use Strassen (rank 7)
    # That covers i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3} with rank 28.
    
    # Remaining elements:
    # - i=2, j in 0..4, k in 0..4: 5*5 = 25 elements (schoolbook)
    # - i in {0,1}, j=4, k in 0..4: 2*5 = 10 elements (schoolbook)
    # - i in {0,1}, j in {0,1,2,3}, k=4: 2*4 = 8 elements (schoolbook)
    
    # Total rank = 28 + 25 + 10 + 8 = 71
    
    # But we can do better by using Strassen on overlapping or different blocks.
    
    # Let's try a different approach: use the known optimal rank for 3x5x5.
    # The rank of matrix multiplication tensor M_{3,5,5} is at most:
    # - Schoolbook: 75
    # - With some Strassen blocks: we can save 4 per 2x2x2 block (save 1 per block)
    
    # Let's construct a rank-71 solution:
    # 4 Strassen blocks (rank 7 each) = 28
    # Remaining: 75 - 4*8 = 75 - 32 = 43 schoolbook terms
    # Total: 28 + 43 = 71
    
    # Define the Strassen blocks
    # Each block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    blocks = [
        (0, 0, 0),  # i in {0,1}, j in {0,1}, k in {0,1}
        (0, 2, 0),  # i in {0,1}, j in {2,3}, k in {0,1}
        (0, 0, 2),  # i in {0,1}, j in {0,1}, k in {2,3}
        (0, 2, 2),  # i in {0,1}, j in {2,3}, k in {2,3}
    ]
    
    # Strassen formulas for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Build the factor matrices
    rank = 71
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Add Strassen blocks
    for (i0, j0, k0) in blocks:
        # M1 = (A11+A22)(B11+B22)
        # A11 is at (i0, j0), A22 is at (i0+1, j0+1)
        # B11 is at (j0, k0), B22 is at (j0+1, k0+1)
        # C11 is at (k0, i0), C22 is at (k0+1, i0+1)
        
        # M1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1  # C11 += M1
        W[(k0+1)*n + (i0+1), col] += 1  # C22 += M1
        col += 1
        
        # M2 = (A21+A22)B11
        # A21 is at (i0+1, j0), A22 is at (i0+1, j0+1)
        # B11 is at (j0, k0)
        # C21 is at (k0, i0+1)
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1  # C21 += M2
        col += 1
        
        # M3 = A11(B21-B22)
        # A11 is at (i0, j0)
        # B21 is at (j0+1, k0), B22 is at (j0+1, k0+1)
        # C12 is at (k0+1, i0)
        U[i0*m + j0, col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += -1
        W[(k0+1)*n + i0, col] += 1  # C12 += M3
        col += 1
        
        # M4 = A22(B12-B11)
        # A22 is at (i0+1, j0+1)
        # B12 is at (j0, k0+1), B11 is at (j0, k0)
        # C11 is at (k0, i0), C21 is at (k0, i0+1)
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + (k0+1), col] += 1
        V[j0*p + k0, col] += -1
        W[k0*n + i0, col] += 1  # C11 += M4
        W[k0*n + (i0+1), col] += 1  # C21 += M4
        col += 1
        
        # M5 = (A11+A12)B22
        # A11 is at (i0, j0), A12 is at (i0, j0+1)
        # B22 is at (j0+1, k0+1)
        # C11 is at (k0, i0), C12 is at (k0+1, i0)
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += -1  # C11 -= M5
        W[(k0+1)*n + i0, col] += 1  # C12 += M5
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21 is at (i0+1, j0), A11 is at (i0, j0)
        # B11 is at (j0, k0), B12 is at (j0, k0+1)
        # C22 is at (k0+1, i0+1)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] += -1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1  # C22 += M6
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12 is at (i0, j0+1), A22 is at (i0+1, j0+1)
        # B21 is at (j0+1, k0), B22 is at (j0+1, k0+1)
        # C11 is at (k0, i0), C22 is at (k0+1, i0+1)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] += -1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1  # C11 += M7
        W[(k0+1)*n + (i0+1), col] += 1  # C22 += M7
        col += 1
    
    # Add schoolbook terms for remaining elements
    # Covered by Strassen: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Need to cover:
    # - i=2, j in 0..4, k in 0..4
    # - i in {0,1}, j=4, k in 0..4
    # - i in {0,1}, j in {0,1,2,3}, k=4
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this element is covered by Strassen
                covered = False
                for (i0, j0, k0) in blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                        covered = True
                        break
                if not covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
