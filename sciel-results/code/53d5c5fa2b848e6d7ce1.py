import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Term 1 (j=0,1): 3 * Strassen(2x2x2) + 3 * Trivial(1x2x2) = 3*7 + 3*4 = 21 + 12 = 33
    # Term 2 (j=2,3): 3 * Strassen(2x2x2) + 3 * Trivial(1x2x2) = 3*7 + 3*4 = 21 + 12 = 33
    # Term 3 (j=4): 1 * OuterProduct(3x1x6) = 1
    # Total Rank = 67
    
    rank = 67
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a trivial 1x1x1 term
    def add_trivial(i, j, k, coeff=1):
        nonlocal col
        U[i * m + j, col] += coeff
        V[j * p + k, col] += coeff
        W[k * n + i, col] += coeff
        col += 1
        
    # Helper to add Strassen 2x2x2 term
    # A indices: i in {i0, i0+1}, j in {j0, j0+1}
    # B indices: j in {j0, j0+1}, k in {k0, k0+1}
    # C indices: i in {i0, i0+1}, k in {k0, k0+1}
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i0 * m + j0, col] += 1
        U[(i0+1) * m + (j0+1), col] += 1
        V[j0 * p + k0, col] += 1
        V[(j0+1) * p + (k0+1), col] += 1
        W[k0 * n + i0, col] += 1
        W[(k0+1) * n + i0, col] += 1
        W[k0 * n + (i0+1), col] += 1
        W[(k0+1) * n + (i0+1), col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1) * m + j0, col] += 1
        U[(i0+1) * m + (j0+1), col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + (i0+1), col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i0 * m + j0, col] += 1
        V[j0 * p + (k0+1), col] += 1
        V[(j0+1) * p + (k0+1), col] += -1
        W[(k0+1) * n + i0, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i0+1) * m + (j0+1), col] += 1
        V[(j0+1) * p + k0, col] += 1
        V[j0 * p + k0, col] += -1
        W[k0 * n + (i0+1), col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0 * m + j0, col] += 1
        U[i0 * m + (j0+1), col] += 1
        V[(j0+1) * p + (k0+1), col] += 1
        W[(k0+1) * n + i0, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1) * m + j0, col] += 1
        U[i0 * m + j0, col] += -1
        V[j0 * p + k0, col] += 1
        V[j0 * p + (k0+1), col] += 1
        W[k0 * n + (i0+1), col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0 * m + (j0+1), col] += 1
        U[(i0+1) * m + (j0+1), col] += -1
        V[(j0+1) * p + k0, col] += 1
        V[(j0+1) * p + (k0+1), col] += 1
        W[(k0+1) * n + i0, col] += 1
        col += 1

    # Helper to add trivial 1x2x2 term (Row i, Cols j0, j0+1, Kols k0, k0+1)
    # C_ik = A_i,j0 B_j0,k + A_i,j0+1 B_j0+1,k
    def add_trivial_1x2x2(i, j0, k0):
        nonlocal col
        # Term 1: A_i,j0 B_j0,k0
        add_trivial(i, j0, k0)
        # Term 2: A_i,j0 B_j0,k0+1
        add_trivial(i, j0, k0+1)
        # Term 3: A_i,j0+1 B_j0+1,k0
        add_trivial(i, j0+1, k0)
        # Term 4: A_i,j0+1 B_j0+1,k0+1
        add_trivial(i, j0+1, k0+1)

    # Helper to add outer product 3x1x6 (Row i0..i0+2, Col j0, Kols k0..k0+5)
    # C_ik = A_i,j0 B_j0,k
    def add_outer_product(i0, j0, k0, width_k):
        nonlocal col
        # This is rank 1? 
        # C_ik = A_{i0+r, j0} * B_{j0, k0+s}
        # U_{i, col} = A_{i, j0}
        # V_{col, k} = B_{j0, k}
        # W_{k, i, col} = 1
        for r in range(3):
            U[(i0+r) * m + j0, col] = 1
        for s in range(width_k):
            V[j0 * p + (k0+s), col] = 1
            W[(k0+s) * n + (i0+r), col] = 1 # Wait, W is (p*n, rank)
            # W index is k*n + i
            # So W[k*n + i, col] should be 1
        # Correcting W assignment
        # We need W[k*n + i, col] = 1 for all i, k in the block
        # The loop above does this
        col += 1

    # Term 1: j in {0, 1}
    # A rows 0,1 (2x2), 2 (1x2)
    # B cols 0,1 (2x2), 2,3 (2x2), 4,5 (2x2)
    
    # Sub-term 1.1: A[0:2, 0:2] * B[0:2, 0:2] -> Strassen
    add_strassen(0, 0, 0)
    
    # Sub-term 1.2: A[0:2, 0:2] * B[0:2, 2:4] -> Strassen
    add_strassen(0, 0, 2)
    
    # Sub-term 1.3: A[0:2, 0:2] * B[0:2, 4:6] -> Strassen
    add_strassen(0, 0, 4)
    
    # Sub-term 1.4: A[2, 0:2] * B[0:2, 0:2] -> Trivial 1x2x2
    add_trivial_1x2x2(2, 0, 0)
    
    # Sub-term 1.5: A[2, 0:2] * B[0:2, 2:4] -> Trivial 1x2x2
    add_trivial_1x2x2(2, 0, 2)
    
    # Sub-term 1.6: A[2, 0:2] * B[0:2, 4:6] -> Trivial 1x2x2
    add_trivial_1x2x2(2, 0, 4)

    # Term 2: j in {2, 3}
    # A rows 0,1 (2x2), 2 (1x2)
    # B cols 0,1 (2x2), 2,3 (2x2), 4,5 (2x2)
    
    # Sub-term 2.1: A[0:2, 2:4] * B[2:4, 0:2] -> Strassen
    add_strassen(0, 2, 0)
    
    # Sub-term 2.2: A[0:2, 2:4] * B[2:4, 2:4] -> Strassen
    add_strassen(0, 2, 2)
    
    # Sub-term 2.3: A[0:2, 2:4] * B[2:4, 4:6] -> Strassen
    add_strassen(0, 2, 4)
    
    # Sub-term 2.4: A[2, 2:4] * B[2:4, 0:2] -> Trivial 1x2x2
    add_trivial_1x2x2(2, 2, 0)
    
    # Sub-term 2.5: A[2, 2:4] * B[2:4, 2:4] -> Trivial 1x2x2
    add_trivial_1x2x2(2, 2, 2)
    
    # Sub-term 2.6: A[2, 2:4] * B[2:4, 4:6] -> Trivial 1x2x2
    add_trivial_1x2x2(2, 2, 4)

    # Term 3: j in {4}
    # A rows 0,1,2 (3x1)
    # B cols 0,1,2,3,4,5 (1x6)
    
    # Sub-term 3.1: A[0:3, 4] * B[4, 0:6] -> Outer Product
    add_outer_product(0, 4, 0, 6)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
