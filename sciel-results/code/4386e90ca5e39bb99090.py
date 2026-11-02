import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 140.0  # Safety margin (evaluator kills at 150s)

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We aim for rank 20 based on the decomposition:
    # M(2,3,8) <= 2 * M(2,3,4) <= 2 * (2 * M(2,3,2)) <= 2 * 2 * (M(2,2,2) + M(2,1,2))
    # M(2,2,2) = 4 (schoolbook)
    # M(2,1,2) = 1
    # M(2,3,2) <= 5
    # M(2,3,4) <= 10
    # M(2,3,8) <= 20
    
    rank = 20
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term to the decomposition
    # u_coeffs: dict { (i, j): val }
    # v_coeffs: dict { (j, k): val }
    # w_coeffs: dict { (k, i): val }
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j, col] = val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k, col] = val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i, col] = val
        col += 1

    # Define the 4 schoolbook terms for 2x2x2
    # A is 2x2 (rows i0, i1; cols j0, j1)
    # B is 2x2 (rows j0, j1; cols k0, k1)
    # C is 2x2 (rows i0, i1; cols k0, k1)
    
    def add_2x2_block(i0, i1, j0, j1, k0, k1):
        # Term 1: A[i0, j0] * B[j0, k0] -> C[i0, k0]
        add_term({(i0, j0): 1}, {(j0, k0): 1}, {(k0, i0): 1})
        # Term 2: A[i0, j1] * B[j1, k0] -> C[i0, k0]
        add_term({(i0, j1): 1}, {(j1, k0): 1}, {(k0, i0): 1})
        # Term 3: A[i1, j0] * B[j0, k1] -> C[i1, k1]
        add_term({(i1, j0): 1}, {(j0, k1): 1}, {(k1, i1): 1})
        # Term 4: A[i1, j1] * B[j1, k1] -> C[i1, k1]
        add_term({(i1, j1): 1}, {(j1, k1): 1}, {(k1, i1): 1})
        
        # Wait, schoolbook is 4 terms:
        # C[i0, k0] = A[i0, j0]B[j0, k0] + A[i0, j1]B[j1, k0]
        # C[i0, k1] = A[i0, j0]B[j0, k1] + A[i0, j1]B[j1, k1]
        # C[i1, k0] = A[i1, j0]B[j0, k0] + A[i1, j1]B[j1, k0]
        # C[i1, k1] = A[i1, j0]B[j0, k1] + A[i1, j1]B[j1, k1]
        
        # My add_term above only adds 4 terms, but they don't cover all C entries correctly?
        # Term 1 covers C[i0, k0].
        # Term 2 covers C[i0, k0].
        # Term 3 covers C[i1, k1].
        # Term 4 covers C[i1, k1].
        # Missing C[i0, k1] and C[i1, k0].
        
        # Let's re-implement add_2x2_block properly.
        
    # Clear U, V, W and col
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    def add_2x2_block_correct(i0, i1, j0, j1, k0, k1):
        # C[i0, k0] += A[i0, j0] B[j0, k0]
        add_term({(i0, j0): 1}, {(j0, k0): 1}, {(k0, i0): 1})
        # C[i0, k0] += A[i0, j1] B[j1, k0]
        add_term({(i0, j1): 1}, {(j1, k0): 1}, {(k0, i0): 1})
        # C[i0, k1] += A[i0, j0] B[j0, k1]
        add_term({(i0, j0): 1}, {(j0, k1): 1}, {(k1, i0): 1})
        # C[i0, k1] += A[i0, j1] B[j1, k1]
        add_term({(i0, j1): 1}, {(j1, k1): 1}, {(k1, i0): 1})
        # C[i1, k0] += A[i1, j0] B[j0, k0]
        add_term({(i1, j0): 1}, {(j0, k0): 1}, {(k0, i1): 1})
        # C[i1, k0] += A[i1, j1] B[j1, k0]
        add_term({(i1, j1): 1}, {(j1, k0): 1}, {(k0, i1): 1})
        # C[i1, k1] += A[i1, j0] B[j0, k1]
        add_term({(i1, j0): 1}, {(j0, k1): 1}, {(k1, i1): 1})
        # C[i1, k1] += A[i1, j1] B[j1, k1]
        add_term({(i1, j1): 1}, {(j1, k1): 1}, {(k1, i1): 1})
        
    # This is rank 8, not 4. Schoolbook is 4 multiplications, but 8 terms in tensor?
    # No, rank is number of terms.
    # M(2,2,2) rank is 4?
    # T[i,j,k] = 1 if i,j,k consistent?
    # No, T is the multiplication tensor.
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # For 2x2, there are 4 terms.
    # So rank 4 is correct.
    # My add_term adds 1 rank-1 tensor.
    # So add_2x2_block_correct adds 8 rank-1 tensors.
    # This is rank 8.
    # But M(2,2,2) is 4.
    # Why?
    # Because the 4 terms are independent?
    # T_{0,0,0} = 1 (A00 B00 -> C00)
    # T_{0,1,0} = 1 (A01 B10 -> C00)
    # T_{0,0,1} = 1 (A00 B01 -> C01)
    # T_{0,1,1} = 1 (A01 B11 -> C01)
    # ...
    # There are 8 non-zero entries in the 2x2x2 tensor?
    # i in 0..1, j in 0..1, k in 0..1.
    # T[i,j,k] is 1 if the term A[i,j]B[j,k] is part of the sum for C[i,k].
    # For C[0,0], terms are A[0,0]B[0,0] and A[0,1]B[1,0].
    # So T[0,0,0]=1, T[0,1,0]=1.
    # For C[0,1], terms are A[0,0]B[0,1] and A[0,1]B[1,1].
    # So T[0,0,1]=1, T[0,1,1]=1.
    # For C[1,0], terms are A[1,0]B[0,0] and A[1,1]B[1,0].
    # So T[1,0,0]=1, T[1,1,0]=1.
    # For C[1,1], terms are A[1,0]B[0,1] and A[1,1]B[1,1].
    # So T[1,0,1]=1, T[1,1,1]=1.
    # Total 8 non-zeros.
    # Rank of this tensor?
    # It is the identity tensor of size 2? No.
    # It can be decomposed into 4 rank-1 tensors?
    # U0 = [1, 0, 0, 1]^T?
    # Let's just use the 8 terms. Rank 8.
    # Then M(2,3,2) <= 8 + 1 = 9.
    # M(2,3,4) <= 18.
    # M(2,3,8) <= 36.
    
    # So rank 36 is the target.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    def add_2x2_block(i0, i1, j0, j1, k0, k1):
        # 8 terms
        add_term({(i0, j0): 1}, {(j0, k0): 1}, {(k0, i0): 1})
        add_term({(i0, j1): 1}, {(j1, k0): 1}, {(k0, i0): 1})
        add_term({(i0, j0): 1}, {(j0, k1): 1}, {(k1, i0): 1})
        add_term({(i0, j1): 1}, {(j1, k1): 1}, {(k1, i0): 1})
        add_term({(i1, j0): 1}, {(j0, k0): 1}, {(k0, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k0): 1}, {(k0, i1): 1})
        add_term({(i1, j0): 1}, {(j0, k1): 1}, {(k1, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k1): 1}, {(k1, i1): 1})
        
    def add_2x1_block(i0, i1, j0, k0, k1):
        # A is 2x1 (col j0), B is 1x2 (row j0, cols k0, k1)
        # C is 2x2
        # Term: (A[i0, j0] + A[i1, j0]) * (B[j0, k0] + B[j0, k1])
        # This covers all 4 terms of C?
        # P = A0 B0 + A0 B1 + A1 B0 + A1 B1
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        # We can't separate them from P.
        # So we need 4 terms?
        # Or is the tensor rank 1?
        # T[i, j, k] = 1 if j=j0.
        # T[i, j0, k] = 1.
        # This is rank 1.
        # U = [1, 1], V = [1], W = [1, 1].
        # Sum = 1*1*1 = 1.
        # So 1 term is enough.
        add_term({(i0, j0): 1, (i1, j0): 1}, {(j0, k0): 1, (j0, k1): 1}, {(k0, i0): 1, (k0, i1): 1, (k1, i0): 1, (k1, i1): 1})
        
    # Construct 2x3x8
    # Block 1: A(2x3) x B(3x4)
    # Sub 1a: A(2x3) x B(3x2) [cols 0,1]
    #   A(2x2) x B(2x2) [A cols 0,1; B rows 0,1]
    add_2x2_block(0, 1, 0, 1, 0, 1)
    #   A(2x1) x B(1x2) [A col 2; B row 2]
    add_2x1_block(0, 1, 2, 0, 1)
    # Sub 1b: A(2x3) x B(3x2) [cols 2,3]
    #   A(2x2) x B(2x2) [A cols 0,1; B rows 0,1]
    add_2x2_block(0, 1, 0, 1, 2, 3)
    #   A(2x1) x B(1x2) [A col 2; B row 2]
    add_2x1_block(0, 1, 2, 2, 3)
    
    # Block 2: A(2x3) x B(3x4) [cols 4,5,6,7]
    # Sub 2a: A(2x3) x B(3x2) [cols 4,5]
    #   A(2x2) x B(2x2) [A cols 0,1; B rows 0,1]
    add_2x2_block(0, 1, 0, 1, 4, 5)
    #   A(2x1) x B(1x2) [A col 2; B row 2]
    add_2x1_block(0, 1, 2, 4, 5)
    # Sub 2b: A(2x3) x B(3x2) [cols 6,7]
    #   A(2x2) x B(2x2) [A cols 0,1; B rows 0,1]
    add_2x2_block(0, 1, 0, 1, 6, 7)
    #   A(2x1) x B(1x2) [A col 2; B row 2]
    add_2x1_block(0, 1, 2, 6, 7)
    
    # Total rank: 4 * (8 + 1) = 36.
    
    if time.time() - start_time > timeout:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
