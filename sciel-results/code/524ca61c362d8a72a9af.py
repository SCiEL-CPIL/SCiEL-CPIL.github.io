import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 59.
    # Start with schoolbook rank 60, then replace 8 terms with 7 Strassen terms.
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We don't know the final rank upfront if we build it dynamically, 
    # but we know it will be 59.
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen decomposition for 2x2x2
    # Indices for the sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: (i, j) -> i*2 + j? No, in the full matrix A is 3x4.
    # We are using the sub-block A[0:2, 0:2] and B[0:2, 0:2].
    # So A indices are (0,0), (0,1), (1,0), (1,1).
    # B indices are (0,0), (0,1), (1,0), (1,1).
    # Output C indices are (k, i) for k in {0,1}, i in {0,1}.
    # W index is k*n + i.
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Note: C indices in Strassen are (i, k) for matrix multiplication C = AB.
    # But our W index is k*n + i, which corresponds to C[k, i] in a p x n matrix.
    # If C = AB, C is 3x5. C[i, k].
    # If W index is k*n + i, it maps to row k, col i.
    # This implies the stored matrix is C^T or the indices are swapped.
    # Let's assume the standard term W[k*n+i] = 1 corresponds to C[i,k] = 1.
    # So W[k*n+i] is the coefficient for C[i,k].
    
    # Strassen output coefficients for C[i,k]:
    # C[0,0] (i=0, k=0): M1 + M4 - M5 + M7
    # C[0,1] (i=0, k=1): M3 + M5
    # C[1,0] (i=1, k=0): M2 + M4
    # C[1,1] (i=1, k=1): M1 - M2 + M3 + M6
    
    # Let's define the 7 terms.
    # Each term has U coeffs, V coeffs, W coeffs.
    
    strassen_terms = []
    
    # Helper to add a term
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        # u_coeffs: dict {(i,j): val}
        # v_coeffs: dict {(j,k): val}
        # w_coeffs: dict {(i,k): val} -> stored at W[k*n+i]
        
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k][col] += val
        for (i, k), val in w_coeffs.items():
            if val != 0:
                W[k * n + i][col] += val
        col += 1

    # M1 = (A00+A11)(B00+B11)
    # U: (0,0):1, (1,1):1
    # V: (0,0):1, (0,1):1, (1,0):1, (1,1):1 -> Wait, (B00+B11) means B[0,0] and B[1,1].
    # V indices are (j,k). B[0,0] is j=0,k=0. B[1,1] is j=1,k=1.
    # So V: (0,0):1, (1,1):1.
    # W: C[0,0]:1, C[0,1]:0, C[1,0]:0, C[1,1]:1?
    # No, W coeffs are the coefficients in the sum for C[i,k].
    # For M1, it contributes 1 to C[0,0] and 1 to C[1,1]?
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes 1 to C00 and 1 to C11.
    # W coeffs for M1: {(0,0):1, (1,1):1}
    
    add_strassen_term(
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    )
    
    # M2 = (A10+A11)B00
    # U: (1,0):1, (1,1):1
    # V: (0,0):1
    # W: M2 contributes 1 to C10, -1 to C11.
    add_strassen_term(
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1, (1,1):-1}
    )
    
    # M3 = A00(B10-B11)
    # U: (0,0):1
    # V: (1,0):1, (1,1):-1
    # W: M3 contributes 1 to C01, 1 to C11.
    add_strassen_term(
        {(0,0):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1}
    )
    
    # M4 = A11(B01-B00)
    # U: (1,1):1
    # V: (0,1):1, (0,0):-1
    # W: M4 contributes 1 to C00, 1 to C10.
    add_strassen_term(
        {(1,1):1},
        {(0,1):1, (0,0):-1},
        {(0,0):1, (1,0):1}
    )
    
    # M5 = (A00+A01)B11
    # U: (0,0):1, (0,1):1
    # V: (1,1):1
    # W: M5 contributes -1 to C00, 1 to C01.
    add_strassen_term(
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,0):-1, (0,1):1}
    )
    
    # M6 = (A10-A00)(B00+B01)
    # U: (1,0):1, (0,0):-1
    # V: (0,0):1, (0,1):1
    # W: M6 contributes 1 to C11.
    add_strassen_term(
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    )
    
    # M7 = (A01-A11)(B10+B11)
    # U: (0,1):1, (1,1):-1
    # V: (1,0):1, (1,1):1
    # W: M7 contributes 1 to C00.
    add_strassen_term(
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1}
    )
    
    # Now add the remaining 52 standard terms.
    # Total standard terms: 60.
    # We replaced 8 terms: i in {0,1}, j in {0,1}, k in {0,1}.
    # We need to add all other standard terms.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 8 terms already handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                if col >= rank:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # If we didn't fill all rank columns, it means we had fewer terms.
    # But we expect exactly 7 + 52 = 59 terms.
    # If col < rank, we have unused columns (zeros), which is fine.
    # If col > rank, we have an error.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
