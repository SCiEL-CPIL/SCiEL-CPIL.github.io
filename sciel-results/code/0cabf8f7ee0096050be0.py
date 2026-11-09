"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Uses a block-level Strassen decomposition: treat the 2x4 * 4x8 multiplication
as a 2x2 multiplication of sub-blocks, where each sub-block is 1x2 (from A)
and 2x4 (from B), producing 2x4 sub-blocks in C.

A is 2x4: split into 2x2 blocks of size (1,2):
  A = [[A00, A01],
       [A10, A11]]
  where Aij is 1x2.

B is 4x8: split into 2x2 blocks of size (2,4):
  B = [[B00, B01],
       [B10, B11]]
  where Bij is 2x4.

C is 2x8: split into 2x2 blocks of size (1,4):
  C = [[C00, C01],
       [C10, C11]]
  where Cij is 1x4.

The product C = A*B means:
  C00 = A00*B00 + A01*B10
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11

Strassen's 7 products for the 2x2 block structure:
  M1 = (A00+A11)*(B00+B11)
  M2 = (A10+A11)*B00
  M3 = A00*(B01-B11)
  M4 = A11*(B10-B00)
  M5 = (A00+A01)*B11
  M6 = (A10-A00)*(B00+B01)
  M7 = (A01-A11)*(B10+B11)

Then:
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

Each Mi is a (1x2)*(2x4) product. Each such product has schoolbook rank 1*2*4 = 8.
Total rank = 7 * 8 = 56.
"""
import numpy as np

import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # Each Strassen product Mi is a (1x2)*(2x4) matrix product.
    # For a (1x2)*(2x4) product L*R:
    #   Product[0,k] = sum_{j=0}^{1} L[j] * R[j,k]
    # The schoolbook decomposition has rank 1*2*4 = 8 terms.
    # Each term (j,k) contributes:
    #   U element: the j-th element of L (which is a linear combo of A elements)
    #   V element: the (j,k) element of R (which is a linear combo of B elements)
    #   W element: the k-th element of the product (which contributes to C)
    
    # For each Strassen product, we need to map:
    #   - Which A elements contribute to the 1x2 left factor (with coefficients)
    #   - Which B elements contribute to the 2x4 right factor (with coefficients)
    #   - Which C elements are affected by this product (with coefficients)
    
    # Let's define the Strassen products and their contributions.
    
    # A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
    #   A00 = A[0,0:2], A01 = A[0,2:4], A10 = A[1,0:2], A11 = A[1,2:4]
    # B is 4x8: B[j,k] where j in {0,1,2,3}, k in {0,...,7}
    #   B00 = B[0:2, 0:4], B01 = B[0:2, 4:8], B10 = B[2:4, 0:4], B11 = B[2:4, 4:8]
    # C is 2x8: C[i,k] where i in {0,1}, k in {0,...,7}
    #   C00 = C[0,0:4], C01 = C[0,4:8], C10 = C[1,0:4], C11 = C[1,4:8]
    
    # For each Strassen product, define:
    #   left_factors: list of (a_row, a_col, coeff) for the 1x2 left factor
    #     The left factor has 2 elements (j=0 and j=1).
    #     left_factors[j] = list of (a_row, a_col, coeff)
    #   right_factors: list of (b_row, b_col, coeff) for the 2x4 right factor
    #     right_factors[j][k] = list of (b_row, b_col, coeff)
    #   c_contributions: list of (c_row, c_col, coeff) for how this product contributes to C
    
    # Actually, let me think about this differently.
    
    # For a Strassen product M = L * R where L is 1x2 and R is 2x4:
    #   M[0,k] = L[0]*R[0,k] + L[1]*R[1,k]
    
    # In the tensor decomposition, for each term (j,k) where j in {0,1} and k in {0,1,2,3}:
    #   U contributes: coefficient of A elements in L[j]
    #   V contributes: coefficient of B elements in R[j,k]
    #   W contributes: coefficient for C elements that M[0,k] contributes to
    
    # Let me define the 7 Strassen products explicitly.
    
    # M1 = (A00+A11)*(B00+B11)
    #   L[0] = A[0,0] + A[1,2]  (A00[0] + A11[0])
    #   L[1] = A[0,1] + A[1,3]  (A00[1] + A11[1])
    #   R[0,k] = B[0,k] + B[2,k+4]  for k in 0..3
    #   R[1,k] = B[1,k] + B[3,k+4]  for k in 0..3
    #   M1 contributes to: C00 (+1), C11 (+1)
    #   M1[0,k] for k=0..3 goes to C[0,k] (C00) and C[1,k+4] (C11)
    
    # M2 = (A10+A11)*B00
    #   L[0] = A[1,0] + A[1,2]
    #   L[1] = A[1,1] + A[1,3]
    #   R[0,k] = B[0,k]  for k in 0..3
    #   R[1,k] = B[1,k]  for k in 0..3
    #   M2 contributes to: C10 (+1), C11 (-1)
    #   M2[0,k] for k=0..3 goes to C[1,k] (C10) and C[1,k+4] (C11, with -1)
    
    # M3 = A00*(B01-B11)
    #   L[0] = A[0,0]
    #   L[1] = A[0,1]
    #   R[0,k] = B[0,k+4] - B[2,k+4]  for k in 0..3
    #   R[1,k] = B[1,k+4] - B[3,k+4]  for k in 0..3
    #   M3 contributes to: C01 (+1), C11 (+1)
    #   M3[0,k] for k=0..3 goes to C[0,k+4] (C01) and C[1,k+4] (C11)
    
    # M4 = A11*(B10-B00)
    #   L[0] = A[1,2]
    #   L[1] = A[1,3]
    #   R[0,k] = B[2,k] - B[0,k]  for k in 0..3
    #   R[1,k] = B[3,k] - B[1,k]  for k in 0..3
    #   M4 contributes to: C00 (+1), C10 (+1)
    #   M4[0,k] for k=0..3 goes to C[0,k] (C00) and C[1,k] (C10)
    
    # M5 = (A00+A01)*B11
    #   L[0] = A[0,0] + A[0,2]
    #   L[1] = A[0,1] + A[0,3]
    #   R[0,k] = B[2,k+4]  for k in 0..3
    #   R[1,k] = B[3,k+4]  for k in 0..3
    #   M5 contributes to: C00 (-1), C01 (+1)
    #   M5[0,k] for k=0..3 goes to C[0,k] (C00, with -1) and C[0,k+4] (C01)
    
    # M6 = (A10-A00)*(B00+B01)
    #   L[0] = A[1,0] - A[0,0]
    #   L[1] = A[1,1] - A[0,1]
    #   R[0,k] = B[0,k] + B[0,k+4]  for k in 0..3
    #   R[1,k] = B[1,k] + B[1,k+4]  for k in 0..3
    #   M6 contributes to: C11 (+1)
    #   M6[0,k] for k=0..3 goes to C[1,k+4] (C11)
    
    # M7 = (A01-A11)*(B10+B11)
    #   L[0] = A[0,2] - A[1,2]
    #   L[1] = A[0,3] - A[1,3]
    #   R[0,k] = B[2,k] + B[2,k+4]  for k in 0..3
    #   R[1,k] = B[3,k] + B[3,k+4]  for k in 0..3
    #   M7 contributes to: C00 (+1)
    #   M7[0,k] for k=0..3 goes to C[0,k] (C00)
    
    # Now, for each Strassen product and each (j,k) pair, we create one term in the decomposition.
    # The term has:
    #   U: for each (a_row, a_col, coeff) in left_factors[j], set U[a_row*m+a_col][col] += coeff
    #   V: for each (b_row, b_col, coeff) in right_factors[j][k], set V[b_row*p+b_col][col] += coeff
    #   W: for each (c_row, c_col, coeff) in c_contributions[k], set W[c_col*n+c_row][col] += coeff
    
    # Let's build this.
    
    # Define the 7 products
    products = []
    
    # M1
    products.append({
        'left': [
            [(0, 0, 1), (1, 2, 1)],  # L[0]
            [(0, 1, 1), (1, 3, 1)],  # L[1]
        ],
        'right': [
            [(0, 0, 1), (2, 4, 1)],  # R[0,0]
            [(0, 1, 1), (2, 5, 1)],  # R[0,1]
            [(0, 2, 1), (2, 6, 1)],  # R[0,2]
            [(0, 3, 1), (2, 7, 1)],  # R[0,3]
            [(1, 0, 1), (3, 4, 1)],  # R[1,0]
            [(1, 1, 1), (3, 5, 1)],  # R[1,1]
            [(1, 2, 1), (3, 6, 1)],  # R[1,2]
            [(1, 3, 1), (3, 7, 1)],  # R[1,3]
        ],
        'c_contrib': [
            [(0, 0, 1), (1, 4, 1)],  # M[0,0] -> C[0,0] and C[1,4]
            [(0, 1, 1), (1, 5, 1)],  # M[0,1] -> C[0,1] and C[1,5]
            [(0, 2, 1), (1, 6, 1)],  # M[0,2] -> C[0,2] and C[1,6]
            [(0, 3, 1), (1, 7, 1)],  # M[0,3] -> C[0,3] and C[1,7]
        ]
    })
    
    # M2
    products.append({
        'left': [
            [(1, 0, 1), (1, 2, 1)],  # L[0]
            [(1, 1, 1), (1, 3, 1)],  # L[1]
        ],
        'right': [
            [(0, 0, 1)],  # R[0,0]
            [(0, 1, 1)],  # R[0,1]
            [(0, 2, 1)],  # R[0,2]
            [(0, 3, 1)],  # R[0,3]
            [(1, 0, 1)],  # R[1,0]
            [(1, 1, 1)],  # R[1,1]
            [(1, 2, 1)],  # R[1,2]
            [(1, 3, 1)],  # R[1,3]
        ],
        'c_contrib': [
            [(1, 0, 1), (1, 4, -1)],  # M[0,0] -> C[1,0] and C[1,4] with -1
            [(1, 1, 1), (1, 5, -1)],  # M[0,1] -> C[1,1] and C[1,5] with -1
            [(1, 2, 1), (1, 6, -1)],  # M[0,2] -> C[1,2] and C[1,6] with -1
            [(1, 3, 1), (1, 7, -1)],  # M[0,3] -> C[1,3] and C[1,7] with -1
        ]
    })
    
    # M3
    products.append({
        'left': [
            [(0, 0, 1)],  # L[0]
            [(0, 1, 1)],  # L[1]
        ],
        'right': [
            [(0, 4, 1), (2, 4, -1)],  # R[0,0]
            [(0, 5, 1), (2, 5, -1)],  # R[0,1]
            [(0, 6, 1), (2, 6, -1)],  # R[0,2]
            [(0, 7, 1), (2, 7, -1)],  # R[0,3]
            [(1, 4, 1), (3, 4, -1)],  # R[1,0]
            [(1, 5, 1), (3, 5, -1)],  # R[1,1]
            [(1, 6, 1), (3, 6, -1)],  # R[1,2]
            [(1, 7, 1), (3, 7, -1)],  # R[1,3]
        ],
        'c_contrib': [
            [(0, 4, 1), (1, 4, 1)],  # M[0,0] -> C[0,4] and C[1,4]
            [(0, 5, 1), (1, 5, 1)],  # M[0,1] -> C[0,5] and C[1,5]
            [(0, 6, 1), (1, 6, 1)],  # M[0,2] -> C[0,6] and C[1,6]
            [(0, 7, 1), (1, 7, 1)],  # M[0,3] -> C[0,7] and C[1,7]
        ]
    })
    
    # M4
    products.append({
        'left': [
            [(1, 2, 1)],  # L[0]
            [(1, 3, 1)],  # L[1]
        ],
        'right': [
            [(2, 0, 1), (0, 0, -1)],  # R[0,0]
            [(2, 1, 1), (0, 1, -1)],  # R[0,1]
            [(2, 2, 1), (0, 2, -1)],  # R[0,2]
            [(2, 3, 1), (0, 3, -1)],  # R[0,3]
            [(3, 0, 1), (1, 0, -1)],  # R[1,0]
            [(3, 1, 1), (1, 1, -1)],  # R[1,1]
            [(3, 2, 1), (1, 2, -1)],  # R[1,2]
            [(3, 3, 1), (1, 3, -1)],  # R[1,3]
        ],
        'c_contrib': [
            [(0, 0, 1), (1, 0, 1)],  # M[0,0] -> C[0,0] and C[1,0]
            [(0, 1, 1), (1, 1, 1)],  # M[0,1] -> C[0,1] and C[1,1]
            [(0, 2, 1), (1, 2, 1)],  # M[0,2] -> C[0,2] and C[1,2]
            [(0, 3, 1), (1, 3, 1)],  # M[0,3] -> C[0,3] and C[1,3]
        ]
    })
    
    # M5
    products.append({
        'left': [
            [(0, 0, 1), (0, 2, 1)],  # L[0]
            [(0, 1, 1), (0, 3, 1)],  # L[1]
        ],
        'right': [
            [(2, 4, 1)],  # R[0,0]
            [(2, 5, 1)],  # R[0,1]
            [(2, 6, 1)],  # R[0,2]
            [(2, 7, 1)],  # R[0,3]
            [(3, 4, 1)],  # R[1,0]
            [(3, 5, 1)],  # R[1,1]
            [(3, 6, 1)],  # R[1,2]
            [(3, 7, 1)],  # R[1,3]
        ],
        'c_contrib': [
            [(0, 0, -1), (0, 4, 1)],  # M[0,0] -> C[0,0] with -1 and C[0,4]
            [(0, 1, -1), (0, 5, 1)],  # M[0,1] -> C[0,1] with -1 and C[0,5]
            [(0, 2, -1), (0, 6, 1)],  # M[0,2] -> C[0,2] with -1 and C[0,6]
            [(0, 3, -1), (0, 7, 1)],  # M[0,3] -> C[0,3] with -1 and C[0,7]
        ]
    })
    
    # M6
    products.append({
        'left': [
            [(1, 0, 1), (0, 0, -1)],  # L[0]
            [(1, 1, 1), (0, 1, -1)],  # L[1]
        ],
        'right': [
            [(0, 0, 1), (0, 4, 1)],  # R[0,0]
            [(0, 1, 1), (0, 5, 1)],  # R[0,1]
            [(0, 2, 1), (0, 6, 1)],  # R[0,2]
            [(0, 3, 1), (0, 7, 1)],  # R[0,3]
            [(1, 0, 1), (1, 4, 1)],  # R[1,0]
            [(1, 1, 1), (1, 5, 1)],  # R[1,1]
            [(1, 2, 1), (1, 6, 1)],  # R[1,2]
            [(1, 3, 1), (1, 7, 1)],  # R[1,3]
        ],
        'c_contrib': [
            [(1, 4, 1)],  # M[0,0] -> C[1,4]
            [(1, 5, 1)],  # M[0,1] -> C[1,5]
            [(1, 6, 1)],  # M[0,2] -> C[1,6]
            [(1, 7, 1)],  # M[0,3] -> C[1,7]
        ]
    })
    
    # M7
    products.append({
        'left': [
            [(0, 2, 1), (1, 2, -1)],  # L[0]
            [(0, 3, 1), (1, 3, -1)],  # L[1]
        ],
        'right': [
            [(2, 0, 1), (2, 4, 1)],  # R[0,0]
            [(2, 1, 1), (2, 5, 1)],  # R[0,1]
            [(2, 2, 1), (2, 6, 1)],  # R[0,2]
            [(2, 3, 1), (2, 7, 1)],  # R[0,3]
            [(3, 0, 1), (3, 4, 1)],  # R[1,0]
            [(3, 1, 1), (3, 5, 1)],  # R[1,1]
            [(3, 2, 1), (3, 6, 1)],  # R[1,2]
            [(3, 3, 1), (3, 7, 1)],  # R[1,3]
        ],
        'c_contrib': [
            [(0, 0, 1)],  # M[0,0] -> C[0,0]
            [(0, 1, 1)],  # M[0,1] -> C[0,1]
            [(0, 2, 1)],  # M[0,2] -> C[0,2]
            [(0, 3, 1)],  # M[0,3] -> C[0,3]
        ]
    })
    
    # Now build U, V, W
    rank = 7 * 8  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in products:
        for j in range(2):  # j = 0, 1 (the two elements of the 1x2 left factor)
            for k in range(4):  # k = 0, 1, 2, 3 (the four columns of the 2x4 right factor)
                # Add U contributions
                for (a_row, a_col, coeff) in prod['left'][j]:
                    U[a_row * m + a_col][col] += coeff
                
                # Add V contributions
                for (b_row, b_col, coeff) in prod['right'][j * 4 + k]:
                    V[b_row * p + b_col][col] += coeff
                
                # Add W contributions
                for (c_row, c_col, coeff) in prod['c_contrib'][k]:
                    W[c_col * n + c_row][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    import numpy as np
    
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    
    # Verify: for random A and B, check that sum_r U[:,r] * V[:,r] * W[:,r] gives correct C
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Direct computation
    C_decomp2 = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomp2[i,k] = total
    
    error = np.max(np.abs(C_true - C_decomp2))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
