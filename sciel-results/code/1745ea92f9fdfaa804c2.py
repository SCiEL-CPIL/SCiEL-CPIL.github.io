import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 31 algorithm
    # 7 Strassen mults for top-left 2x2 block
    # 24 schoolbook mults for the rest
    rank = 31
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    def set_term(c, i, j, k, u_val, v_val, w_val):
        U[i * m + j][c] = u_val
        V[j * p + k][c] = v_val
        W[k * n + i][c] = w_val

    # Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # This term contributes to C00, C11
    # In tensor form, a rank-1 term U o V o W
    # U[i,j], V[j,k], W[k,i]
    
    # We need to decompose the Strassen mults into rank-1 tensors.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is a sum of 4 schoolbook terms.
    # Strassen replaces these 4 with 1 multiplication, but in the tensor decomposition,
    # we need to represent the linear combinations.
    # The tensor decomposition is a sum of rank-1 tensors.
    # Each rank-1 tensor corresponds to a "multiplication" in the algorithm.
    # So for M1, we have one rank-1 tensor where:
    # U has 1 at (0,0) and (1,1)
    # V has 1 at (0,0) and (1,1)
    # W has 1 at (0,0) and (1,1)
    # Wait, the product is U[i,j] V[j,k] W[k,i].
    # For M1, we want the term to be (A00+A11)(B00+B11) * (something for C).
    # The Strassen algorithm computes intermediate values.
    # The tensor decomposition directly represents the final sum.
    # So we need to express the final C entries as sums of products of linear forms of A and B.
    # C00 = M1 + M4 - M5 + M7
    # C00 = (A00+A11)(B00+B11) + A11(B01-B00) - A00(B10-B11) + (A01-A11)(B10+B11)
    # This is a sum of 4 products.
    # Each product is a rank-1 term in the decomposition?
    # Yes, each product $L_A(L_B)$ corresponds to a rank-1 tensor.
    # So M1 corresponds to 1 rank-1 tensor.
    # M4 corresponds to 1 rank-1 tensor.
    # etc.
    
    # So for the Strassen block, we have 7 rank-1 tensors.
    # For the rest, we have 24 schoolbook terms, each is 1 rank-1 tensor.
    # Total rank = 7 + 24 = 31.
    
    # Let's define the 7 Strassen terms.
    # Indices for Strassen: i,j in {0,1}, k in {0,1}
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    set_term(0, 0, 0, 0, 1, 1, 1)
    set_term(0, 1, 1, 0, 1, 1, 1) # Wait, V[j,k] and W[k,i] must match.
    # M1 is (A00+A11) * (B00+B11).
    # This product contributes to C00 and C11.
    # In the tensor, the term is U o V o W.
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    # W[0,0]=1, W[1,1]=1
    # Then sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For i=0,j=0,k=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1.
    # For i=1,j=1,k=1: U[1,1]V[1,1]W[1,1] = 1*1*1 = 1.
    # For i=0,j=1,k=0: U[0,1]=0.
    # So this term contributes 1 to T[0,0,0] and T[1,1,1].
    # But we want it to contribute to C00 and C11.
    # The tensor T has 1 at (i,j,k) if A[i,j]B[j,k] contributes to C[i,k].
    # So T[0,0,0] is 1. T[1,1,1] is 1.
    # This matches.
    
    # However, Strassen's M1 is used in C00 and C11.
    # C00 = M1 + ...
    # C11 = M1 + ...
    # So the coefficient of M1 in C00 is 1, and in C11 is 1.
    # In the tensor decomposition, the coefficient is determined by W[k,i].
    # So W[0,0] should be 1 (for C00) and W[1,1] should be 1 (for C11).
    # This is consistent.
    
    # Let's fill the 7 Strassen terms.
    # c=0: M1
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*4+0][0] = 1; V[1*4+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*4+0][1] = 1
    W[0*3+1][1] = 1 # C10
    
    # c=2: M3 = A00(B10-B11)
    U[0*3+0][2] = 1
    V[1*4+0][2] = 1; V[1*4+1][2] = -1
    W[0*3+0][2] = 1 # C00
    
    # c=3: M4 = A11(B01-B00)
    U[1*3+1][3] = 1
    V[0*4+1][3] = 1; V[0*4+0][3] = -1
    W[1*3+1][3] = 1 # C11
    
    # c=4: M5 = (A00+A01)B11
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*4+1][4] = 1
    W[1*3+0][4] = 1 # C01
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*4+0][5] = 1; V[0*4+1][5] = 1
    W[0*3+1][5] = 1 # C10
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*4+0][6] = 1; V[1*4+1][6] = 1
    W[1*3+0][6] = 1 # C01
    
    # Now the 24 schoolbook terms for the rest.
    # Rest: 
    # C02, C03, C12, C13, C20, C21, C22, C23
    # Each has 3 terms.
    # C02: A00B02, A01B12, A02B22
    # C03: A00B03, A01B13, A02B23
    # C12: A10B02, A11B12, A12B22
    # C13: A10B03, A11B13, A12B23
    # C20: A20B00, A21B10, A22B20
    # C21: A20B01, A21B11, A22B21
    # C22: A20B02, A21B12, A22B22
    # C23: A20B03, A21B13, A22B23
    
    col = 7
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [0, 1, 2, 3]:
                # Skip if i in {0,1} and j in {0,1} and k in {0,1}
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*3+j][col] = 1
                V[j*4+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
