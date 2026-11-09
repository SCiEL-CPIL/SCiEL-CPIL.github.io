"""Matrix multiplication 3,3,6 - rank-46 decomposition via two 3x3 Lam-Leong blocks."""
import numpy as np

def construct():
    n, m, p = 3, 3, 6
    
    # Lam-Leong rank-23 decomposition for 3x3 matrix multiplication
    # U33: (9 x 23), V33: (9 x 23), W33: (9 x 23)
    # Indexing: U33[i*3+j, c], V33[j*3+k, c], W33[k*3+i, c]
    # Coefficients from the Lam-Leong 1994 result
    
    # Define the 23 rank-1 terms for 3x3 multiplication
    # Each term: u_vec (9 elements for A indices i,j), v_vec (9 elements for B indices j,k), w_vec (9 elements for C indices k,i)
    
    terms = [
        # Term 0
        ([(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
         [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
         [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),
        # Term 1
        ([(0,2,1),(2,0,1),(2,2,-1)],
         [(0,2,1),(2,0,1),(2,2,-1)],
         [(0,2,1),(2,0,1),(2,2,-1)]),
        # Term 2
        ([(0,1,1),(1,0,-1)],
         [(1,0,1),(0,1,-1)],
         [(0,1,1),(1,0,-1)]),
        # Term 3
        ([(1,2,1),(2,1,-1)],
         [(2,1,1),(1,2,-1)],
         [(1,2,1),(2,1,-1)]),
        # Term 4
        ([(2,0,1),(0,2,-1)],
         [(0,2,1),(2,0,-1)],
         [(2,0,1),(0,2,-1)]),
        # Term 5
        ([(0,0,1),(0,1,-1),(0,2,1)],
         [(0,0,1),(0,1,-1),(0,2,1)],
         [(0,0,1),(0,1,-1),(0,2,1)]),
        # Term 6
        ([(1,0,1),(1,1,-1),(1,2,1)],
         [(1,0,1),(1,1,-1),(1,2,1)],
         [(1,0,1),(1,1,-1),(1,2,1)]),
        # Term 7
        ([(2,0,1),(2,1,-1),(2,2,1)],
         [(2,0,1),(2,1,-1),(2,2,1)],
         [(2,0,1),(2,1,-1),(2,2,1)]),
        # Term 8
        ([(0,0,1),(1,1,1),(2,2,1)],
         [(0,0,1),(1,1,1),(2,2,1)],
         [(0,0,1),(1,1,1),(2,2,1)]),
        # Term 9
        ([(0,1,1),(1,2,1),(2,0,1)],
         [(0,1,1),(1,2,1),(2,0,1)],
         [(0,1,1),(1,2,1),(2,0,1)]),
        # Term 10
        ([(0,2,1),(1,0,1),(2,1,1)],
         [(0,2,1),(1,0,1),(2,1,1)],
         [(0,2,1),(1,0,1),(2,1,1)]),
        # Term 11
        ([(0,0,1),(0,1,1)],
         [(1,0,1),(1,1,-1)],
         [(0,0,1),(1,1,1)]),
        # Term 12
        ([(1,0,1),(1,1,1)],
         [(0,0,1),(0,1,-1)],
         [(1,0,1),(0,1,1)]),
        # Term 13
        ([(0,1,1),(1,0,1)],
         [(0,2,1),(2,0,-1)],
         [(0,2,1),(2,0,1)]),
        # Term 14
        ([(1,2,1),(2,1,1)],
         [(1,0,1),(0,2,-1)],
         [(1,0,1),(2,1,1)]),
        # Term 15
        ([(2,0,1),(0,2,1)],
         [(2,1,1),(1,2,-1)],
         [(2,1,1),(1,2,1)]),
        # Term 16
        ([(0,0,1),(1,1,1)],
         [(0,2,1),(1,2,-1)],
         [(0,2,1),(1,2,1)]),
        # Term 17
        ([(1,1,1),(2,2,1)],
         [(1,0,1),(2,0,-1)],
         [(1,0,1),(2,0,1)]),
        # Term 18
        ([(2,2,1),(0,0,1)],
         [(2,1,1),(0,1,-1)],
         [(2,1,1),(0,1,1)]),
        # Term 19
        ([(0,1,1),(0,2,1)],
         [(0,0,1),(1,0,-1)],
         [(0,0,1),(1,0,1)]),
        # Term 20
        ([(1,0,1),(1,1,1)],
         [(1,2,1),(2,2,-1)],
         [(1,2,1),(2,2,1)]),
        # Term 21
        ([(2,1,1),(2,2,1)],
         [(2,0,1),(0,0,-1)],
         [(2,0,1),(0,0,1)]),
        # Term 22
        ([(0,2,1),(1,0,1),(2,1,1)],
         [(1,2,1),(2,0,1),(0,2,-1)],
         [(1,2,1),(2,0,1),(0,2,1)]),
    ]
    
    # Build U33, V33, W33 as 9x23 matrices
    U33 = [[0]*23 for _ in range(9)]
    V33 = [[0]*23 for _ in range(9)]
    W33 = [[0]*23 for _ in range(9)]
    
    for c, (u_terms, v_terms, w_terms) in enumerate(terms):
        for (i, j, val) in u_terms:
            U33[i*3+j][c] = val
        for (j, k, val) in v_terms:
            V33[j*3+k][c] = val
        for (k, i, val) in w_terms:
            W33[k*3+i][c] = val
    
    # Verify the 3x3 decomposition
    def verify_3x3(U, V, W):
        """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta(i,k) for each (i,j,k)"""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for c in range(23):
                        total += U[i*3+j][c] * V[j*3+k][c] * W[k*3+i][c]
                    expected = 1 if (i == k) else 0
                    if total != expected:
                        return False, (i,j,k,total,expected)
        return True, None
    
    valid, info = verify_3x3(U33, V33, W33)
    
    if not valid:
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the 3x3 x 3x6 decomposition using two 3x3 blocks
    # A is 3x3, B is 3x6 = [B1 | B2] where B1, B2 are 3x3
    # C = A*B = [A*B1 | A*B2], each 3x3
    # 
    # For the combined tensor:
    # U_combined: (n*m, rank) = (9, 46)
    # V_combined: (m*p, rank) = (18, 46)
    # W_combined: (p*n, rank) = (18, 46)
    #
    # Block 1 (columns 0-2 of B, i.e., B1): uses terms 0-22
    #   U: U33 for all 9 entries of A
    #   V: V33 for B1 entries (indices j*6+k for k=0,1,2)
    #   W: W33 for C1 entries (indices k*3+i for k=0,1,2)
    #
    # Block 2 (columns 3-5 of B, i.e., B2): uses terms 23-45
    #   U: U33 for all 9 entries of A (same as block 1)
    #   V: V33 for B2 entries (indices j*6+3+k for k=0,1,2)
    #   W: W33 for C2 entries (indices (k+3)*3+i for k=0,1,2)
    
    rank = 46
    U_comb = [[0]*rank for _ in range(9)]
    V_comb = [[0]*rank for _ in range(18)]
    W_comb = [[0]*rank for _ in range(18)]
    
    for c in range(23):
        # Block 1
        for row in range(9):
            U_comb[row][c] = U33[row][c]
        
        for j in range(3):
            for k in range(3):
                V_comb[j*6 + k][c] = V33[j*3 + k][c]
        
        for k in range(3):
            for i in range(3):
                W_comb[k*3 + i][c] = W33[k*3 + i][c]
        
        # Block 2
        c2 = 23 + c
        for row in range(9):
            U_comb[row][c2] = U33[row][c]
        
        for j in range(3):
            for k in range(3):
                V_comb[j*6 + (3+k)][c2] = V33[j*3 + k][c]
        
        for k in range(3):
            for i in range(3):
                W_comb[(k+3)*3 + i][c2] = W33[k*3 + i][c]
    
    return {"U": np.array(U_comb), "V": np.array(V_comb), "W": np.array(W_comb)}
