"""Matrix multiplication 4x4 * 4x7 tensor decomposition.

Uses a block-level decomposition:
- 4x4x4 part: 7 block products (Strassen-style) x 4 entries each = 28 terms... 
  Actually, let me think more carefully.

The approach: partition A into 2x2 blocks, B into 2x2 blocks (for first 4 cols),
and C into 2x2 blocks. Use 7 bilinear forms to compute all 4 block products.
Each block product contributes 4 scalar terms (one per output entry of the 2x2 block).
So 7 * 4 = 28 terms for the 4x4x4 part.

For the last 3 columns of B (4x3), we can't use 2x2 blocks directly.
We can use 2x2x3 or 2x2x2 + 2x2x1 approaches.

Actually, let me use a cleaner approach:
- For columns 0-3 of B: use 2x2 block Strassen (7 products, each giving 2x2 output)
- For columns 4-6 of B: use schoolbook or another block approach

Let me implement the verified approach from the inspiration more carefully.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    """Verify that the decomposition reproduces the matmul tensor."""
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=4, m=4, p=7):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_block_decomposition():
    """Build a block-level decomposition.
    
    Strategy:
    - For B columns 0-3: Use 7 Strassen products on 2x2 blocks
      Each product Pi = Xi * Yi where Xi, Yi are 2x2 matrices
      Pi is a 2x2 matrix, contributing to a 2x2 block of C
      Each of the 4 entries of Pi is a sum over the intermediate index
      So each Pi gives 4 rank-1 terms? No, each Pi[r,c] = sum_a Xi[r,a]*Yi[a,c]
      This is NOT rank-1 in the original tensor sense.
      
    Let me reconsider. The correct approach from the inspiration:
    
    For each of the 7 Strassen products, and for each of the 4 output positions
    of the 2x2 result, we create ONE term. In that term:
    - U has non-zero entries at specific (i,j) positions (linear combo of A entries)
    - V has non-zero entries at specific (j,k) positions (linear combo of B entries)  
    - W has non-zero entries at specific (k,i) positions (the output coefficient)
    
    The key insight: Xi[r,a] is a linear combination of A entries, and Yi[a,c]
    is a linear combination of B entries. The product Xi[r,a]*Yi[a,c] expands to
    a sum of A[i,j]*B[j',k] terms. For this to be a valid rank-1 tensor term,
    we need j=j' for all non-zero terms.
    
    This works when the block structure is aligned! For example:
    Xi[0,0] = A[0,0] + A[2,2] (uses j=0 and j=2)
    Yi[0,0] = B[0,0] + B[2,2] (uses j=0 and j=2)
    Product: A[0,0]*B[0,0] + A[0,0]*B[2,2] + A[2,2]*B[0,0] + A[2,2]*B[2,2]
    
    In the matmul tensor, A[0,0]*B[0,0] has j=0, A[0,0]*B[2,2] has j=0 in A but 
    j=2 in B - these don't match! So this naive expansion doesn't work.
    
    The resolution from the inspiration: they're treating this differently. Let me 
    look at their code more carefully...
    
    Actually, I think the correct interpretation is that each "term" in their 
    decomposition corresponds to a specific (r,a,c) combination for a given 
    Strassen product, and the U, V, W vectors encode the full linear combination.
    
    Let me just implement it directly and verify.
    """
    n, m, p = 4, 4, 7
    
    # The 7 Strassen products for 2x2 matrices:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # For each product, define:
    # X_entries: dict from (local_row, local_col) to list of (global_i, global_j, coeff)
    # Y_entries: dict from (local_row, local_col) to list of (global_j, global_k, coeff)
    # C_entries: dict from (local_row, local_col) to list of (global_i, global_k, coeff)
    
    # For the 4x4x4 part (columns 0-3 of B, rows 0-3 of C):
    # Block (br, bc) of A covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Block (br, bc) of B covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Block (br, bc) of C covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    
    strassen_products = [
        # P1: X = A11+A22, Y = B11+B22, contributes to C11 and C22
        {
            'X': {(0,0): [(0,0,1),(2,2,1)], (0,1): [(0,1,1),(2,3,1)],
                  (1,0): [(1,0,1),(3,2,1)], (1,1): [(1,1,1),(3,3,1)]},
            'Y': {(0,0): [(0,0,1),(2,2,1)], (0,1): [(0,1,1),(2,3,1)],
                  (1,0): [(1,0,1),(3,2,1)], (1,1): [(1,1,1),(3,3,1)]},
            'C': {(0,0): [(0,0,1)], (1,1): [(3,3,1)]}  # C11[0,0] and C22[1,1] -> C[0,0] and C[3,3]
        },
        # P2: X = A21+A22, Y = B11, contributes to C21
        {
            'X': {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                  (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]},
            'Y': {(0,0): [(0,0,1)], (0,1): [(0,1,1)],
                  (1,0): [(1,0,1)], (1,1): [(1,1,1)]},
            'C': {(1,0): [(2,0,1)], (1,1): [(3,0,1)]}  # C21 block
        },
        # P3: X = A11, Y = B12-B22, contributes to C12
        {
            'X': {(0,0): [(0,0,1)], (0,1): [(0,1,1)],
                  (1,0): [(1,0,1)], (1,1): [(1,1,1)]},
            'Y': {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                  (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]},
            'C': {(0,1): [(0,2,1)], (1,1): [(1,3,1)]}  # C12 block
        },
        # P4: X = A22, Y = B21-B11, contributes to C11 and C21
        {
            'X': {(0,0): [(2,2,1)], (0,1): [(2,3,1)],
                  (1,0): [(3,2,1)], (1,1): [(3,3,1)]},
            'Y': {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                  (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]},
            'C': {(0,0): [(0,0,1)], (1,0): [(2,0,1)]}  # C11[0,0] and C21[1,0]
        },
        # P5: X = A11+A12, Y = B22, contributes to C11 and C12
        {
            'X': {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                  (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]},
            'Y': {(0,0): [(2,2,1)], (0,1): [(2,3,1)],
                  (1,0): [(3,2,1)], (1,1): [(3,3,1)]},
            'C': {(0,0): [(0,0,1)], (0,1): [(0,2,1)]}  # C11[0,0] and C12[0,1]
        },
        # P6: X = A21-A11, Y = B11+B12, contributes to C22
        {
            'X': {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                  (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]},
            'Y': {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                  (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]},
            'C': {(1,1): [(3,3,1)]}  # C22[1,1] -> C[3,3]
        },
        # P7: X = A12-A22, Y = B21+B22, contributes to C11 and C22
        {
            'X': {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                  (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]},
            'Y': {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                  (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]},
            'C': {(0,0): [(0,0,1)], (1,1): [(3,3,1)]}  # C11[0,0] and C22[1,1]
        },
    ]
    
    # For each Strassen product and each output position, create a term
    # Pi[r,c] = sum_a Xi[r,a] * Yi[a,c]
    # For each a in {0,1}, create a term where:
    #   U[i,j] = coeff from Xi[r,a] for A[i,j]
    #   V[j,k] = coeff from Yi[a,c] for B[j,k]
    #   W[k,i] = coeff from C for output position
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    for prod in strassen_products:
        X = prod['X']
        Y = prod['Y']
        C = prod['C']
        
        for r in range(2):
            for c in range(2):
                # Find which C position this contributes to
                c_coeff = 0.0
                c_pos = None
                for (cr, cc), c_list in C.items():
                    if cr == r and cc == c:
                        for (gi, gk, coeff) in c_list:
                            c_coeff = coeff
                            c_pos = (gi, gk)
                            break
                    if c_pos is not None:
                        break
                
                if c_pos is None:
                    continue
                    
                for a in range(2):
                    # Xi[r,a] and Yi[a,c]
                    x_entries = X.get((r, a), [])
                    y_entries = Y.get((a, c), [])
                    
                    if not x_entries or not y_entries:
                        continue
                    
                    U_dict = {}
                    V_dict = {}
                    
                    for (gi, gj, coeff) in x_entries:
                        U_dict[(gi, gj)] = coeff
                    
                    for (gj, gk, coeff) in y_entries:
                        V_dict[(gj, gk)] = coeff
                    
                    W_dict = {c_pos: c_coeff}
                    
                    terms.append((U_dict, V_dict, W_dict))
    
    # Now handle the last 3 columns of B (columns 4,5,6)
    # For these, we can use schoolbook: 4*4*3 = 48 terms
    # Or we can try a block approach
    
    # Schoolbook for columns 4-6:
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):  # columns 4,5,6
                U_dict = {(i, j): 1.0}
                V_dict = {(j, k): 1.0}
                W_dict = {(k, i): 1.0}
                terms.append((U_dict, V_dict, W_dict))
    
    # Build the matrices
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(terms):
        for (i, j), coeff in U_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try the block decomposition
    result = _build_block_decomposition()
    
    # Verify it
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fallback to schoolbook
    result = _schoolbook(n, m, p)
    return result
