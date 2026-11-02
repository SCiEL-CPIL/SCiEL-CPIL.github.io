import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by adding blocks.
    # Total rank will be determined by the number of blocks.
    
    # We'll use lists of lists for U, V, W, appending columns as we go.
    # Or we can pre-allocate if we know the rank.
    # Rank 1 (Strassen 2x2x2) = 7
    # Rank 2 (Strassen 2x2x2) = 7
    # Rank 3 (Strassen 2x2x2) = 7
    # Rank 4 (Schoolbook 2x2x1) = 4
    # Total Rank = 7 + 7 + 7 + 4 = 25.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)] # 4 x 25
    V = [[0] * rank for _ in range(m * p)] # 14 x 25
    W = [[0] * rank for _ in range(p * n)] # 14 x 25
    
    col_offset = 0
    
    def add_strassen(b_col_start):
        nonlocal col_offset
        # b_col_start is the starting column index in B for this 2x2x2 block.
        # B indices in this block: j in {0,1}, k in {0,1}
        # Global B index: j*p + (b_col_start + k)
        
        # M1 = (a+d)(e+h)
        # a=A[0,0], d=A[1,1]
        # e=B[0, k0], h=B[1, k1]
        # C[0,0] and C[1,1] get M1.
        # W indices: C[0,0] -> 0*2+0=0. C[1,1] -> 1*2+1=3.
        
        # We need to add 7 columns to U, V, W.
        
        # Helper to add one term
        def add_term(u_idx, v_idx, w_idx, val=1):
            U[u_idx][col_offset] += val
            V[v_idx][col_offset] += val
            W[w_idx][col_offset] += val
            
        # M1
        # U: a(0,0), d(1,1)
        # V: e(0, k0), h(1, k1)
        # W: c00(0,0), c11(1,1)
        # Note: Strassen M1 = (A00+A11)(B00+B11)
        # This is not a single product of linear forms in U and V directly in the standard tensor definition?
        # Yes it is. U col has 1 at 00, 1 at 11. V col has 1 at 00, 1 at 11.
        # W col has 1 at 00, 1 at 11.
        
        # Wait, the standard Strassen decomposition is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00+B01)
        # M3 = (A00-A10)(B10-B11)
        # M4 = (A01-A11)(B00-B10)
        # M5 = (A00+A01)(B11)
        # M6 = (A10+A11)(B00)
        # M7 = (A00-A01)(B01+B11)
        
        # Let's map these to U, V, W.
        # U row indices: 0:00, 1:01, 2:10, 3:11
        # V row indices: 0:00, 1:01, 2:10, 3:11 (local k=0,1)
        # Global V row: j*p + (b_col_start + k)
        
        # M1:
        # U: 1 at 0, 1 at 3.
        # V: 1 at 0 (j=0,k=0), 1 at 2 (j=1,k=1).
        # W: 1 at 0 (k=0,i=0), 1 at 3 (k=1,i=1).
        # Note: W index is k*n + i.
        # C00 (i=0, k=0) -> W[0].
        # C11 (i=1, k=1) -> W[3].
        
        # M2:
        # U: 1 at 2, 1 at 3.
        # V: 1 at 0 (j=0,k=0), 1 at 1 (j=0,k=1).
        # W: 1 at 0 (k=0,i=0)? No.
        # C10 (i=1, k=0) -> W[0*2+1]=1? No, W[k*n+i].
        # C10: i=1, k=0 -> W[0*2+1] = W[1].
        # C00? No.
        # C = [[C00, C01], [C10, C11]]
        # C00 = M1 - M2 + M4 - M6
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M3 - M5 + M7
        
        # So M2 contributes to C00 and C10.
        # C00: W[0]. C10: W[1].
        # W: 1 at 0, 1 at 1.
        
        # M3:
        # U: 1 at 0, -1 at 2.
        # V: 1 at 2 (j=1,k=0), -1 at 3 (j=1,k=1).
        # W: 1 at 2 (k=0, i=1? No. C01: i=0, k=1 -> W[1*2+0]=2).
        # C01: W[2].
        # W: 1 at 2.
        
        # M4:
        # U: 1 at 1, -1 at 3.
        # V: 1 at 0 (j=0,k=0), -1 at 2 (j=1,k=0).
        # W: 1 at 0 (C00).
        # W: 1 at 0.
        
        # M5:
        # U: 1 at 0, 1 at 1.
        # V: 1 at 3 (j=1,k=1).
        # W: 1 at 2 (C01).
        # W: 1 at 2.
        
        # M6:
        # U: 1 at 2, 1 at 3.
        # V: 1 at 0 (j=0,k=0).
        # W: 1 at 0 (C00), 1 at 1 (C10).
        # W: 1 at 0, 1 at 1.
        
        # M7:
        # U: 1 at 0, -1 at 1.
        # V: 1 at 1 (j=0,k=1), 1 at 3 (j=1,k=1).
        # W: 1 at 3 (C11).
        # W: 1 at 3.
        
        # Let's implement these 7 steps.
        
        # M1
        U[0][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + b_col_start + 0][col_offset] += 1; V[1*p + b_col_start + 1][col_offset] += 1
        W[0*n + 0][col_offset] += 1; W[1*n + 1][col_offset] += 1
        col_offset += 1
        
        # M2
        U[2][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + b_col_start + 0][col_offset] += 1; V[0*p + b_col_start + 1][col_offset] += 1
        W[0*n + 0][col_offset] += 1; W[0*n + 1][col_offset] += 1
        col_offset += 1
        
        # M3
        U[0][col_offset] += 1; U[2][col_offset] -= 1
        V[1*p + b_col_start + 0][col_offset] += 1; V[1*p + b_col_start + 1][col_offset] -= 1
        W[1*n + 0][col_offset] += 1
        col_offset += 1
        
        # M4
        U[1][col_offset] += 1; U[3][col_offset] -= 1
        V[0*p + b_col_start + 0][col_offset] += 1; V[1*p + b_col_start + 0][col_offset] -= 1
        W[0*n + 0][col_offset] += 1
        col_offset += 1
        
        # M5
        U[0][col_offset] += 1; U[1][col_offset] += 1
        V[1*p + b_col_start + 1][col_offset] += 1
        W[1*n + 0][col_offset] += 1
        col_offset += 1
        
        # M6
        U[2][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + b_col_start + 0][col_offset] += 1
        W[0*n + 0][col_offset] += 1; W[0*n + 1][col_offset] += 1
        col_offset += 1
        
        # M7
        U[0][col_offset] += 1; U[1][col_offset] -= 1
        V[0*p + b_col_start + 1][col_offset] += 1; V[1*p + b_col_start + 1][col_offset] += 1
        W[1*n + 1][col_offset] += 1
        col_offset += 1

    def add_schoolbook_1col(b_col_start):
        nonlocal col_offset
        # B col index is b_col_start.
        # j in {0,1}, k is fixed (0 in local 1-col sense, but global col is b_col_start).
        # V index: j*p + b_col_start.
        
        # C00 = A00*B0k + A01*B1k
        # C01 = A00*B1k? No.
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Here k is the output column index? No, C is 2x7.
        # Wait, the output C is 2x7.
        # The task says "multiplying a 2 by 2 matrix by a 2 by 7 matrix".
        # So A is 2x2, B is 2x7. C is 2x7.
        # My previous analysis assumed C was 2x2 for the sub-blocks.
        # But the sub-blocks of B are 2x2 and 2x1.
        # So the sub-outputs C1 are 2x2 and C2 is 2x1.
        # The W matrix indices are $k \times n + i$.
        # For the full problem, $k$ ranges from 0 to 6.
        # So W has $7 \times 2 = 14$ rows.
        
        # In add_strassen, I used local k=0,1.
        # I need to map local k to global k.
        # In add_strassen(b_col_start), the local k=0 maps to global k=b_col_start.
        # Local k=1 maps to global k=b_col_start+1.
        
        # Let's re-verify W indices in add_strassen.
        # W index = global_k * n + i.
        # In M1:
        # C00 (i=0, global_k=b_col_start) -> W[b_col_start * 2 + 0]
        # C11 (i=1, global_k=b_col_start+1) -> W[(b_col_start+1) * 2 + 1]
        
        # I hardcoded W[0] and W[3] in the draft above. That is WRONG for general b_col_start.
        # I must use dynamic W indices.
        
        pass

    # Let's rewrite the helpers with correct W indices.
    
    # Reset col_offset
    col_offset = 0
    
    def add_strassen(b_col_start):
        nonlocal col_offset
        k0 = b_col_start
        k1 = b_col_start + 1
        
        # M1: (A00+A11)(B0k0+B1k1) -> C0k0, C1k1
        U[0][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + k0][col_offset] += 1; V[1*p + k1][col_offset] += 1
        W[k0*n + 0][col_offset] += 1; W[k1*n + 1][col_offset] += 1
        col_offset += 1
        
        # M2: (A10+A11)(B0k0+B0k1) -> C0k0, C1k0
        U[2][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + k0][col_offset] += 1; V[0*p + k1][col_offset] += 1
        W[k0*n + 0][col_offset] += 1; W[k0*n + 1][col_offset] += 1
        col_offset += 1
        
        # M3: (A00-A10)(B1k0-B1k1) -> C0k1
        U[0][col_offset] += 1; U[2][col_offset] -= 1
        V[1*p + k0][col_offset] += 1; V[1*p + k1][col_offset] -= 1
        W[k1*n + 0][col_offset] += 1
        col_offset += 1
        
        # M4: (A01-A11)(B0k0-B1k0) -> C0k0
        U[1][col_offset] += 1; U[3][col_offset] -= 1
        V[0*p + k0][col_offset] += 1; V[1*p + k0][col_offset] -= 1
        W[k0*n + 0][col_offset] += 1
        col_offset += 1
        
        # M5: (A00+A01)(B1k1) -> C0k1
        U[0][col_offset] += 1; U[1][col_offset] += 1
        V[1*p + k1][col_offset] += 1
        W[k1*n + 0][col_offset] += 1
        col_offset += 1
        
        # M6: (A10+A11)(B0k0) -> C0k0, C1k0
        U[2][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + k0][col_offset] += 1
        W[k0*n + 0][col_offset] += 1; W[k0*n + 1][col_offset] += 1
        col_offset += 1
        
        # M7: (A00-A01)(B0k1+B1k1) -> C1k1
        U[0][col_offset] += 1; U[1][col_offset] -= 1
        V[0*p + k1][col_offset] += 1; V[1*p + k1][col_offset] += 1
        W[k1*n + 1][col_offset] += 1
        col_offset += 1

    def add_schoolbook_1col(b_col_start):
        nonlocal col_offset
        k = b_col_start
        # C0k = A00*B0k + A01*B1k
        # C1k = A10*B0k + A11*B1k
        
        # M1: A00*B0k -> C0k
        U[0][col_offset] += 1
        V[0*p + k][col_offset] += 1
        W[k*n + 0][col_offset] += 1
        col_offset += 1
        
        # M2: A01*B1k -> C0k
        U[1][col_offset] += 1
        V[1*p + k][col_offset] += 1
        W[k*n + 0][col_offset] += 1
        col_offset += 1
        
        # M3: A10*B0k -> C1k
        U[2][col_offset] += 1
        V[0*p + k][col_offset] += 1
        W[k*n + 1][col_offset] += 1
        col_offset += 1
        
        # M4: A11*B1k -> C1k
        U[3][col_offset] += 1
        V[1*p + k][col_offset] += 1
        W[k*n + 1][col_offset] += 1
        col_offset += 1

    # Apply blocks
    # B cols 0,1 -> Strassen
    add_strassen(0)
    # B cols 2,3 -> Strassen
    add_strassen(2)
    # B cols 4,5 -> Strassen
    add_strassen(4)
    # B col 6 -> Schoolbook
    add_schoolbook_1col(6)
    
    return {"U": U, "V": V, "W": W}
