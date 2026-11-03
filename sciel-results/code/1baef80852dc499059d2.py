import time

def construct():
    n, m, p = 3, 3, 4
    # We aim for a rank of 31
    # 7 terms for Strassen 2x2x2 on C[0:2, 0:2]
    # 12 terms for C[2, 0:4] (Row 2 of A)
    # 12 terms for C[0:2, 2:4] (Cols 2,3 of B)
    
    rank = 31
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # 1. Strassen for C[0:2, 0:2]
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00, A11. V: B00, B11. W: C00, C11
    # Contribution: C00 += M1, C11 += M1
    # In tensor form:
    # Term 1.1: U[0,0]=1, V[0,0]=1, W[0,0]=1 (A00 B00 -> C00)
    # Term 1.2: U[0,0]=1, V[0,1]=1, W[1,0]=1 (A00 B01 -> C10? No, W index is k*n+i. C[1,0] is k=1, i=0. W[1*3+0]=W[3].)
    # Wait, W[k*n+i].
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[1]
    # C[1,0] -> k=0, i=1 -> W[3]
    # C[1,1] -> k=1, i=1 -> W[4]
    
    # Strassen Terms:
    # M1: (A00+A11)(B00+B11)
    # A00 is U[0*3+0]=U[0]. A11 is U[1*3+1]=U[4].
    # B00 is V[0*4+0]=V[0]. B11 is V[1*4+1]=V[5].
    # C00 is W[0]. C11 is W[4].
    
    # We need to add terms such that the sum of products matches.
    # The standard Strassen decomposition for 2x2x2 has 7 terms.
    # I will hardcode the 7 terms for the 2x2x2 subproblem embedded in indices 0,1.
    
    # Term 0: M1
    # U[0]=1, U[4]=1. V[0]=1, V[5]=1. W[0]=1, W[4]=1.
    # This creates:
    # A00 B00 W0 + A00 B01? No.
    # Sum_j U[i,j] V[j,k] W[k,i].
    # If U[0,0]=1, U[1,1]=1.
    # i=0, k=0: j=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1. j=1: U[0,1]V[1,0]W[0,0] = 0.
    # So this term contributes 1 to C[0,0].
    # i=1, k=1: j=1: U[1,1]V[1,1]W[1,1] = 1*1*1 = 1.
    # So this term contributes 1 to C[1,1].
    # This is correct for M1's contribution to C00 and C11.
    
    # But M1 also needs to contribute to C01 and C10? No, M1 is added to C00, C11.
    # In Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 + M5 - M6
    # C10 = M2 + M3 - M1 + M6
    # C11 = M1 - M4 + M2 - M3
    
    # So M1 contributes +1 to C00, 0 to C01, -1 to C10, +1 to C11.
    # My single term above only contributes to C00 and C11. It does NOT contribute to C10.
    # Why? Because the rank-1 tensor defined by U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    # produces:
    # T[0,0,0] = 1. (C00)
    # T[1,1,1] = 1. (C11)
    # T[0,1,0] = 0. (C10? No, C10 is i=1, k=0. T[1,0,0].)
    # T[1,0,0] = U[1,0]V[0,0]W[0,1] = 0.
    
    # So this term is correct for M1's contribution to C00 and C11.
    # But M1 also needs to be subtracted from C10.
    # This requires a separate term?
    # No, the rank-1 tensor IS the term.
    # If I set W[0,1] = -1 (i.e., W for C10), then T[1,0,0] would be non-zero.
    # But W[0,1] corresponds to k=0, i=1.
    # If I set W[0,0]=1, W[1,1]=1, and W[0,1]=-1?
    # Then T[1,0,0] = U[1,0]V[0,0]W[0,1] = 0 * 1 * -1 = 0.
    # Because U[1,0] is 0.
    
    # The issue is that the "Strassen term" M1 = (A00+A11)(B00+B11) is a product of two sums.
    # In the tensor decomposition, a single rank-1 term corresponds to a product of single elements?
    # NO.
    # A rank-1 term is $U[i,j,r] V[j,k,r] W[k,i,r]$.
    # If $U[0,0,r]=1, U[1,1,r]=1$, then $U$ is not a single element, it's a vector.
    # The product $U[i,j,r] V[j,k,r]$ sums over $j$.
    # For $i=0, k=0$: $U[0,0]V[0,0] + U[0,1]V[1,0]$.
    # If $U[0,0]=1, U[1,1]=1$ and $V[0,0]=1, V[1,1]=1$:
    # $i=0, k=0 \rightarrow 1\cdot1 + 0\cdot0 = 1$.
    # $i=0, k=1 \rightarrow 1\cdot0 + 0\cdot1 = 0$.
    # $i=1, k=0 \rightarrow 0\cdot1 + 1\cdot0 = 0$.
    # $i=1, k=1 \rightarrow 0\cdot0 + 1\cdot1 = 1$.
    
    # So this single term produces the tensor with 1s at (0,0,0) and (1,1,1) and 0 elsewhere.
    # This matches the contribution of M1 to C00 and C11.
    # It does NOT contribute to C01 or C10.
    # This is correct because M1 is only added to C00 and C11.
    
    # So, for each Strassen multiplication Mi, we define ONE column r.
    # The entries in U, V, W for that column are the coefficients of the linear forms.
    
    # Let's define the 7 Strassen terms for indices 0,1.
    
    # M1: (A00+A11)(B00+B11)
    # U: U[0]=1, U[4]=1. V: V[0]=1, V[5]=1. W: W[0]=1, W[4]=1.
    add_term(0, 0, 0, 1, 1, 1)
    add_term(4, 5, 4, 1, 1, 1)
    # Wait, add_term adds to specific indices.
    # I need to set multiple indices in the same column.
    # My add_term function increments col.
    # I need a way to add multiple entries to the SAME column.
    
    # Let's rewrite the helper.
    
    col = 0
    def start_term():
        nonlocal col
        if col >= rank:
            return -1
        c = col
        col += 1
        return c
        
    def set_entry(mat, idx, c, val):
        if c >= 0:
            mat[idx][c] += val

    # Reset U,V,W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # M1
    c = start_term()
    set_entry(U, 0, c, 1) # A00
    set_entry(U, 4, c, 1) # A11
    set_entry(V, 0, c, 1) # B00
    set_entry(V, 5, c, 1) # B11
    set_entry(W, 0, c, 1) # C00
    set_entry(W, 4, c, 1) # C11

    # M2: A11(B01-B00)
    c = start_term()
    set_entry(U, 4, c, 1) # A11
    set_entry(V, 1, c, 1) # B01
    set_entry(V, 0, c, -1) # -B00
    set_entry(W, 1, c, 1) # C01
    set_entry(W, 3, c, 1) # C10

    # M3: (A00+A01)B11
    c = start_term()
    set_entry(U, 0, c, 1) # A00
    set_entry(U, 1, c, 1) # A01
    set_entry(V, 5, c, 1) # B11
    set_entry(W, 1, c, 1) # C01
    set_entry(W, 4, c, -1) # -C11

    # M4: (A10-A00)(B00+B01)
    c = start_term()
    set_entry(U, 3, c, 1) # A10
    set_entry(U, 0, c, -1) # -A00
    set_entry(V, 0, c, 1) # B00
    set_entry(V, 1, c, 1) # B01
    set_entry(W, 0, c, 1) # C00
    set_entry(W, 4, c, -1) # -C11

    # M5: (A01-A11)(B10+B11)
    c = start_term()
    set_entry(U, 1, c, 1) # A01
    set_entry(U, 4, c, -1) # -A11
    set_entry(V, 4, c, 1) # B10
    set_entry(V, 5, c, 1) # B11
    set_entry(W, 1, c, 1) # C01
    set_entry(W, 3, c, -1) # -C10

    # M6: (A00+A10)B00
    c = start_term()
    set_entry(U, 0, c, 1) # A00
    set_entry(U, 3, c, 1) # A10
    set_entry(V, 0, c, 1) # B00
    set_entry(W, 1, c, -1) # -C01
    set_entry(W, 3, c, 1) # C10

    # M7: (A10+A11)(B11-B10)
    c = start_term()
    set_entry(U, 3, c, 1) # A10
    set_entry(U, 4, c, 1) # A11
    set_entry(V, 5, c, 1) # B11
    set_entry(V, 4, c, -1) # -B10
    set_entry(W, 0, c, 1) # C00
    set_entry(W, 3, c, 1) # C10

    # 2. Schoolbook for C[2, 0:4]
    # C[2, k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k]
    # A[2,0] is U[2*3+0]=U[6]
    # A[2,1] is U[7]
    # A[2,2] is U[8]
    # B[j,k] is V[j*4+k]
    # C[2,k] is W[k*3+2]
    
    for j in range(3):
        for k in range(4):
            c = start_term()
            set_entry(U, 6+j, c, 1)
            set_entry(V, j*4+k, c, 1)
            set_entry(W, k*3+2, c, 1)

    # 3. Schoolbook for C[0:2, 2:4]
    # C[i, k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] for i in 0,1 and k in 2,3
    for i in range(2):
        for j in range(3):
            for k in range(2, 4):
                c = start_term()
                set_entry(U, i*3+j, c, 1)
                set_entry(V, j*4+k, c, 1)
                set_entry(W, k*3+i, c, 1)

    return {"U": U, "V": V, "W": W}
