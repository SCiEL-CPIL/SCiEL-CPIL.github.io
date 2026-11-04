"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8.
    
    Strategy: Use Kronecker product of Strassen 2x2x2 (rank 7) with a rank-2
    decomposition for the rectangular 2x1x2 slice, yielding total rank 28.
    
    The 4x4x8 multiplication can be viewed as:
    - Split the 4x4 input A into four 2x2 blocks
    - Split the 4x8 input B into two 2x4 blocks (vertically) and then each into 2x2
    - Actually, let's think more carefully.
    
    For 4x4 * 4x8:
    - A is 4x4, B is 4x8
    - Result C is 4x8
    
    We can decompose this as:
    - View A as a 2x2 block matrix of 2x2 blocks: A = [[A11, A12], [A21, A22]]
    - View B as a 2x2 block matrix where each block is 2x4: B = [[B11, B12], [B21, B22]]
      but wait, B is 4x8, so if we split rows into 2 groups of 2, we get 2x4 blocks.
    
    Actually, let's use a cleaner approach:
    - 4x4 * 4x8 = (2x2 x 2x2) * (2x2 x 2x4)
    - This can be seen as a Kronecker product structure
    
    Let me think of it as:
    - A ⊗ structure for 4x4
    - B ⊗ structure for 4x8
    
    The tensor for 4x4*4x8 has indices (i,j,k) where i in [0,4), j in [0,4), k in [0,8).
    
    We can write i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2 where i1,j1 in [0,2), i2,j2 in [0,2), k1 in [0,4), k2 in [0,2).
    
    Wait, k goes up to 8, so k = 4*k1 + k2 where k1 in [0,2), k2 in [0,4)? No, that doesn't work cleanly.
    
    Let me reconsider. k in [0,8). We can write k = 2*k1 + k2 where k1 in [0,4), k2 in [0,2).
    
    So the tensor M[i,j,k] = delta (the multiplication tensor) has:
    M[2*i1+i2, 2*j1+j2, 2*k1+k2]
    
    For matrix multiplication, M[i,j,k] = 1 if the output element (i,k) depends on input elements (i,j) and (j,k).
    
    Actually, the multiplication tensor T[i,j,k] for (n x m) * (m x p) -> (n x p) is defined as:
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T is a 3-way tensor with T[i,j,k] = 1 for all valid i,j,k in the multiplication.
    
    For our case: n=4, m=4, p=8.
    T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [0,8).
    
    Now, using the decomposition i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2:
    - i1 in [0,2), i2 in [0,2)
    - j1 in [0,2), j2 in [0,2)  
    - k1 in [0,4), k2 in [0,2)
    
    The tensor T[2*i1+i2, 2*j1+j2, 2*k1+k2] = 1
    
    This can be factored as a Kronecker product of:
    - A 2x2x2 tensor (for i1,j1,k2 part?) 
    - An 2x2x4 tensor (for i2,j2,k1 part?)
    
    Hmm, this is getting complicated. Let me use a different approach.
    
    Actually, the standard way to get a Kronecker product decomposition:
    
    If we have a decomposition for (n1 x m1) * (m1 x p1) of rank r1,
    and a decomposition for (n2 x m2) * (m2 x p2) of rank r2,
    then we get a decomposition for (n1*n2 x m1*m2) * (m1*m2 x p1*p2) of rank r1*r2.
    
    For 4x4 * 4x8:
    - 4 = 2*2, so n1=2, n2=2, m1=2, m2=2, p1=2, p2=4? No, p=8=2*4.
    
    So we need:
    - A 2x2 * 2x2 decomposition (Strassen, rank 7)
    - A 2x2 * 2x4 decomposition
    
    For 2x2 * 2x4:
    - n=2, m=2, p=4
    - We can view this as 2x2 * (2x2 x 2x2) horizontally
    - This is equivalent to two independent 2x2*2x2 multiplications
    - So rank = 7 * 2 = 14? No wait...
    
    Actually, 2x2 * 2x4:
    - A is 2x2, B is 2x4
    - C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,2), k in [0,4)
    - This is like doing two 2x2*2x2 multiplications side by side
    - So the rank is 2 * 7 = 14
    
    Therefore, 4x4 * 4x8 has rank at most 7 * 14 = 98.
    
    But wait, that's the same as what was measured before (rank 98).
    
    The directive mentions a rank-2 decomposition for 2x1x2. Let me reconsider.
    
    Perhaps the idea is:
    - 4x4 * 4x8 can be decomposed differently.
    
    Let me think of 4x8 as 4 * (1x8) repeated 4 times? No...
    
    Actually, let me try: 4x4 * 4x8 = (4x4 * 4x4) concatenated with (4x4 * 4x4)
    So it's two independent 4x4*4x4 multiplications.
    Rank = 2 * rank(4x4*4x4) = 2 * 49 = 98 (using Strassen Kronecker).
    
    That gives 98, same as before.
    
    The directive says "rank-2 decomposition for the 2x1x2 rectangular slice, which yields a total rank of 28".
    
    28 = 7 * 4. So maybe they're using a rank-4 decomposition for some slice?
    
    Or 28 = 4 * 7. 
    
    Let me reconsider the factorization. If we write:
    4x4 * 4x8 as a Kronecker product of:
    - 2x2 * 2x2 (rank 7, Strassen)
    - 2x2 * 2x4 (rank ?)
    
    If 2x2 * 2x4 has rank 4, then total rank = 7 * 4 = 28.
    
    Can 2x2 * 2x4 have rank 4?
    - n=2, m=2, p=4
    - Number of multiplications in schoolbook: 2*2*4 = 16
    - Lower bound: max(n*p, m*p, n*m) = max(8, 8, 4) = 8
    - So rank 4 is below the lower bound, which is impossible.
    
    So rank 28 is not achievable via that route. The directive may be incorrect or I'm misunderstanding.
    
    Let me just go with the known best: Kronecker product of Strassen for 4x4*4x4 (rank 49) replicated for the two 4x4 halves of the 4x8 matrix, giving rank 98.
    
    Actually wait - let me re-read: "Tensor product of Strassen 2x2x2 gives exact 4x4x4 rank-49 decomposition; replicating for 8-wide output yields rank 98, beating schoolbook 128"
    
    So the current best known is 98. Let me implement that.
    
    For 4x4 * 4x4 using Kronecker of Strassen:
    - Strassen for 2x2*2x2 has rank 7
    - 4x4*4x4 = Kronecker product of two 2x2*2x2, rank = 7*7 = 49
    
    For 4x4 * 4x8:
    - Split the 8 columns into two groups of 4
    - Each group gives a 4x4*4x4 multiplication
    - Total rank = 49 + 49 = 98
    
    Let me implement the Strassen decomposition for 2x2*2x2 first.
    
    Strassen's algorithm for 2x2*2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    The factor matrices for Strassen:
    U (2x2 x 7): For each of the 7 products, what's the coefficient of A's elements
    V (2x2 x 7): For each of the 7 products, what's the coefficient of B's elements
    W (2x2 x 7): For each of the 7 products, what's the contribution to C's elements
    
    U[i,j, t] where i,j in [0,2), t in [0,7):
    t=0: (a+d) -> U[0,0]=1, U[1,1]=1
    t=1: (c+d) -> U[1,0]=1, U[1,1]=1
    t=2: a -> U[0,0]=1
    t=3: d -> U[1,1]=1
    t=4: (a+b) -> U[0,0]=1, U[0,1]=1
    t=5: (c-a) -> U[1,0]=1, U[0,0]=-1
    t=6: (b-d) -> U[0,1]=1, U[1,1]=-1
    
    V[i,j, t] where i,j in [0,2), t in [0,7):
    t=0: (e+h) -> V[0,0]=1, V[1,1]=1
    t=1: e -> V[0,0]=1
    t=2: (f-h) -> V[0,1]=1, V[1,1]=-1
    t=3: (g-e) -> V[1,0]=1, V[0,0]=-1
    t=4: h -> V[1,1]=1
    t=5: (e+f) -> V[0,0]=1, V[0,1]=1
    t=6: (g+h) -> V[1,0]=1, V[1,1]=1
    
    W[i,j, t] where i,j in [0,2), t in [0,7):
    C[0,0] = m1+m4-m5+m7 -> W[0,0,0]=1, W[0,0,3]=1, W[0,0,4]=-1, W[0,0,6]=1
    C[0,1] = m3+m5 -> W[0,1,2]=1, W[0,1,4]=1
    C[1,0] = m2+m4 -> W[1,0,1]=1, W[1,0,3]=1
    C[1,1] = m1-m2+m3+m6 -> W[1,1,0]=1, W[1,1,1]=-1, W[1,1,2]=1, W[1,1,5]=1
    
    Now for the Kronecker product:
    If T1 has factor matrices (U1, V1, W1) of shapes (n1*m1, r1), (m1*p1, r1), (p1*n1, r1)
    and T2 has factor matrices (U2, V2, W2) of shapes (n2*m2, r2), (m2*p2, r2), (p2*n2, r2)
    
    Then the Kronecker product T1 ⊗ T2 has:
    - n = n1*n2, m = m1*m2, p = p1*p2
    - rank = r1*r2
    
    U[i*m+j, t1*r2+t2] = U1[i1*m1+j1, t1] * U2[i2*m2+j2, t2]
    where i = i1*n2 + i2, j = j1*m2 + j2, t = t1*r2 + t2
    
    Similarly for V and W.
    
    For 4x4*4x4:
    T1 = T2 = Strassen 2x2*2x2, n1=m1=p1=2, r1=7
    n=m=p=4, r=49
    
    For 4x4*4x8:
    We can view this as two independent 4x4*4x4 multiplications.
    So we take the 4x4*4x4 decomposition (rank 49) and duplicate it for the second half of the 8 columns.
    
    Total rank = 98.
    
    Let me implement this.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition, rank 7."""
        n, m, p = 2, 2, 2
        r = 7
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        W = [[0.0] * r for _ in range(p * n)]
        
        # U: coefficients for A elements
        # t=0: a+d, t=1: c+d, t=2: a, t=3: d, t=4: a+b, t=5: c-a, t=6: b-d
        U[0][0] = 1.0; U[1][0] = 1.0      # a+d
        U[2][1] = 1.0; U[3][1] = 1.0      # c+d (indices: row 1, col 0 = index 2; row 1, col 1 = index 3)
        U[0][2] = 1.0                       # a
        U[3][3] = 1.0                       # d
        U[0][4] = 1.0; U[1][4] = 1.0      # a+b
        U[2][5] = 1.0; U[0][5] = -1.0     # c-a
        U[1][6] = 1.0; U[3][6] = -1.0     # b-d
        
        # V: coefficients for B elements
        # t=0: e+h, t=1: e, t=2: f-h, t=3: g-e, t=4: h, t=5: e+f, t=6: g+h
        V[0][0] = 1.0; V[3][0] = 1.0      # e+h
        V[0][1] = 1.0                       # e
        V[1][2] = 1.0; V[3][2] = -1.0     # f-h
        V[2][3] = 1.0; V[0][3] = -1.0     # g-e
        V[3][4] = 1.0                       # h
        V[0][5] = 1.0; V[1][5] = 1.0      # e+f
        V[2][6] = 1.0; V[3][6] = 1.0      # g+h
        
        # W: coefficients for C elements
        # C[0,0] = m1+m4-m5+m7, C[0,1] = m3+m5, C[1,0] = m2+m4, C[1,1] = m1-m2+m3+m6
        W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0; W[0][6] = 1.0   # C[0,0]
        W[1][2] = 1.0; W[1][4] = 1.0                                           # C[0,1]
        W[2][1] = 1.0; W[2][3] = 1.0                                           # C[1,0]
        W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][5] = 1.0  # C[1,1]
        
        return U, V, W
    
    def kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, r1, n2, m2, p2, r2):
        """Kronecker product of two decompositions."""
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        r = r1 * r2
        
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        W = [[0.0] * r for _ in range(p * n)]
        
        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        i = i1 * n2 + i2
                        j = j1 * m2 + j2
                        for t1 in range(r1):
                            for t2 in range(r2):
                                t = t1 * r2 + t2
                                U[i * m + j][t] = U1[i1 * m1 + j1][t1] * U2[i2 * m2 + j2][t2]
        
        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        j = j1 * m2 + j2
                        k = k1 * p2 + k2
                        for t1 in range(r1):
                            for t2 in range(r2):
                                t = t1 * r2 + t2
                                V[j * p + k][t] = V1[j1 * p1 + k1][t1] * V2[j2 * p2 + k2][t2]
        
        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        k = k1 * p2 + k2
                        i = i1 * n2 + i2
                        for t1 in range(r1):
                            for t2 in range(r2):
                                t = t1 * r2 + t2
                                W[k * n + i][t] = W1[k1 * n1 + i1][t1] * W2[k2 * n2 + i2][t2]
        
        return U, V, W
    
    # Get Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = strassen_2x2()
    
    # Kronecker product to get 4x4*4x4
    U_44, V_44, W_44 = kron_decompose(U_s, V_s, W_s, U_s, V_s, W_s, 2, 2, 2, 7, 2, 2, 2, 7)
    
    # For 4x4*4x8: split into two 4x4*4x4 multiplications
    # The 4x8 matrix B is [B_left | B_right] where each is 4x4
    # C = A * B = [A*B_left | A*B_right]
    # So we use the 4x4*4x4 decomposition twice
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # First half: columns 0-3 of B
    for i in range(n * m):
        for t in range(49):
            U[i][t] = U_44[i][t]
    
    for j in range(m):
        for k in range(4):  # first 4 columns
            for t in range(49):
                V[j * p + k][t] = V_44[j * 4 + k][t]
    
    for k in range(4):  # first 4 columns of C
        for i in range(n):
            for t in range(49):
                W[k * n + i][t] = W_44[k * n + i][t]
    
    # Second half: columns 4-7 of B
    for i in range(n * m):
        for t in range(49):
            U[i][t + 49] = U_44[i][t]
    
    for j in range(m):
        for k in range(4, 8):  # last 4 columns
            for t in range(49):
                V[j * p + k][t + 49] = V_44[j * 4 + (k - 4)][t]
    
    for k in range(4, 8):  # last 4 columns of C
        for i in range(n):
            for t in range(49):
                W[k * n + i][t + 49] = W_44[(k - 4) * n + i][t]
    
    # Convert to integers where possible for cleanliness
    for matrix in [U, V, W]:
        for row in matrix:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
