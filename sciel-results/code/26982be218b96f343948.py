"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by splitting the problem into blocks.
    # Block 1: Multiplying 2x3 by 3x4 (columns 0-3 of B)
    # Block 2: Multiplying 2x3 by 3x3 (columns 4-6 of B)
    
    # Helper to create a Strassen decomposition for 2x2 by 2x2
    def strassen_2x2():
        # Returns U, V, W factors for 2x2 * 2x2 -> 2x2
        # Rank 7
        # Standard Strassen formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)] # 2x2 A
        V = [[0]*rank for _ in range(4)] # 2x2 B
        W = [[0]*rank for _ in range(4)] # 2x2 C
        
        # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Helper to create schoolbook decomposition for n x m by m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Function to combine two decompositions for disjoint output blocks
    # decomp1: (U1, V1, W1) for n x m1 by m1 x p1
    # decomp2: (U2, V2, W2) for n x m2 by m2 x p2
    # This assumes we are splitting the middle dimension m.
    # U1 uses A indices 0..m1-1, U2 uses A indices m1..m-1?
    # Actually, if we split m, A is n x m.
    # A1 is n x m1, A2 is n x m2.
    # B1 is m1 x p1, B2 is m2 x p2.
    # C1 is n x p1, C2 is n x p2.
    # We need to map these to the global indices.
    
    def combine_split_m(decomp1, decomp2, n, m1, m2, p1, p2):
        U1, V1, W1 = decomp1
        U2, V2, W2 = decomp2
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        
        U = [[0]*total_rank for _ in range(n*(m1+m2))]
        V = [[0]*total_rank for _ in range((m1+m2)*(p1+p2))]
        W = [[0]*total_rank for _ in range((p1+p2)*n)]
        
        # Fill U
        for i in range(n):
            for j in range(m1):
                U[i*(m1+m2)+j][:r1] = U1[i*m1+j]
            for j in range(m2):
                U[i*(m1+m2)+m1+j][r1:] = U2[i*m2+j]
                
        # Fill V
        for j in range(m1):
            for k in range(p1):
                V[j*(p1+p2)+k][:r1] = V1[j*p1+k]
        for j in range(m2):
            for k in range(p2):
                V[(m1+j)*(p1+p2)+k][r1:] = V2[j*p2+k]
                
        # Fill W
        for k in range(p1):
            for i in range(n):
                W[k*n+i][:r1] = W1[k*n+i]
        for k in range(p2):
            for i in range(n):
                W[(p1+k)*n+i][r1:] = W2[k*n+i]
                
        return U, V, W

    # Function to combine two decompositions for disjoint output columns (split p)
    # decomp1: n x m by m x p1
    # decomp2: n x m by m x p2
    
    def combine_split_p(decomp1, decomp2, n, m, p1, p2):
        U1, V1, W1 = decomp1
        U2, V2, W2 = decomp2
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*(p1+p2))]
        W = [[0]*total_rank for _ in range((p1+p2)*n)]
        
        # U is the same for both (since A is same)
        # Wait, U1 and U2 should be identical if A is the same.
        # In our recursive construction, if we split p, A is unchanged.
        # So U1 == U2. We can just use U1.
        for i in range(n*m):
            U[i][:r1] = U1[i]
            U[i][r1:] = U2[i] # Should be same values
            
        # V
        for j in range(m):
            for k in range(p1):
                V[j*(p1+p2)+k][:r1] = V1[j*p1+k]
            for k in range(p2):
                V[j*(p1+p2)+p1+k][r1:] = V2[j*p2+k]
                
        # W
        for k in range(p1):
            for i in range(n):
                W[k*n+i][:r1] = W1[k*n+i]
        for k in range(p2):
            for i in range(n):
                W[(p1+k)*n+i][r1:] = W2[k*n+i]
                
        return U, V, W

    # Base cases and recursive builders
    
    def build_2x2x2():
        return strassen_2x2()

    def build_2x2x1():
        return schoolbook(2, 2, 1)

    def build_2x1xp(p):
        return schoolbook(2, 1, p)

    def build_2x2xp(p):
        if p == 2:
            return build_2x2x2()
        if p == 1:
            return build_2x2x1()
        # Split p into 2 and p-2
        if p >= 4:
            d1 = build_2x2xp(2)
            d2 = build_2x2xp(p-2)
            return combine_split_p(d1, d2, 2, 2, 2, p-2)
        else:
            return schoolbook(2, 2, p)

    def build_2x3xp(p):
        # Split m=3 into 2 and 1
        # Part 1: 2x2 by 2xp
        # Part 2: 2x1 by 1xp
        d1 = build_2x2xp(p)
        d2 = build_2x1xp(p)
        return combine_split_m(d1, d2, 2, 2, 1, p, p)

    # Main construction
    # Split p=7 into 4 and 3
    d1 = build_2x3xp(4)
    d2 = build_2x3xp(3)
    
    # Combine by splitting p
    U, V, W = combine_split_p(d1, d2, 2, 3, 4, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
