import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using a recursive block strategy.
    # The global U, V, W will be lists of lists.
    # We start with empty lists and append blocks.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, u_block, v_block, w_block):
        """
        Adds a block decomposition for M_{n_b, m_b, p_b} 
        located at offsets (i_off, j_off, k_off).
        u_block, v_block, w_block are local factor matrices.
        """
        # Pad u_block to size (n*m, rank_b)
        # Local U has shape (n_b*m_b, rank_b)
        # Global U has shape (n*m, rank_total)
        # We need to expand local U to global size.
        
        rank_b = len(u_block[0]) if u_block else 0
        if rank_b == 0:
            return

        # Create global U block
        # u_block[i_local * m_b + j_local, c]
        # Global index: (i_local + i_off) * m + (j_local + j_off)
        
        # We store U as list of rows.
        # We will append to a global accumulator later, or build the full matrix now.
        # Since we don't know total rank, we store blocks.
        
        # For each row in global U (n*m), we need to fill the columns corresponding to this block.
        # It is easier to build the full U, V, W matrices at the end.
        # Let's store the blocks with their offsets.
        
        pass # This logic is getting complex to manage offsets in lists.
        # Better approach: Build the full U, V, W matrices directly by iterating through the decomposition steps.
        
    # Given the complexity of manual offset management in a recursive structure,
    # and the risk of errors, let's stick to the schoolbook if we can't easily verify the block logic.
    # However, 96 is a significant improvement.
    
    # Let's try a simpler improvement:
    # Is there a known rank for 4x5x6?
    # If we just use schoolbook, rank is 120.
    
    # Let's implement the 96 rank decomposition carefully.
    
    # We will define a helper to generate the U, V, W for a specific tensor M_{n,m,p}
    # using the specified strategy.
    
    def get_decomp(n, m, p):
        # Returns U, V, W for M_{n,m,p}
        # Strategy:
        # If n==2 and m==2 and p==2: Strassen
        # If n==2 and m==2 and p==1: Schoolbook (rank 2)
        # If n==2 and m==1 and p==6: Schoolbook (rank 12)
        # Otherwise: Split and combine.
        
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        if n == 2 and m == 2 and p == 1:
            return schoolbook_2x2x1()
            
        if n == 2 and m == 1 and p == 6:
            return schoolbook_2x1x6()
            
        # Split m into m1, m2
        if m > 1:
            m1 = m // 2
            m2 = m - m1
            # M_{n, m1, p} and M_{n, m2, p}
            # Note: j index is split.
            # The blocks are independent in j.
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            
            # Combine
            # U1 has shape (n*m1, r1), U2 has shape (n*m2, r2)
            # We need to map to (n*m, r1+r2)
            
            U = [[0]*(len(U1[0])+len(U2[0])) for _ in range(n*m)]
            V = [[0]*(len(V1[0])+len(V2[0])) for _ in range(m*p)]
            W = [[0]*(len(W1[0])+len(W2[0])) for _ in range(p*n)]
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            
            # Fill U
            for i in range(n):
                for j in range(m1):
                    # Local index in U1: i*m1 + j
                    # Global index: i*m + j
                    for c in range(r1):
                        U[i*m + j][c] = U1[i*m1 + j][c]
                        
                for j in range(m2):
                    # Local index in U2: i*m2 + j
                    # Global index: i*m + (m1 + j)
                    for c in range(r2):
                        U[i*m + (m1 + j)][r1 + c] = U2[i*m2 + j][c]
            
            # Fill V
            # V1 shape (m1*p, r1), V2 shape (m2*p, r2)
            for j in range(m1):
                for k in range(p):
                    for c in range(r1):
                        V[j*p + k][c] = V1[j*p + k][c]
                        
            for j in range(m2):
                for k in range(p):
                    for c in range(r2):
                        V[(m1 + j)*p + k][r1 + c] = V2[j*p + k][c]
            
            # Fill W
            # W1 shape (p*n, r1), W2 shape (p*n, r2)
            for k in range(p):
                for i in range(n):
                    for c in range(r1):
                        W[k*n + i][c] = W1[k*n + i][c]
                    for c in range(r2):
                        W[k*n + i][r1 + c] = W2[k*n + i][c]
                        
            return U, V, W
        
        # Split n into n1, n2
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            
            U = [[0]*(len(U1[0])+len(U2[0])) for _ in range(n*m)]
            V = [[0]*(len(V1[0])+len(V2[0])) for _ in range(m*p)]
            W = [[0]*(len(W1[0])+len(W2[0])) for _ in range(p*n)]
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            
            # Fill U
            for i in range(n1):
                for j in range(m):
                    for c in range(r1):
                        U[i*m + j][c] = U1[i*m + j][c]
                        
            for i in range(n2):
                for j in range(m):
                    for c in range(r2):
                        U[(n1 + i)*m + j][r1 + c] = U2[i*m + j][c]
            
            # Fill V
            # V1 shape (m*p, r1), V2 shape (m*p, r2)
            for j in range(m):
                for k in range(p):
                    for c in range(r1):
                        V[j*p + k][c] = V1[j*p + k][c]
                    for c in range(r2):
                        V[j*p + k][r1 + c] = V2[j*p + k][c]
            
            # Fill W
            for k in range(p):
                for i in range(n1):
                    for c in range(r1):
                        W[k*n + i][c] = W1[k*n1 + i][c]
                        
                for i in range(n2):
                    for c in range(r2):
                        W[k*n + (n1 + i)][r1 + c] = W2[k*n2 + i][c]
                        
            return U, V, W
            
        # Split p into p1, p2
        if p > 1:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            
            U = [[0]*(len(U1[0])+len(U2[0])) for _ in range(n*m)]
            V = [[0]*(len(V1[0])+len(V2[0])) for _ in range(m*p)]
            W = [[0]*(len(W1[0])+len(W2[0])) for _ in range(p*n)]
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    for c in range(r1):
                        U[i*m + j][c] = U1[i*m + j][c]
                    for c in range(r2):
                        U[i*m + j][r1 + c] = U2[i*m + j][c]
            
            # Fill V
            for j in range(m):
                for k in range(p1):
                    for c in range(r1):
                        V[j*p + k][c] = V1[j*p1 + k][c]
                        
                for k in range(p2):
                    for c in range(r2):
                        V[j*p + (p1 + k)][r1 + c] = V2[j*p2 + k][c]
            
            # Fill W
            for k in range(p1):
                for i in range(n):
                    for c in range(r1):
                        W[k*n + i][c] = W1[k*n + i][c]
                        
            for k in range(p2):
                for i in range(n):
                    for c in range(r2):
                        W[(p1 + k)*n + i][r1 + c] = W2[k*n + i][c]
                        
            return U, V, W
            
        # Base case: Schoolbook for n x m x p
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][c] = 1
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
                    c += 1
        return U, V, W

    def strassen_2x2x2():
        # Strassen decomposition for 2x2x2
        # U, V, W are 4x7, 4x7, 4x7
        # Indices:
        # A: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        # B: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        # C: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # W coefficients for C
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U (A factors)
        # M1: A00 + A11 -> U[0,0]=1, U[3,0]=1
        U[0][0] = 1; U[3][0] = 1
        # M2: A10 + A11 -> U[2,1]=1, U[3,1]=1
        U[2][1] = 1; U[3][1] = 1
        # M3: A00 -> U[0,2]=1
        U[0][2] = 1
        # M4: A11 -> U[3,3]=1
        U[3][3] = 1
        # M5: A00 + A01 -> U[0,4]=1, U[1,4]=1
        U[0][4] = 1; U[1][4] = 1
        # M6: A10 - A00 -> U[2,5]=1, U[0,5]=-1
        U[2][5] = 1; U[0][5] = -1
        # M7: A01 - A11 -> U[1,6]=1, U[3,6]=-1
        U[1][6] = 1; U[3][6] = -1
        
        # V (B factors)
        # M1: B00 + B11 -> V[0,0]=1, V[3,0]=1
        V[0][0] = 1; V[3][0] = 1
        # M2: B00 -> V[0,1]=1
        V[0][1] = 1
        # M3: B10 - B11 -> V[2,2]=1, V[3,2]=-1
        V[2][2] = 1; V[3][2] = -1
        # M4: B01 - B00 -> V[1,3]=1, V[0,3]=-1
        V[1][3] = 1; V[0][3] = -1
        # M5: B11 -> V[3,4]=1
        V[3][4] = 1
        # M6: B00 + B01 -> V[0,5]=1, V[1,5]=1
        V[0][5] = 1; V[1][5] = 1
        # M7: B10 + B11 -> V[2,6]=1, V[3,6]=1
        V[2][6] = 1; V[3][6] = 1
        
        # W (C coefficients)
        # C00: M1, M4, -M5, M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        # C01: M3, M5 -> W[1,2]=1, W[1,4]=1
        W[1][2] = 1; W[1][4] = 1
        # C10: M2, M4 -> W[2,1]=1, W[2,3]=1
        W[2][1] = 1; W[2][3] = 1
        # C11: M1, -M2, M3, M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        # n=2, m=2, p=1
        # Rank 4? No, rank 2?
        # C00 = A00 B00 + A01 B01
        # C10 = A10 B00 + A11 B01
        # Schoolbook rank is 4.
        # Can we do better?
        # M_{2,2,1} is just two dot products.
        # Rank is 4.
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        
        c = 0
        for i in range(2):
            for j in range(2):
                U[i*2 + j][c] = 1
                V[j*1 + 0][c] = 1
                W[0*2 + i][c] = 1
                c += 1
        return U, V, W

    def schoolbook_2x1x6():
        # n=2, m=1, p=6
        # Rank 12
        rank = 12
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(12)]
        
        c = 0
        for i in range(2):
            for j in range(1):
                for k in range(6):
                    U[i*1 + j][c] = 1
                    V[j*6 + k][c] = 1
                    W[k*2 + i][c] = 1
                    c += 1
        return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
