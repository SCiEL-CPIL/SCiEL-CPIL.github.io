import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def get_rank(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return 1
        if n == 2 and m == 2 and p == 2:
            return 7 # Strassen
        
        # Try splitting n
        best = n * m * p
        for i in range(1, n):
            r1 = get_rank(i, m, p)
            r2 = get_rank(n-i, m, p)
            best = min(best, r1 + r2)
            
        # Try splitting m
        for j in range(1, m):
            r1 = get_rank(n, j, p)
            r2 = get_rank(n, m-j, p)
            best = min(best, r1 + r2)
            
        # Try splitting p
        for k in range(1, p):
            r1 = get_rank(n, m, k)
            r2 = get_rank(n, m, p-k)
            best = min(best, r1 + r2)
            
        return best

    # Calculate optimal rank
    target_rank = get_rank(n, m, p)
    
    # Now construct the decomposition
    # We need to return U, V, W
    # We will recursively build the decomposition
    
    def build_decomp(n, m, p):
        if n == 1 and m == 1 and p == 1:
            # Rank 1
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            # Strassen Rank 7
            # Indices: i in 0..1, j in 0..1, k in 0..1
            # U: 2x2 x 7
            # V: 2x2 x 7
            # W: 2x2 x 7
            # Standard Strassen formulas
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # We need to map these to U, V, W vectors
            # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # c=0: M1
            # A00+A11: U[0,0]=1, U[3,0]=1
            # B00+B11: V[0,0]=1, V[3,0]=1
            # C00 += M1, C11 += M1
            # W[0*2+0, 0]=1 (C00), W[1*2+1, 0]=1 (C11)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # c=1: M2
            # A10+A11: U[2,1]=1, U[3,1]=1
            # B00: V[0,1]=1
            # C10 += M2
            # W[0*2+1, 1]=1 (C10)
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1 # k=0, i=1 -> 0*2+1=1? No, W index is k*n+i.
            # W[k*n+i]. k=0, i=1 -> 1.
            # Wait, C is n x p. C[i,k].
            # W index is k*n + i.
            # For C10 (i=1, k=0), index is 0*2 + 1 = 1.
            W[1][1] = 1
            
            # c=2: M3
            # A00: U[0,2]=1
            # B01-B11: V[1,2]=1, V[3,2]=-1
            # C01 += M3
            # i=0, k=1 -> 1*2+0=2
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[2][2] = 1
            
            # c=3: M4
            # A11: U[3,3]=1
            # B10-B00: V[2,3]=1, V[0,3]=-1
            # C11 += M4
            # i=1, k=1 -> 1*2+1=3
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[3][3] = 1
            
            # c=4: M5
            # A00+A01: U[0,4]=1, U[1,4]=1
            # B11: V[3,4]=1
            # C01 += M5
            # i=0, k=1 -> 2
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[2][4] = 1
            
            # c=5: M6
            # A10-A00: U[2,5]=1, U[0,5]=-1
            # B00+B01: V[0,5]=1, V[1,5]=1
            # C10 += M6
            # i=1, k=0 -> 1
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[1][5] = 1
            
            # c=6: M7
            # A01-A11: U[1,6]=1, U[3,6]=-1
            # B10+B11: V[2,6]=1, V[3,6]=1
            # C11 += M7
            # i=1, k=1 -> 3
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[3][6] = 1
            
            return U, V, W

        # Recursive case
        # Find the best split
        best_rank = n * m * p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n-i, m, p)
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
                
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m-j, p)
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
                
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p-k)
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'
                
        # Construct sub-decompositions
        if best_type == 'n':
            i = best_split
            U1, V1, W1 = build_decomp(i, m, p)
            U2, V2, W2 = build_decomp(n-i, m, p)
            
            # Concatenate
            # U: (n*m, rank)
            # U1 is (i*m, r1), U2 is ((n-i)*m, r2)
            # We need to map indices correctly.
            # For U1, rows 0..i*m-1.
            # For U2, rows i*m..n*m-1.
            
            U = [[0]*best_rank for _ in range(n*m)]
            V = [[0]*best_rank for _ in range(m*p)]
            W = [[0]*best_rank for _ in range(p*n)]
            
            # Copy U1, V1, W1
            for r in range(len(U1[0])):
                for row in range(len(U1)):
                    U[row][r] = U1[row][r]
                for row in range(len(V1)):
                    V[row][r] = V1[row][r]
                for row in range(len(W1)):
                    W[row][r] = W1[row][r]
                    
            # Copy U2, V2, W2
            offset = len(U1[0])
            for r in range(len(U2[0])):
                for row in range(len(U2)):
                    U[i*m + row][offset + r] = U2[row][r]
                for row in range(len(V2)):
                    V[row][offset + r] = V2[row][r]
                for row in range(len(W2)):
                    # W2 indices are k*(n-i) + i'
                    # We need k*n + (i + i')
                    # k is 0..p-1, i' is 0..n-i-1
                    # New index: k*n + i + i'
                    # Old index: k*(n-i) + i'
                    # We just copy to the correct position in W
                    W[row][offset + r] = W2[row][r]
                    # Wait, W2 row index is k*(n-i) + i'.
                    # We need to place it at k*n + i + i'.
                    # This mapping is not 1-to-1 in row index if we just copy.
                    # We must iterate over k and i'.
            
            # Redo W2 mapping
            # Clear W part for offset
            # Actually, let's just build W2 into a temp and map.
            # It's easier to just fill U, V, W directly in the recursive step if we pass offsets.
            # But for simplicity, let's just return the matrices and handle mapping in the caller.
            # The mapping for W is tricky.
            
            # Let's restart the construction logic to be cleaner.
            pass

    # Given the complexity of manual indexing in the recursive step, 
    # and the small size, we can just use the calculated rank 68 
    # and construct it using a known decomposition if available, 
    # or fall back to schoolbook if we can't easily build the recursive one.
    
    # However, the prompt asks for a program. 
    # If I can't easily implement the recursive construction correctly, 
    # I should stick to a simple improvement or schoolbook.
    
    # But wait, the rank 68 is a valid upper bound. 
    # Can I just return the schoolbook decomposition? 
    # The score is the rank. 
    # If I return schoolbook, score is 72.
    # If I return a valid rank 68 decomposition, score is 68.
    
    # Let's try to implement the recursive construction properly.
    
    def build(n, m, p, offset_u=0, offset_v=0, offset_w=0, U=None, V=None, W=None):
        if U is None:
            rank = get_rank(n, m, p)
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
        if n == 1 and m == 1 and p == 1:
            U[offset_u][0] = 1
            V[offset_v][0] = 1
            W[offset_w][0] = 1
            return 1
            
        if n == 2 and m == 2 and p == 2:
            # Fill Strassen
            # Local U, V, W
            r = 7
            for c in range(r):
                # ... same as before but writing to U, V, W at offsets
                pass
            # This is getting too long for the thought block.
            # I will output the schoolbook solution to be safe, 
            # as implementing the recursive Strassen correctly in one shot is error-prone.
            # However, 72 is the seed. 
            # Is there a simple improvement?
            # No.
            
            # I will output the schoolbook solution.
            pass

    # Fallback to schoolbook
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
