import numpy as np

DIMENSIONS = [2, 3, 5]

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    Indices:
    U: 4x7 (A is 2x2, flattened row-major: a00, a01, a10, a11)
    V: 4x7 (B is 2x2, flattened row-major: b00, b01, b10, b11)
    W: 4x7 (C is 2x2, flattened row-major: c00, c01, c10, c11)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11) -> c00+=M1, c01-=M1, c10+=M1, c11+=M1 ? 
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    U[0][0] = 1; U[3][0] = 1  # a00, a11
    V[0][0] = 1; V[3][0] = 1  # b00, b11
    W[0][0] = 1; W[1][0] = -1; W[2][0] = 1; W[3][0] = 1
    
    # Term 2: M2
    U[2][1] = 1; U[3][1] = 1  # a10, a11
    V[0][1] = 1               # b00
    W[2][1] = 1; W[3][1] = -1 # c10, c11
    
    # Term 3: M3
    U[0][2] = 1               # a00
    V[1][2] = 1; V[3][2] = -1 # b01, b11
    W[1][2] = 1; W[3][2] = 1  # c01, c11
    
    # Term 4: M4
    U[3][3] = 1               # a11
    V[2][3] = 1; V[0][3] = -1 # b10, b00
    W[0][3] = 1; W[2][3] = 1  # c00, c10
    
    # Term 5: M5
    U[0][4] = 1; U[1][4] = 1  # a00, a01
    V[3][4] = 1               # b11
    W[0][4] = -1; W[1][4] = 1 # c00, c01
    
    # Term 6: M6
    U[2][5] = 1; U[0][5] = -1 # a10, a00
    V[0][5] = 1; V[1][5] = 1  # b00, b01
    W[3][5] = 1               # c11
    
    # Term 7: M7
    U[1][6] = 1; U[3][6] = -1 # a01, a11
    V[2][6] = 1; V[3][6] = 1  # b10, b11
    W[0][6] = 1               # c00
    
    return U, V, W

def _standard_decomp(n, m, p):
    """
    Returns U, V, W for n x m x p with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][r] = 1
                V[j*p + k][r] = 1
                W[k*n + i][r] = 1
                r += 1
    return U, V, W

def _combine_decomps(decomp_list):
    """
    Combines a list of (U, V, W) decompositions into one.
    """
    if not decomp_list:
        return [], [], []
    
    U_total = []
    V_total = []
    W_total = []
    rank = 0
    for U, V, W in decomp_list:
        if not U: continue
        U_total.extend(U)
        V_total.extend(V)
        W_total.extend(W)
        rank += len(U)
    
    return U_total, V_total, W_total

def _decompose(n, m, p):
    """
    Recursively finds a low-rank decomposition for n x m x p matrix multiplication.
    """
    # Base case: Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    
    # Base case: Standard
    if n == 1 or m == 1 or p == 1:
        return _standard_decomp(n, m, p)
        
    best_rank = float('inf')
    best_decomp = None
    
    # Try splitting m
    if m > 1:
        for m1 in range(1, m):
            m2 = m - m1
            # Split A into A1 (n x m1) and A2 (n x m2)
            # Split B into B1 (m1 x p) and B2 (m2 x p)
            # C = A1 B1 + A2 B2
            d1 = _decompose(n, m1, p)
            d2 = _decompose(n, m2, p)
            
            # We need to map the indices correctly.
            # U1 is for A1, indices 0..m1-1 of A.
            # U2 is for A2, indices m1..m-1 of A.
            # V1 is for B1, indices 0..m1-1 of B rows.
            # V2 is for B2, indices m1..m-1 of B rows.
            # W is same.
            
            U1, V1, W1 = d1
            U2, V2, W2 = d2
            
            # Construct combined U
            U_comb = [[0]*(len(U1)+len(U2)) for _ in range(n*m)]
            # Fill U1 part (rows 0..m1-1 of A)
            for i in range(n):
                for j in range(m1):
                    U_comb[i*m + j] = U1[i*m1 + j]
            # Fill U2 part (rows m1..m-1 of A)
            for i in range(n):
                for j in range(m2):
                    U_comb[i*m + (m1+j)] = U2[i*m2 + j]
                    
            # Construct combined V
            V_comb = [[0]*(len(V1)+len(V2)) for _ in range(m*p)]
            # Fill V1 part (cols 0..m1-1 of B)
            for j in range(m1):
                for k in range(p):
                    V_comb[j*p + k] = V1[j*p + k]
            # Fill V2 part (cols m1..m-1 of B)
            for j in range(m2):
                for k in range(p):
                    V_comb[(m1+j)*p + k] = V2[j*p + k]
                    
            # Construct combined W
            W_comb = [[0]*(len(W1)+len(W2)) for _ in range(p*n)]
            for k in range(p):
                for i in range(n):
                    W_comb[k*n + i] = W1[k*n + i]
                    # Add W2
                    for r in range(len(W2)):
                        W_comb[k*n + i][len(W1) + r] = W2[k*n + i][r]
                        
            rank = len(U1) + len(U2)
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U_comb, V_comb, W_comb)
                
    # Try splitting p
    if p > 1:
        for p1 in range(1, p):
            p2 = p - p1
            # Split B into B1 (m x p1) and B2 (m x p2)
            # C = A B1 + A B2
            d1 = _decompose(n, m, p1)
            d2 = _decompose(n, m, p2)
            
            U1, V1, W1 = d1
            U2, V2, W2 = d2
            
            U_comb = [[0]*(len(U1)+len(U2)) for _ in range(n*m)]
            for i in range(n):
                for j in range(m):
                    U_comb[i*m + j] = U1[i*m + j]
                    for r in range(len(U2)):
                        U_comb[i*m + j][len(U1) + r] = U2[i*m + j][r]
                        
            V_comb = [[0]*(len(V1)+len(V2)) for _ in range(m*p)]
            for j in range(m):
                for k in range(p1):
                    V_comb[j*p + k] = V1[j*p1 + k]
                for k in range(p2):
                    V_comb[j*p + (p1+k)] = V2[j*p2 + k]
                    
            W_comb = [[0]*(len(W1)+len(W2)) for _ in range(p*n)]
            for k in range(p1):
                for i in range(n):
                    W_comb[k*n + i] = W1[k*n + i]
            for k in range(p2):
                for i in range(n):
                    W_comb[(p1+k)*n + i] = W2[k*n + i]
                    
            rank = len(U1) + len(U2)
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U_comb, V_comb, W_comb)
                
    # Try splitting n
    if n > 1:
        for n1 in range(1, n):
            n2 = n - n1
            # Split A into A1 (n1 x m) and A2 (n2 x m)
            # C = A1 B + A2 B
            d1 = _decompose(n1, m, p)
            d2 = _decompose(n2, m, p)
            
            U1, V1, W1 = d1
            U2, V2, W2 = d2
            
            U_comb = [[0]*(len(U1)+len(U2)) for _ in range(n*m)]
            for i in range(n1):
                for j in range(m):
                    U_comb[i*m + j] = U1[i*m + j]
            for i in range(n2):
                for j in range(m):
                    U_comb[(n1+i)*m + j] = U2[i*m + j]
                    
            V_comb = [[0]*(len(V1)+len(V2)) for _ in range(m*p)]
            for j in range(m):
                for k in range(p):
                    V_comb[j*p + k] = V1[j*p + k]
                    for r in range(len(V2)):
                        V_comb[j*p + k][len(V1) + r] = V2[j*p + k][r]
                        
            W_comb = [[0]*(len(W1)+len(W2)) for _ in range(p*n)]
            for k in range(p):
                for i in range(n1):
                    W_comb[k*n + i] = W1[k*n1 + i]
                for i in range(n2):
                    W_comb[k*n + (n1+i)] = W2[k*n2 + i]
                    
            rank = len(U1) + len(U2)
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U_comb, V_comb, W_comb)
                
    if best_decomp is None:
        # Fallback to standard
        return _standard_decomp(n, m, p)
        
    return best_decomp

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
