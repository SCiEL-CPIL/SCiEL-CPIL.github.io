"""Program using Kronecker product structure for 3x5*5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with Strassen rank 7."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products
    U[:, 0] = [1, 0, 0, 1]  # (a11+a22)
    V[:, 0] = [1, 0, 0, 1]  # (b11+b22)
    W[:, 0] = [1, 0, 0, 1]  # c11, c22
    
    U[:, 1] = [0, 0, 1, 1]  # (a21+a22)
    V[:, 1] = [1, 0, 0, 0]  # b11
    W[:, 1] = [0, 0, 1, 1]  # c21, c22
    
    U[:, 2] = [1, 0, 0, 0]  # a11
    V[:, 2] = [0, 1, 0, -1] # (b12-b22)
    W[:, 2] = [0, 1, 0, -1] # c12, -c22
    
    U[:, 3] = [0, 0, 0, 1]  # a22
    V[:, 3] = [-1, 0, 1, 0] # (b21-b11)
    W[:, 3] = [-1, 0, 0, 1] # -c11, c22
    
    U[:, 4] = [1, 1, 0, 0]  # (a11+a12)
    V[:, 4] = [0, 0, 0, 1]  # b22
    W[:, 4] = [1, 0, 0, 1]  # c11, c22
    
    U[:, 5] = [-1, 0, 1, 0] # (a21-a11)
    V[:, 5] = [1, 1, 0, 0]  # (b11+b12)
    W[:, 5] = [0, 0, 1, 1]  # c21, c22
    
    U[:, 6] = [0, 1, 0, -1] # (a12-a22)
    V[:, 6] = [0, 0, 1, 1]  # (b21+b22)
    W[:, 6] = [0, 1, 0, -1] # c12, -c22
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Combine two decompositions via Kronecker product.
    If (U1,V1,W1) decomposes n1xm1*m1xp1 with rank r1,
    and (U2,V2,W2) decomposes n2xm2*m2xp2 with rank r2,
    then the Kronecker product decomposes (n1*n2)x(m1*m2)*(m1*m2)x(p1*p2) with rank r1*r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U: row index is (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
            # U1 row: i1*m1+j1, U2 row: i2*m2+j2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u1_val = U1[i1*m1 + j1, c1]
                            u2_val = U2[i2*m2 + j2, c2]
                            if u1_val != 0 and u2_val != 0:
                                full_row = (i1*n2 + i2) * m + (j1*m2 + j2)
                                U[full_row, c] = u1_val * u2_val
            
            # V: row index is (j1*m2+j2)*p + (k1*p2+k2)
            # V1 row: j1*p1+k1, V2 row: j2*p2+k2
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v1_val = V1[j1*p1 + k1, c1]
                            v2_val = V2[j2*p2 + k2, c2]
                            if v1_val != 0 and v2_val != 0:
                                full_row = (j1*m2 + j2) * p + (k1*p2 + k2)
                                V[full_row, c] = v1_val * v2_val
            
            # W: row index is (k1*p2+k2)*n + (i1*n2+i2)
            # W1 row: k1*n1+i1, W2 row: k2*n2+i2
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w1_val = W1[k1*n1 + i1, c1]
                            w2_val = W2[k2*n2 + i2, c2]
                            if w1_val != 0 and w2_val != 0:
                                full_row = (k1*p2 + k2) * n + (i1*n2 + i2)
                                W[full_row, c] = w1_val * w2_val
    
    return U, V, W

def _decompose_3x5x5x6():
    """Decompose 3x5 * 5x6 using Kronecker structure.
    
    Strategy: 3x5*5x6 can be seen as combining smaller blocks.
    We'll use the fact that:
    - 3 = 1+2, 5 = 2+3, 6 = 2+4
    
    We can decompose this into block multiplications and use
    optimal decompositions for each block.
    
    Alternative: Use Kronecker products where possible.
    3x5*5x6 doesn't factorize cleanly as a Kronecker product,
    but we can use block decomposition with Strassen for 2x2 blocks.
    """
    n, m, p = 3, 5, 6
    
    # Partition: n = 1+2, m = 2+3, p = 2+4
    # A (3x5) = [A11 A12; A21 A22]
    #   A11: 1x2, A12: 1x3
    #   A21: 2x2, A22: 2x3
    # B (5x6) = [B11 B12; B21 B22]
    #   B11: 2x2, B12: 2x4
    #   B21: 3x2, B22: 3x4
    
    # C = A*B
    # C11 = A11*B11 + A12*B21: 1x2
    # C12 = A11*B12 + A12*B22: 1x4
    # C21 = A21*B11 + A22*B21: 2x2
    # C22 = A21*B12 + A22*B22: 2x4
    
    # We need to compute each block multiplication and combine.
    # This requires careful index management.
    
    # Let's use a simpler approach: try multiple partition strategies
    # and use the one with minimum rank.
    
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try splitting m into m1 + m2
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        
        # Get decompositions for (n, m1, p) and (n, m2, p)
        U1, V1, W1 = _decompose_general(n, m1, p)
        U2, V2, W2 = _decompose_general(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Map U1 rows to full U
            for i in range(n):
                for j in range(m1):
                    u1_row = i * m1 + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            # Map U2 rows to full U
            for i in range(n):
                for j in range(m2):
                    u2_row = i * m2 + j
                    full_row = i * m + (m1 + j)
                    U[full_row, r1:] = U2[u2_row, :]
            
            # Map V1 rows to full V
            V[:m1*p, :r1] = V1
            
            # Map V2 rows to full V
            V[m1*p:, r1:] = V2
            
            # W is the same
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U, best_V, best_W = U, V, W
    
    return best_U, best_V, best_W

def _decompose_general(n, m, p):
    """General decomposition using recursive block decomposition."""
    # Base cases
    if n == 0 or m == 0 or p == 0:
        return _schoolbook(n, m, p)
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    if m == 1:
        return _schoolbook(n, m, p)
    
    if n == 1:
        return _schoolbook(n, m, p)
    
    if p == 1:
        return _schoolbook(n, m, p)
    
    # Try splitting m
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        
        U1, V1, W1 = _decompose_general(n, m1, p)
        U2, V2, W2 = _decompose_general(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for i in range(n):
                for j in range(m1):
                    u1_row = i * m1 + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            for i in range(n):
                for j in range(m2):
                    u2_row = i * m2 + j
                    full_row = i * m + (m1 + j)
                    U[full_row, r1:] = U2[u2_row, :]
            
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U, best_V, best_W = U, V, W
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose_3x5x5x6()
    rank = U.shape[1]
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in U.tolist()]
    V_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in V.tolist()]
    W_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
