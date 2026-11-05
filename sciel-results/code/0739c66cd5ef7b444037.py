import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_dec(n, m, p):
        # Returns U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank)
        
        # Base Case 1: 1x1x1
        if n == 1 and m == 1 and p == 1:
            return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
            
        # Base Case 2: 1x2x2
        if n == 1 and m == 2 and p == 2:
            U = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ])
            V = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
            W = np.array([
                [1, 1, 0, 0],
                [0, 0, 1, 1]
            ])
            return U, V, W

        # Base Case 3: 2x2x2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            # M1 = A00(B01-B11)
            # M2 = (A00+A10)B11
            # M3 = A10(B01-B00)
            # M4 = A11(B10-B00)
            # M5 = (A00+A10)B01
            # M6 = (A01-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # Indices:
            # U: A00=0, A01=1, A10=2, A11=3
            # V: B00=0, B01=1, B10=2, B11=3
            # W: C00=0, C01=1, C10=2, C11=3
            
            # M1: A00 * (B01 - B11)
            U[0, 0] = 1
            V[1, 0] = 1; V[3, 0] = -1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2: (A00 + A10) * B11
            U[0, 1] = 1; U[2, 1] = 1
            V[3, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            
            # M3: A10 * (B01 - B00)
            U[2, 2] = 1
            V[1, 2] = 1; V[0, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            
            # M4: A11 * (B10 - B00)
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            
            # M5: (A00 + A10) * B01
            U[0, 4] = 1; U[2, 4] = 1
            V[1, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            
            # M6: (A01 - A00) * (B00 + B10)
            U[1, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[2, 5] = 1
            W[3, 5] = 1
            
            # M7: (A01 - A11) * (B01 + B11)
            U[1, 6] = 1; U[3, 6] = -1
            V[1, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1; W[3, 6] = 1
            
            return U, V, W

        # Recursive Step
        best_rank = float('inf')
        best_U = best_V = best_W = None
        
        # Try splitting m
        if m % 2 == 0:
            m1 = m // 2
            U1, V1, W1 = get_dec(n, m1, p)
            U2, V2, W2 = get_dec(n, m1, p)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                # Pad and combine
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                # Map U1 to left half of A columns
                for i in range(n):
                    for j in range(m1):
                        U[i * m + j, :U1.shape[1]] = U1[i * m1 + j, :]
                
                # Map U2 to right half of A columns
                for i in range(n):
                    for j in range(m1):
                        U[i * m + (m1 + j), U1.shape[1]:] = U2[i * m1 + j, :]
                        
                # Map V1 to top half of B rows
                for j in range(m1):
                    for k in range(p):
                        V[j * p + k, :V1.shape[1]] = V1[j * p + k, :]
                        
                # Map V2 to bottom half of B rows
                for j in range(m1):
                    for k in range(p):
                        V[(m1 + j) * p + k, V1.shape[1]:] = V2[j * p + k, :]
                        
                W[:, :W1.shape[1]] = W1
                W[:, W1.shape[1]:] = W2
                
                best_rank = r
                best_U, best_V, best_W = U, V, W

        # Try splitting p
        if p % 2 == 0:
            p1 = p // 2
            U1, V1, W1 = get_dec(n, m, p1)
            U2, V2, W2 = get_dec(n, m, p1)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                U[:, :U1.shape[1]] = U1
                U[:, U1.shape[1]:] = U2
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p1):
                            V[j * p + k, :V1.shape[1]] = V1[j * p1 + k, :]
                            V[j * p + (p1 + k), V1.shape[1]:] = V2[j * p1 + k, :]
                            
                for k in range(p1):
                    for i in range(n):
                        W[k * n + i, :W1.shape[1]] = W1[k * p1 + i, :]
                        W[(p1 + k) * n + i, W1.shape[1]:] = W2[k * p1 + i, :]
                        
                best_rank = r
                best_U, best_V, best_W = U, V, W

        # Try splitting n
        if n % 2 == 0:
            n1 = n // 2
            U1, V1, W1 = get_dec(n1, m, p)
            U2, V2, W2 = get_dec(n1, m, p)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                for i in range(n1):
                    for j in range(m):
                        U[i * m + j, :U1.shape[1]] = U1[i * m + j, :]
                        U[(n1 + i) * m + j, U1.shape[1]:] = U2[i * m + j, :]
                        
                V[:, :V1.shape[1]] = V1
                V[:, V1.shape[1]:] = V2
                
                for k in range(p):
                    for i in range(n1):
                        W[k * n + i, :W1.shape[1]] = W1[k * n1 + i, :]
                        W[k * n + (n1 + i), W1.shape[1]:] = W2[k * n1 + i, :]
                        
                best_rank = r
                best_U, best_V, best_W = U, V, W
        else:
            # Split n into 1 and n-1
            if n > 1:
                U1, V1, W1 = get_dec(1, m, p)
                U2, V2, W2 = get_dec(n-1, m, p)
                r = U1.shape[1] + U2.shape[1]
                if r < best_rank:
                    U = np.zeros((n * m, r))
                    V = np.zeros((m * p, r))
                    W = np.zeros((p * n, r))
                    
                    for j in range(m):
                        U[j, :U1.shape[1]] = U1[j, :]
                        U[(n-1)*m + j, U1.shape[1]:] = U2[j, :] # Wait, U2 is for n-1 rows
                        # Correction: U2 indices are for rows 0..n-2.
                        # We need to map U2 row i to U row (1+i).
                        for i in range(n-1):
                            U[(1+i)*m + j, U1.shape[1]:] = U2[i*m + j, :]
                        
                    V[:, :V1.shape[1]] = V1
                    V[:, V1.shape[1]:] = V2
                    
                    for k in range(p):
                        W[k * n + 0, :W1.shape[1]] = W1[k * 1 + 0, :]
                        for i in range(n-1):
                            W[k * n + (1+i), W1.shape[1]:] = W2[k * (n-1) + i, :]
                            
                    best_rank = r
                    best_U, best_V, best_W = U, V, W

        if best_U is None:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return U, V, W

        return best_U, best_V, best_W

    U, V, W = get_dec(n, m, p)
    
    # Convert to list of lists as per interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
