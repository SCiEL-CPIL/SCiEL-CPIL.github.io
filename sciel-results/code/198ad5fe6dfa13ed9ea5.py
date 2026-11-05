import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    def is_valid(U, V, W, R):
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i*m + j1, r] * V[j2*p + k, r] * W[k*n + i, r]
                        target = 1 if j1 == j2 else 0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def get_schoolbook():
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

    start_time = time.time()
    deadline = start_time + 100
    
    # Try to find a decomposition with R=40
    # Use structured initialization: Strassen for 2x2x2 block
    # This is a heuristic within the search family
    
    for R in [40, 38, 36]:
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            
            # Initialize first 7 columns with Strassen pattern for 2x2x2 block
            # Indices:
            # A: 2x2 (rows 0,1; cols 0,1) -> U indices: 0,1,2,3
            # B: 2x2 (rows 0,1; cols 0,1) -> V indices: 0,1, 5,6 (since B is 3x5, row 0 is 0-4, row 1 is 5-9)
            # W: 2x2 (rows 0,1; cols 0,1) -> W indices: 0,1, 3,4 (since W is 5x3, col 0 is 0-2, col 1 is 3-5? No, W[k*n+i]. k is col of C.
            # C is 3x5. k in 0..4, i in 0..2.
            # Top-left 2x2 of C: k in 0,1; i in 0,1.
            # W indices:
            # k=0, i=0 -> 0
            # k=0, i=1 -> 1
            # k=1, i=0 -> 3
            # k=1, i=1 -> 4
            
            if R >= 7:
                # Strassen M1..M7
                # U rows: 0,1,2,3 (A11, A12, A21, A22)
                # V cols: 0,1,5,6 (B11, B12, B21, B22) -- Note: B is 3x5.
                # B11 is B[0,0], B12 is B[0,1], B21 is B[1,0], B22 is B[1,1].
                # V indices: 
                # B[0,0] -> 0
                # B[0,1] -> 1
                # B[1,0] -> 5
                # B[1,1] -> 6
                
                # M1 = (A11+A22)(B11+B22)
                U[0, 0] = 1; U[3, 0] = 1
                V[0, 0] = 1; V[6, 0] = 1
                W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1; W[4, 0] = 1
                
                # M2 = (A21+A22)B11
                U[2, 1] = 1; U[3, 1] = 1
                V[0, 1] = 1
                W[3, 1] = 1; W[4, 1] = 1
                
                # M3 = A11(B12-B22)
                U[0, 2] = 1
                V[1, 2] = 1; V[6, 2] = -1
                W[0, 2] = 1; W[1, 2] = 1
                
                # M4 = A22(B21-B11)
                U[3, 3] = 1
                V[5, 3] = 1; V[0, 3] = -1
                W[3, 3] = 1; W[4, 3] = 1
                
                # M5 = (A11+A12)B22
                U[0, 4] = 1; U[1, 4] = 1
                V[6, 4] = 1
                W[0, 4] = 1; W[1, 4] = 1
                
                # M6 = (A21-A11)(B11+B12)
                U[2, 5] = 1; U[0, 5] = -1
                V[0, 5] = 1; V[1, 5] = 1
                W[3, 5] = 1; W[4, 5] = 1
                
                # M7 = (A12-A22)(B21+B22)
                U[1, 6] = 1; U[3, 6] = -1
                V[5, 6] = 1; V[6, 6] = 1
                W[0, 6] = 1; W[1, 6] = 1
                
            # Fill rest with random
            for r in range(7, R):
                U[:, r] = np.random.randint(-1, 2, n*m)
                V[:, r] = np.random.randint(-1, 2, m*p)
            
            # Solve for W
            num_eq = n * p * m * m
            num_unk = p * n * R
            
            A = np.zeros((num_eq, num_unk))
            b = np.zeros(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            for r in range(R):
                                coeff = U[i*m + j1, r] * V[j2*p + k, r]
                                if coeff != 0:
                                    w_idx = (k*n + i) * R + r
                                    A[eq_idx, w_idx] = coeff
                            b[eq_idx] = 1 if j1 == j2 else 0
                            eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_flat = np.round(W_flat * 2) / 2
            W = W_flat.reshape(p*n, R)
            
            if is_valid(U, V, W, R):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        if time.time() > deadline - 10:
            break

    return get_schoolbook()
