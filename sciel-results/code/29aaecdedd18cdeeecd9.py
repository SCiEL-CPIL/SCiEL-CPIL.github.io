import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110
    
    # Schoolbook fallback
    R0 = n * m * p
    U0 = np.zeros((n*m, R0))
    V0 = np.zeros((m*p, R0))
    W0 = np.zeros((p*n, R0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, c] = 1
                V0[j*p+k, c] = 1
                W0[k*n+i, c] = 1
                c += 1
    best_U, best_V, best_W = U0, V0, W0
    best_R = R0

    # Try Rank 59
    # 23 terms for 3x3x3
    # 36 terms for 3x3x4 (schoolbook)
    R1 = 59
    U1 = np.zeros((n*m, R1))
    V1 = np.zeros((m*p, R1))
    W1 = np.zeros((p*n, R1))
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx: list of (i,j)
        # v_idx: list of (j,k)
        # w_idx: list of (i,k)
        pass

    # Hardcoded 23 terms for 3x3x3
    # Indices: A is 3x3 (0-2), B is 3x3 (0-2), C is 3x3 (0-2)
    # Term format: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i,j): val}
    # v_coeffs: dict {(j,k): val}
    # w_coeffs: dict {(i,k): val}
    
    terms_23 = []
    
    # 1-3: Diagonal
    terms_23.append[({(0,0):1}, {(0,0):1}, {(0,0):1})]
    terms_23.append[({(1,1):1}, {(1,1):1}, {(1,1):1})]
    terms_23.append[({(2,2):1}, {(2,2):1}, {(2,2):1})]
    
    # 4-6: Shifted Diagonals
    # (A00+A01)B11 -> C01
    terms_23.append[({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1})]
    # (A11+A12)B22 -> C12
    terms_23.append[({(1,1):1, (1,2):1}, {(2,2):1}, {(1,2):1})]
    # (A22+A20)B00 -> C20
    terms_23.append[({(2,2):1, (2,0):1}, {(0,0):1}, {(2,0):1})]
    
    # 7-9: Row/Col sums
    # A01(B10+B11) -> C00, C01
    terms_23.append[({(0,1):1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):1})]
    # A12(B21+B22) -> C11, C12
    terms_23.append[({(1,2):1}, {(2,1):1, (2,2):1}, {(1,1):1, (1,2):1})]
    # A20(B02+B00) -> C22, C20
    terms_23.append[({(2,0):1}, {(0,2):1, (0,0):1}, {(2,2):1, (2,0):1})]
    
    # 10-12: 2x2 Strassen-like
    # (A00+A11)(B00+B11) -> C00, C01, C10, C11
    terms_23.append[({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1})]
    # (A11+A22)(B11+B22) -> C11, C12, C21, C22
    terms_23.append[({(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}, {(1,1):1, (1,2):1, (2,1):1, (2,2):1})]
    # (A22+A00)(B22+B00) -> C22, C20, C02, C00
    terms_23.append[({(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}, {(2,2):1, (2,0):1, (0,2):1, (0,0):1})]
    
    # 13-15: More shifts
    # A02(B20+B21) -> C00, C01
    terms_23.append[({(0,2):1}, {(2,0):1, (2,1):1}, {(0,0):1, (0,1):1})]
    # A10(B01+B02) -> C11, C12
    terms_23.append[({(1,0):1}, {(0,1):1, (0,2):1}, {(1,1):1, (1,2):1})]
    # A21(B12+B10) -> C22, C20
    terms_23.append[({(2,1):1}, {(1,2):1, (1,0):1}, {(2,2):1, (2,0):1})]
    
    # 16-18: Cyclic sums
    # (A01+A12)(B20+B01) -> C00, C01, C10, C11
    terms_23.append[({(0,1):1, (1,2):1}, {(2,0):1, (0,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1})]
    # (A12+A20)(B02+B12) -> C12, C10, C22, C20
    terms_23.append[({(1,2):1, (2,0):1}, {(0,2):1, (1,2):1}, {(1,2):1, (1,0):1, (2,2):1, (2,0):1})]
    # (A20+A01)(B10+B21) -> C20, C21, C00, C01
    terms_23.append[({(2,0):1, (0,1):1}, {(1,0):1, (2,1):1}, {(2,0):1, (2,1):1, (0,0):1, (0,1):1})]
    
    # 19-21: Global sums
    # (A00+A11+A22)(B00+B11+B22)
    terms_23.append[({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1})]
    # (A01+A12+A20)(B10+B21+B02)
    terms_23.append[({(0,1):1, (1,2):1, (2,0):1}, {(1,0):1, (2,1):1, (0,2):1}, {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1})]
    # (A02+A10+A21)(B20+B01+B12)
    terms_23.append[({(0,2):1, (1,0):1, (2,1):1}, {(2,0):1, (0,1):1, (1,2):1}, {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1})]
    
    # 22-23: Differences
    # (A00+A11-A22)(B00+B11-B22)
    terms_23.append[({(0,0):1, (1,1):1, (2,2):-1}, {(0,0):1, (1,1):1, (2,2):-1}, {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1})]
    # (A00-A11+A22)(B00-B11+B22)
    terms_23.append[({(0,0):1, (1,1):-1, (2,2):1}, {(0,0):1, (1,1):-1, (2,2):1}, {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1})]

    # Fill U1, V1, W1 for the 23 terms
    col = 0
    for u_dict, v_dict, w_dict in terms_23:
        for (i, j), val in u_dict.items():
            U1[i*m+j, col] = val
        for (j, k), val in v_dict.items():
            V1[j*p+k, col] = val # Note: k is 0-2 here
        for (i, k), val in w_dict.items():
            W1[k*n+i, col] = val
        col += 1
        
    # Fill remaining 36 terms for 3x3x4 (k=3..6)
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U1[i*m+j, col] = 1
                V1[j*p+k, col] = 1
                W1[k*n+i, col] = 1
                col += 1
                
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for _ in range(20):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            
            # Compute C using decomposition
            C_comp = np.zeros((n, p))
            for c in range(rank):
                u_val = np.dot(U[:, c], A.flatten())
                v_val = np.dot(V[:, c], B.flatten())
                # w is a matrix? No, w is a vector of length p*n.
                # W[k*n+i, c] is the coefficient for C[i, k].
                w_vec = W[:, c]
                # Reshape w_vec to (p, n) ? No, W is (p*n, rank).
                # W[k*n + i] corresponds to C[i, k].
                # So C[i, k] += u_val * v_val * W[k*n + i, c]
                
                # Vectorized:
                # C_comp += np.outer(???)
                # Let's do it explicitly for safety
                for i in range(n):
                    for k in range(p):
                        C_comp[i, k] += u_val * v_val * W[k*n+i, c]
                        
            if not np.allclose(C_true, C_comp, atol=1e-6):
                return False
        return True

    if verify(U1, V1, W1, n, m, p, R1):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    else:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
