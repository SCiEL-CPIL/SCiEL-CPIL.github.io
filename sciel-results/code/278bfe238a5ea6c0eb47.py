import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11       -> C21 += M2, C22 -= M2
    # M3 = A11(B12-B22)       -> C12 += M3, C22 += M3
    # M4 = A22(B21-B11)       -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22       -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is list of (local_idx, coeff)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),           # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),           # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),           # M7
    ]
    
    col = 0
    for u in range(2):  # A column block: columns 2u, 2u+1
        for v in range(4):  # B column block: columns 2v, 2v+1
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                
                # U: A[i, 2u+j'] -> index i*m + (2u+j')
                for (i_prime, j_prime), coeff in u_coeffs.items():
                    U[i_prime * m + (2*u + j_prime), col] += coeff
                
                # V: B[2u+j', 2v+k'] -> index (2u+j')*p + (2v+k')
                for (j_prime, k_prime), coeff in v_coeffs.items():
                    V[(2*u + j_prime) * p + (2*v + k_prime), col] += coeff
                
                # W: C[i', 2v+k'] -> W index (2v+k')*n + i'
                for (i_prime, k_prime), coeff in w_coeffs.items():
                    W[(2*v + k_prime) * n + i_prime, col] += coeff
                
                col += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
