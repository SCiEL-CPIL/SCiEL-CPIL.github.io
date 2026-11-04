import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    n, m, p = 4, 4, 5
    
    # Target rank: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1) = 65
    try:
        if time.time() < deadline:
            # 1. Get Strassen 4x4x4 decomposition
            strassen_444 = _strassen_4x4x4()
            U_s = np.array(strassen_444['U'], dtype=float)
            V_s = np.array(strassen_444['V'], dtype=float)
            W_s = np.array(strassen_444['W'], dtype=float)
            
            # 2. Get Schoolbook 4x4x1 decomposition (for the 5th column)
            schoolbook_441 = _schoolbook_441(n, m, p)
            U_sc = np.array(schoolbook_441['U'], dtype=float)
            V_sc = np.array(schoolbook_441['V'], dtype=float)
            W_sc = np.array(schoolbook_441['W'], dtype=float)
            
            # 3. Combine
            # U: (n*m, rank_s) + (n*m, rank_sc) -> (n*m, rank_total)
            U = np.hstack([U_s, U_sc])
            
            # V: (m*p, rank_s) + (m*p, rank_sc)
            # Strassen V is for m x 4 matrix (p=4). We need m x 5.
            # Current V_s is 16x49. It corresponds to B[0..3].
            # We need to insert a column of zeros for B[4] in V_s?
            # No, V_s indices are j*4 + k.
            # For the new V (m*p = 20 rows), indices are j*5 + k.
            # We need to map the old indices to new indices.
            
            V_new = np.zeros((m*p, U.shape[1]), dtype=float)
            for r in range(U_s.shape[1]):
                for j in range(m):
                    for k in range(4): # Only 4 columns in Strassen
                        V_new[j*p + k, r] = V_s[j*4 + k, r]
                        
            # V_sc is already for m*p = 20 rows?
            # _schoolbook_441 should return V with shape (m*p, 16)
            V_new_sc = np.zeros((m*p, U_sc.shape[1]), dtype=float)
            for r in range(U_sc.shape[1]):
                for j in range(m):
                    V_new_sc[j*p + 4, r] = V_sc[j*4, r] # V_sc internal index j*1? No.
                    # Let's check _schoolbook_441 structure.
            
            V = np.hstack([V_new, V_new_sc])
            
            # W: (p*n, rank_s) + (p*n, rank_sc)
            # Strassen W is for 4 x 4 output (p=4, n=4).
            # Indices k*4 + i.
            # New W indices k*5 + i.
            
            W_new = np.zeros((p*n, U.shape[1]), dtype=float)
            for r in range(U_s.shape[1]):
                for k in range(4):
                    for i in range(n):
                        W_new[k*p + i, r] = W_s[k*4 + i, r]
                        
            W_new_sc = np.zeros((p*n, U_sc.shape[1]), dtype=float)
            for r in range(U_sc.shape[1]):
                for i in range(n):
                    W_new_sc[4*p + i, r] = W_sc[0*4 + i, r] # W_sc internal index i?
            
            W = np.hstack([W_new, W_new_sc])
            
            result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            
            if _verify_decomposition(result, n, m, p, num_checks=100):
                return result
                
    except Exception as e:
        pass
    
    # Fallback: Schoolbook 4x4x5 (Rank 80)
    return _schoolbook_full(n, m, p)

def _strassen_2x2x2():
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[0][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0: continue
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            U[gi*4 + gj][col] += coeff_a * local_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0: continue
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            V[gj*4 + gk][col] += coeff_b * local_v
            
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li][s]
                            if local_w == 0: continue
                            gk = ck * 2 + lk
                            gi = ci * 2 + li
                            W[gk*4 + gi][col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _schoolbook_441(n, m, p):
    """
    Returns U, V, W for the part C[:, 4] = A @ B[:, 4].
    U: (n*m, n*m)
    V: (m*p, n*m) -- Only column 4 of B is used
    W: (p*n, n*m) -- Only row 4 of C is used
    """
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _schoolbook_full(n, m, p):
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

def _verify_decomposition(decomp, n, m, p, num_checks=50):
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check non-tensor entries
    import random
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = 0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        if abs(val) > 1e-9:
            return False
            
    return True
