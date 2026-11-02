"""
Rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.

Strategy:
  - Use Strassen's rank-7 algorithm for the 2x2 x 2x2 block (columns 0,1 of B).
  - Use schoolbook rank-4 for the remaining column (column 2 of B).
  - Total rank = 7 + 4 = 11.

  The Strassen decomposition for C = A*B where A is 2x2, B is 2x2:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

  For the 2x2 x 2x2 block (B columns 0,1):
    A = [[a00, a01], [a10, a11]]  (rows i=0,1; cols j=0,1)
    B_block = [[b00, b01], [b10, b11]]  (rows j=0,1; cols k=0,1)

    Output C for this block:
    C00 = a00*b00 + a01*b10
    C01 = a00*b01 + a01*b11
    C10 = a10*b00 + a11*b10
    C11 = a10*b01 + a11*b11

  The trace convention is W[k*n + i, r] where n=2 (rows of A), p=3 (cols of B).
  So W has shape (p*n, rank) = (6, rank).
  Index in W: k*n + i, where k is column of B (0,1,2) and i is row of A (0,1).

  U has shape (n*m, rank) = (4, rank). Index: i*m + j where i in {0,1}, j in {0,1}.
  V has shape (m*p, rank) = (6, rank). Index: j*p + k where j in {0,1}, k in {0,1,2}.

  For the Strassen part (k=0,1):
    U indices: 0*(2)+0=0 (a00), 0*(2)+1=1 (a01), 1*(2)+0=2 (a10), 1*(2)+1=3 (a11)
    V indices: 0*(3)+0=0 (b00), 0*(3)+1=1 (b01), 1*(3)+0=3 (b10), 1*(3)+1=4 (b11)
    W indices for k=0: 0*2+0=0 (C00), 0*2+1=1 (C10)
    W indices for k=1: 1*2+0=2 (C01), 1*2+1=3 (C11)

  For the schoolbook part (k=2):
    V index for b02: 0*(3)+2=2, for b12: 1*(3)+2=5
    W indices for k=2: 2*2+0=4 (C02), 2*2+1=5 (C12)

  Schoolbook for k=2:
    C02 = a00*b02 + a01*b12
    C12 = a10*b02 + a11*b12
"""

import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 2, 3
    
    # Helper to create Rank 11 solution
    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen part: columns 0,1 of B (7 terms)
        # Term 0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1
        W[3][0] = 1
        
        # Term 1: M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        W[3][1] = -1
        
        # Term 2: M3
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1
        W[3][2] = 1
        
        # Term 3: M4
        U[3][3] = 1
        V[0][3] = -1; V[3][3] = 1
        W[0][3] = 1
        W[1][3] = 1
        
        # Term 4: M5
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1
        W[2][4] = 1
        
        # Term 5: M6
        U[0][5] = -1; U[2][5] = 1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: M7
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Schoolbook for column 2 (4 terms)
        # Term 7: a00 * b02
        U[0][7] = 1
        V[2][7] = 1
        W[4][7] = 1
        
        # Term 8: a01 * b12
        U[1][8] = 1
        V[5][8] = 1
        W[4][8] = 1
        
        # Term 9: a10 * b02
        U[2][9] = 1
        V[2][9] = 1
        W[5][9] = 1
        
        # Term 10: a11 * b12
        U[3][10] = 1
        V[5][10] = 1
        W[5][10] = 1
        
        return {"U": U, "V": V, "W": W}

    # Attempt Rank 10 search
    def try_rank10():
        start_time = time.time()
        deadline = time.time() + 100 # 100s limit for search
        
        R = 10
        # Indices for the 12 non-zero tensor entries
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((u_idx, v_idx, w_idx))
        
        targets = np.ones(len(indices))
        
        def residual(x):
            U = x[:4*R].reshape(4, R)
            V = x[4*R:4*R+6*R].reshape(6, R)
            W = x[4*R+6*R:4*R+6*R+6*R].reshape(6, R)
            
            res = np.zeros(len(indices))
            for idx, (u_i, v_i, w_i) in enumerate(indices):
                res[idx] = np.dot(np.dot(U[u_i, :], V[v_i, :]), W[w_i, :])
            return res - targets
        
        best_x = None
        min_error = np.inf
        
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            x0 = np.random.normal(0, 1, 4*R + 6*R + 6*R)
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=2000)
                err = np.linalg.norm(sol.fun)
                if err < min_error:
                    min_error = err
                    best_x = sol.x
                    if err < 1e-8:
                        break
            except Exception:
                continue
        
        if min_error < 1e-8 and best_x is not None:
            U = best_x[:4*R].reshape(4, R)
            V = best_x[4*R:4*R+6*R].reshape(6, R)
            W = best_x[4*R+6*R:4*R+6*R+6*R].reshape(6, R)
            
            # Check for integer/half-integer constraint
            U2 = U * 2
            V2 = V * 2
            W2 = W * 2
            
            U_int = np.round(U2)
            V_int = np.round(V2)
            W_int = np.round(W2)
            
            U_check = U_int / 2
            V_check = V_int / 2
            W_check = W_int / 2
            
            # Verify
            res_check = np.zeros(len(indices))
            for idx, (u_i, v_i, w_i) in enumerate(indices):
                res_check[idx] = np.dot(np.dot(U_check[u_i, :], V_check[v_i, :]), W_check[w_i, :])
            
            err_check = np.linalg.norm(res_check - targets)
            
            if err_check < 1e-9:
                return {"U": U_check.tolist(), "V": V_check.tolist(), "W": W_check.tolist()}
                
        return None

    result_10 = try_rank10()
    if result_10:
        return result_10
        
    return make_rank11()

    # Strassen part: columns 0,1 of B (7 terms)
    # M1 = (A11 + A22) * (B11 + B22)
    # In our indexing: A11=a00 (U[0]), A22=a11 (U[3]), B11=b00 (V[0]), B22=b11 (V[4])
    # C11 += M1, C22 += M1  => W[0] += 1, W[3] += 1
    # Wait, let me re-derive with our specific indexing.
    
    # A = [[a00, a01], [a10, a11]]
    #   a00 -> U index 0
    #   a01 -> U index 1
    #   a10 -> U index 2
    #   a11 -> U index 3
    
    # B_block (cols 0,1) = [[b00, b01], [b10, b11]]
    #   b00 -> V index 0
    #   b01 -> V index 1
    #   b10 -> V index 3
    #   b11 -> V index 4
    
    # C_block (cols 0,1) = [[C00, C01], [C10, C11]]
    #   C00 -> W index 0 (k=0, i=0)
    #   C10 -> W index 1 (k=0, i=1)
    #   C01 -> W index 2 (k=1, i=0)
    #   C11 -> W index 3 (k=1, i=1)
    
    # Strassen:
    # M1 = (a00 + a11) * (b00 + b11)
    #   U: [1, 1, 0, 1] for indices 0,1,2,3
    #   V: [1, 0, 0, 1, 0, 0] for indices 0,1,2,3,4,5
    #   C00 = M1 + M4 - M5 + M7
    #   C11 = M1 - M2 + M3 + M6
    
    # M2 = (a10 + a11) * b00
    #   U: [0, 0, 1, 1]
    #   V: [1, 0, 0, 0, 0, 0]
    #   C10 += M2, C11 -= M2
    
    # M3 = a00 * (b01 - b11)
    #   U: [1, 0, 0, 0]
    #   V: [0, 1, 0, 0, -1, 0]
    #   C01 += M3, C11 += M3
    
    # M4 = a11 * (b10 - b00)
    #   U: [0, 0, 0, 1]
    #   V: [-1, 0, 0, 1, 0, 0]
    #   C00 += M4, C10 += M4
    
    # M5 = (a00 + a01) * b11
    #   U: [1, 1, 0, 0]
    #   V: [0, 0, 0, 0, 1, 0]
    #   C00 -= M5, C01 += M5
    
    # M6 = (a10 - a00) * (b00 + b01)
    #   U: [-1, 0, 1, 0]
    #   V: [1, 1, 0, 0, 0, 0]
    #   C11 += M6
    
    # M7 = (a01 - a11) * (b10 + b11)
    #   U: [0, 1, 0, -1]
    #   V: [0, 0, 0, 1, 1, 0]
    #   C00 += M7
    
    # Let me set up each Strassen term:
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # Term 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # Term 3: M4
    U[3][3] = 1
    V[0][3] = -1; V[3][3] = 1
    W[0][3] = 1  # C00
    W[1][3] = 1  # C10
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1 # C00
    W[2][4] = 1  # C01
    
    # Term 5: M6
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1  # C00
    
    # Schoolbook for column 2 (4 terms):
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    
    # Term 7: a00 * b02
    U[0][7] = 1
    V[2][7] = 1  # b02 -> index 0*3+2=2
    W[4][7] = 1  # C02 -> index 2*2+0=4
    
    # Term 8: a01 * b12
    U[1][8] = 1
    V[5][8] = 1  # b12 -> index 1*3+2=5
    W[4][8] = 1  # C02
    
    # Term 9: a10 * b02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1  # C12 -> index 2*2+1=5
    
    # Term 10: a11 * b12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1  # C12
    
    return {"U": U, "V": V, "W": W}
