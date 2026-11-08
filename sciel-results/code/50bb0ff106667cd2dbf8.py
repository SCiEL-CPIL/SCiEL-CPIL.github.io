import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x6 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with shapes (16, R), (24, R), (24, R).
    Score is the rank R (lower is better).
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Use the current best known approach: top-level Strassen (7 terms) 
    # with inner decomposition for each (2x2) x (2x3) block.
    # 
    # For (2x2) x (2x3):
    # - Use Strassen for the 2x2 x 2x2 part (7 multiplications)
    # - Use schoolbook for the 2x2 x 2x1 part (4 multiplications)
    # Total per block: 11 multiplications
    # Total rank: 7 * 11 = 77
    #
    # Let's verify this is correct and try to optimize.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix, Y_s is a 2x3 matrix
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify the decomposition is correct
    # The tensor T[i,j,k] = 1 if i == k (for matrix multiplication A*B where A is n x m, B is m x p)
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = delta(i,k) ... no wait.
    # T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = delta_{i,k} ... no.
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k
    # and the decomposition is: M[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # where U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # and the index mapping is:
    # U: row index = i*m + j (i in [0,n), j in [0,m))
    # V: row index = j*p + k (j in [0,m), k in [0,p))
    # W: row index = k*n + i (k in [0,p), i in [0,n))
    
    # Let's verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k (since matrix multiplication tensor is all ones)
    # Wait, no. The matrix multiplication tensor M_{n,m,p} is defined by:
    # M[i,j,k] = 1 if and only if the (i,k)-th entry of C = A*B uses A[i,j] and B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k.
    
    # Let's check a few entries
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :])
                # This should equal sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                # But we need to multiply by W[w_idx,r] too
                # Let me redo this
                pass
    
    # Actually, the tensor decomposition is:
    # M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and M[i,j,k] should be 1 for all i,j,k
    
    # Let's verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    # If max_err is small, the decomposition is correct
    # If not, there's a bug
    
    # Convert to lists
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
