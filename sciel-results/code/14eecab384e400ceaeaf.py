import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try ranks from max(n,m,p) upwards
    for r in range(max(n, m, p), 25):
        if time.time() > deadline - 5.0:
            break
        
        # Search for U, V
        # Entries from {-1, -0.5, 0, 0.5, 1}
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Generate random U, V with correct shapes
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
            
            # Check if we can find W
            # For each i in 0..n-1, k in 0..p-1:
            # Solve A_ik W_ik = b_ik
            W = np.zeros((p * n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # A_ik is (m*m) x r
                    # b_ik is (m*m) x 1
                    A_ik = np.zeros((m * m, r))
                    b_ik = np.zeros(m * m)
                    
                    for j in range(m):
                        for jp in range(m):
                            idx = j * m + jp
                            b_ik[idx] = 1.0 if j == jp else 0.0
                            # A_ik[idx, r] = U[i*m + j, r] * V[jp*p + k, r]
                            # Vectorized:
                            # U_row = U[i*m + j, :]
                            # V_col = V[jp*p + k, :]
                            # A_ik[idx, :] = U_row * V_col
                            pass
                            
                    # Vectorize the filling of A_ik
                    # U is (n*m, r). We need U[i*m+j, :] for j in 0..m-1.
                    # V is (m*p, r). We need V[jp*p+k, :] for jp in 0..m-1.
                    
                    # Construct U_part: (m, r)
                    U_part = U[i*m : (i+1)*m, :]
                    # Construct V_part: (m, r)
                    # V indices: jp*p + k for jp in 0..m-1
                    v_indices = [jp * p + k for jp in range(m)]
                    V_part = V[v_indices, :]
                    
                    # A_ik[j*m + jp, r] = U_part[j, r] * V_part[jp, r]
                    # This is an outer product-like operation but element-wise across r.
                    # Shape of U_part is (m, r). Shape of V_part is (m, r).
                    # We want (m*m, r).
                    # A_ik = np.einsum('jr, kp -> jkp r', U_part, V_part).reshape(m*m, r)
                    try:
                        A_ik = np.einsum('jr, kp -> jkp r', U_part, V_part).reshape(m * m, r)
                    except:
                        valid = False
                        break
                        
                    # Solve A_ik W_ik = b_ik
                    try:
                        W_ik, residuals, rank, s = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check residual
                    if np.max(np.abs(A_ik @ W_ik - b_ik)) > 1e-6:
                        valid = False
                        break
                        
                    # Check if W_ik is half-integer
                    W_ik_rounded = np.round(W_ik * 2) / 2
                    if np.max(np.abs(W_ik - W_ik_rounded)) > 1e-6:
                        valid = False
                        break
                        
                    W[k * n + i, :] = W_ik_rounded
                    
            if valid:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # If not found, return schoolbook
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
