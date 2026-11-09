import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Target Tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Schoolbook fallback
    rank_school = N_U * N_V // n # No, rank is n*m*p
    rank_school = n * m * p
    U_school = np.zeros((N_U, rank_school))
    V_school = np.zeros((N_V, rank_school))
    W_school = np.zeros((N_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
                
    # Convert to lists for return
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit
    
    best_result = None
    best_rank = rank_school
    
    # Try ranks from 30 down to 20
    for r in range(30, 19, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Random init
            U = np.random.randn(N_U, r)
            V = np.random.randn(N_V, r)
            W = np.random.randn(N_W, r)
            
            # ALS iterations
            for it in range(50):
                # Solve U
                # T = sum_c U[:, c] (V[:, c] W[:, c])^T
                # Flatten T and U
                # A_U shape: (N_U*N_V*N_W, N_U*r)
                # This is too big to build explicitly every time?
                # 1296 x 270 is small.
                
                # Construct A_U
                # For each c, and each u, the column in A_U corresponding to U[u, c]
                # has non-zeros at positions (u, v, w) with value V[v, c] * W[w, c]
                
                # Efficient construction:
                # We can use einsum or just loops for small sizes.
                # Let's use a loop to fill A_U.
                A_U = np.zeros((N_U * N_V * N_W, N_U * r))
                b_U = T.flatten()
                
                for c in range(r):
                    # For each u, the vector of coefficients is V[:, c] * W[:, c]
                    coeffs = V[:, c] * W[:, c]
                    # This vector is placed in the block of rows corresponding to u
                    # Rows for u are u*N_V*N_W to (u+1)*N_V*N_W
                    start_row = u * N_V * N_W if False else 0 # Wait, indexing
                    # The index in flattened T is u * N_V * N_W + v * N_W + w
                    # So for fixed u, the rows are contiguous.
                    start_row = u * N_V * N_W
                    end_row = start_row + N_V * N_W
                    col_idx = u * r + c
                    A_U[start_row:end_row, col_idx] = coeffs
                    
                # Solve least squares
                U_new, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_new.reshape(N_U, r)
                
                # Solve V
                A_V = np.zeros((N_U * N_V * N_W, N_V * r))
                for c in range(r):
                    coeffs = U[:, c] * W[:, c] # Shape (N_U, ) and (N_W, ) -> need outer?
                    # T[u, v, w] = U[u, c] V[v, c] W[w, c]
                    # For fixed v, the coefficients are U[:, c] (reshaped to (N_U, 1, 1)) * W[:, c] (reshaped to (1, 1, N_W))
                    # Flattened, for fixed v, the rows are not contiguous in u, w?
                    # Index: u * N_V * N_W + v * N_W + w
                    # For fixed v, u varies from 0 to N_U-1, w from 0 to N_W-1.
                    # The rows are scattered.
                    # Let's just fill the whole matrix.
                    for u in range(N_U):
                        for w in range(N_W):
                            row_idx = u * N_V * N_W + v * N_W + w
                            col_idx = v * r + c
                            A_V[row_idx, col_idx] = U[u, c] * W[w, c]
                            
                V_new, _, _, _ = np.linalg.lstsq(A_V, b_U, rcond=None)
                V = V_new.reshape(N_V, r)
                
                # Solve W
                A_W = np.zeros((N_U * N_V * N_W, N_W * r))
                for c in range(r):
                    for u in range(N_U):
                        for v in range(N_V):
                            row_idx = u * N_V * N_W + v * N_W + w # w is loop var?
                            # w is the index for W
                            for w in range(N_W):
                                row_idx = u * N_V * N_W + v * N_W + w
                                col_idx = w * r + c
                                A_W[row_idx, col_idx] = U[u, c] * V[v, c]
                                
                W_new, _, _, _ = np.linalg.lstsq(A_W, b_U, rcond=None)
                W = W_new.reshape(N_W, r)
                
                # Check residual
                # Reconstruct T
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                residual = np.linalg.norm(T - T_recon)
                
                if residual < 1e-9:
                    # Try to snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    T_snap = np.einsum('uc,vc,wc->uvw', U_snap, V_snap, W_snap)
                    res_snap = np.linalg.norm(T - T_snap)
                    if res_snap < 1e-9:
                        best_result = (U_snap, V_snap, W_snap, r)
                        found = True
                        break
                    else:
                        # If snap fails, maybe keep float? 
                        # Prompt requires integer/half-integer.
                        # So we must snap.
                        pass
                        
            if found:
                break
                
        if found:
            break
            
    if best_result:
        U, V, W, r = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
