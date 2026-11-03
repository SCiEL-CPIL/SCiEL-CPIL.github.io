import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Leave 5s margin
    
    n, m, p = 3, 4, 5
    
    # Dimensions of U, V, W
    dim_U = n * m  # 12
    dim_V = m * p  # 20
    dim_W = p * n  # 15
    
    # We will build the columns of U, V, W
    # Each column corresponds to a rank-1 term
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a schoolbook term
    def add_schoolbook_term(i, j, k):
        u = np.zeros(dim_U)
        v = np.zeros(dim_V)
        w = np.zeros(dim_W)
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Helper to add Strassen terms for a 2x2x2 block
    # i0, j0, k0 are the top-left offsets
    def add_strassen_terms(i0, j0, k0):
        # Define the 7 terms (u_pattern, v_pattern, w_pattern)
        # Patterns are lists of (local_index, value)
        # Local index for U is (di, dj) -> di*2 + dj
        # Local index for V is (dj, dk) -> dj*2 + dk
        # Local index for W is (dk, di) -> dk*2 + di
        
        terms = [
            # T1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # T2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # T3
            ([(0, 1)], [(2, 1), (0, -1)], [(1, 1), (3, 1)]),
            # T4
            ([(3, 1)], [(3, 1), (1, -1)], [(0, 1), (2, 1)]),
            # T5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # T6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # T7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        for u_pat, v_pat, w_pat in terms:
            u = np.zeros(dim_U)
            v = np.zeros(dim_V)
            w = np.zeros(dim_W)
            
            for idx, val in u_pat:
                di, dj = divmod(idx, 2)
                global_idx = (i0 + di) * m + (j0 + dj)
                u[global_idx] = val
                
            for idx, val in v_pat:
                dj, dk = divmod(idx, 2)
                global_idx = (j0 + dj) * p + (k0 + dk)
                v[global_idx] = val
                
            for idx, val in w_pat:
                dk, di = divmod(idx, 2)
                global_idx = (k0 + dk) * n + (i0 + di)
                w[global_idx] = val
                
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    # Partitioning:
    # n=3: split 2, 1 -> i_offsets: [0, 2], i_sizes: [2, 1]
    # m=4: split 2, 2 -> j_offsets: [0, 2], j_sizes: [2, 2]
    # p=5: split 2, 3 -> k_offsets: [0, 2], k_sizes: [2, 3]
    
    i_splits = [(0, 2), (2, 1)]
    j_splits = [(0, 2), (2, 2)]
    k_splits = [(0, 2), (2, 3)]
    
    for i0, i_size in i_splits:
        for j0, j_size in j_splits:
            for k0, k_size in k_splits:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    add_strassen_terms(i0, j0, k0)
                else:
                    # Schoolbook
                    for di in range(i_size):
                        for dj in range(j_size):
                            for dk in range(k_size):
                                add_schoolbook_term(i0 + di, j0 + dj, k0 + dk)
                                if time.time() > DEADLINE:
                                    # Fallback to full schoolbook if running out of time? 
                                    # But we are just building the list, so it's fast.
                                    pass

    rank = len(U_cols)
    
    # Construct final matrices
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    for r in range(rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]
        
    # Verification
    # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # This is the condition for the tensor decomposition to be exact.
    
    # We can verify a few random points or all points since n,m,p are small.
    # n=3, m=4, p=5 -> 60 points.
    # rank=58. 60*58 operations is trivial.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                # Using vectorized dot product
                val = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                
                if not np.isclose(val, 1.0, atol=1e-9):
                    # If verification fails, fall back to schoolbook rank 60
                    # This should not happen if the logic is correct
                    rank = n * m * p
                    U = np.zeros((dim_U, rank))
                    V = np.zeros((dim_V, rank))
                    W = np.zeros((dim_W, rank))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj, col] = 1
                                V[jj * p + kk, col] = 1
                                W[kk * n + ii, col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
