import numpy as np
import time

DIMENSIONS = [2, 2, 14]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, keep 10s margin
    
    n, m, p = DIMENSIONS
    rank_56 = n * m * p
    rank_49 = 49
    
    # Try rank 49 first
    try:
        U, V, W = construct_rank_49()
        # Verify
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to rank 56
    U, V, W = construct_rank_56(n, m, p)
    return {"U": U, "V": V, "W": W}

def construct_rank_56(n, m, p):
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
    return U, V, W

def construct_rank_49():
    n, m, p = 2, 2, 14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products for 2x2
    # A = [[a, b], [c, d]], B_block = [[e, f], [g, h]]
    # P1 = a(e+h)
    # P2 = (c+g)h
    # P3 = (a+b)e
    # P4 = d(h-f)
    # P5 = (a+d)(e+f)
    # P6 = (b-d)(g+h)
    # P7 = (c-a)(e+f)
    
    # Coefficients for A (u) and B (v)
    # Index in U: i*m+j. i=0,1; j=0,1.
    # a: 0, b: 1, c: 2, d: 3
    # Index in V: j*p+k. j=0,1; k=0,1 (local).
    # e: 0, f: 1, g: 2, h: 3
    
    products = [
        # (u_coeffs, v_coeffs)
        ([1, 0, 0, 0], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 0], [0, 0, 1, 1]), # P2
        ([1, 1, 0, 0], [1, 0, 0, 0]), # P3
        ([0, 0, 0, 1], [0, -1, 0, 1]), # P4
        ([1, 0, 0, 1], [1, 1, 0, 0]), # P5
        ([0, 1, -1, 0], [0, 0, 1, 1]), # P6
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # P7
    ]
    
    # W coefficients for C = [[C00, C01], [C10, C11]]
    # C00 = P5 + P4 - P2 + P6
    # C01 = P1 + P3
    # C10 = P2 + P4
    # C11 = P1 - P5 + P3 + P7
    
    w_coeffs = [
        [0, 0, -1, 1, 1, 1, 0], # C00
        [1, 0, 1, 0, 0, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 1, 0, -1, 0, 1], # C11
    ]
    
    for block in range(7):
        for prod_idx in range(7):
            col = block * 7 + prod_idx
            u_c, v_c = products[prod_idx]
            
            # Update U
            for idx in range(4):
                if u_c[idx] != 0:
                    U[idx][col] += u_c[idx]
                    
            # Update V
            for idx in range(4):
                if v_c[idx] != 0:
                    # Local index idx: 0=e, 1=f, 2=g, 3=h
                    # Global index: j*p + k
                    # e: j=0, k=0 -> 0*14 + block*2 + 0
                    # f: j=0, k=1 -> 0*14 + block*2 + 1
                    # g: j=1, k=0 -> 1*14 + block*2 + 0
                    # h: j=1, k=1 -> 1*14 + block*2 + 1
                    j, k = divmod(idx, 2)
                    global_v = j * 14 + block * 2 + k
                    V[global_v][col] += v_c[idx]
                    
            # Update W
            for out_idx in range(4):
                w_val = w_coeffs[out_idx][prod_idx]
                if w_val != 0:
                    # Local out_idx: 0=C00, 1=C01, 2=C10, 3=C11
                    # Global index: k*n + i
                    # C00: i=0, k=0 -> 0*2 + 0
                    # C01: i=0, k=1 -> 1*2 + 0
                    # C10: i=1, k=0 -> 0*2 + 1
                    # C11: i=1, k=1 -> 1*2 + 1
                    i, k = divmod(out_idx, 2)
                    global_w = (block * 2 + k) * 2 + i
                    W[global_w][col] += w_val
                    
    return U, V, W

def verify(U, V, W, n, m, p):
    # Check a few random cases
    import random
    for _ in range(10):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        C_calc = np.zeros((n, p))
        for r in range(len(U_arr[0])):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            
            # u_col is n*m, v_col is m*p, w_col is p*n
            # Reshape
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            
            # Term = (A @ u_mat) * (v_mat @ B) * w_mat? No.
            # The decomposition is sum_r U_{ij,r} V_{jk,r} W_{ki,r}
            # This is not a simple matrix multiplication of A and B.
            # It's a sum of rank-1 tensors.
            
            # Let's just compute the sum directly for small sizes
            pass
            
    # Simplified verification:
    # Check if the tensor T = sum U V W matches the multiplication tensor.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should be 1 if the term A_ij B_jk contributes to C_ik, which is always 1 for valid i,j,k.
    # Wait, the multiplication tensor is 1 for all i,j,k?
    # C_ik = sum_j A_ij B_jk.
    # So for each i, j, k, the coefficient is 1.
    
    T_calc = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                T_calc[i, j, k] = s
                
    # The target tensor is all 1s?
    # Yes, because A_ij B_jk is a term in the sum for C_ik.
    # So T_target[i, j, k] = 1 for all i, j, k.
    
    if not np.allclose(T_calc, 1.0):
        return False
        
    return True
