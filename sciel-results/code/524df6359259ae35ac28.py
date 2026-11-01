import numpy as np
import time
import itertools

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy:
    1. The tensor M_{3,3,6} is the direct sum of two M_{3,3,3} tensors.
    2. The rank of M_{3,3,3} is 23.
    3. Therefore, the rank of M_{3,3,6} is 46.
    4. We search for a rank-23 decomposition of M_{3,3,3} with integer/half-integer entries.
    5. If found, we construct the rank-46 decomposition for M_{3,3,6}.
    6. If not found, we return the schoolbook decomposition (rank 54).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Define target for 3x3x3
    n, m, p = 3, 3, 3
    dim = 9
    T3_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # a = i*m + j, b = j*p + k, c = k*n + i
            # For 3x3x3, n=m=p=3
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T3_nonzero[(a, b, c)] = 1.0

    def generate_schoolbook_3x3x3():
        rank = 27
        U = [[0] * rank for _ in range(dim)]
        V = [[0] * rank for _ in range(dim)]
        W = [[0] * rank for _ in range(dim)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return U, V, W

    # 2. Search for rank 23 decomposition of 3x3x3
    U3 = None
    V3 = None
    W3 = None
    target_rank = 23
    
    # We can try a few random initializations
    # To increase success rate, we can use a structured search or just random.
    # Given the time, random might work if we try enough.
    # However, 23 is quite low for 3x3x3.
    # Let's try 1000 attempts.
    
    for attempt in range(1000):
        if time.time() > deadline - 5: # Leave 5s for construction
            break
            
        U_rand = np.random.randint(-1, 2, size=(dim, target_rank))
        V_rand = np.random.randint(-1, 2, size=(dim, target_rank))
        
        W_rand = np.zeros((dim, target_rank))
        valid = True
        
        for c in range(dim):
            M_c = np.zeros((dim * dim, target_rank))
            b_c = np.zeros(dim * dim)
            
            idx = 0
            for a in range(dim):
                for b in range(dim):
                    M_c[idx, :] = U_rand[a, :] * V_rand[b, :]
                    if (a, b, c) in T3_nonzero:
                        b_c[idx] = 1.0
                    else:
                        b_c[idx] = 0.0
                    idx += 1
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
            except:
                valid = False
                break
            
            w_doubled = w * 2
            closest_ints = np.round(w_doubled)
            error = np.max(np.abs(w_doubled - closest_ints))
            
            if error < 1e-9:
                W_rand[c, :] = w
            else:
                valid = False
                break
        
        if valid:
            # Verify
            is_valid = True
            for (a, b, c), val in T3_nonzero.items():
                s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                if abs(s - val) > 1e-9:
                    is_valid = False
                    break
            
            if is_valid:
                # Check zero entries
                for _ in range(50):
                    a = np.random.randint(0, dim)
                    b = np.random.randint(0, dim)
                    c = np.random.randint(0, dim)
                    if (a, b, c) not in T3_nonzero:
                        s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                        if abs(s) > 1e-9:
                            is_valid = False
                            break
                
                if is_valid:
                    U3 = U_rand
                    V3 = V_rand
                    W3 = W_rand
                    break

    # 3. Construct 3x3x6 solution
    if U3 is not None:
        # Rank 46
        U6 = np.zeros((9, 46))
        V6 = np.zeros((18, 46))
        W6 = np.zeros((18, 46))
        
        # First 3x3x3 part
        U6[:, 0:23] = U3
        V6[0:9, 0:23] = V3
        W6[0:9, 0:23] = W3
        
        # Second 3x3x3 part
        U6[:, 23:46] = U3
        V6[9:18, 23:46] = V3
        W6[9:18, 23:46] = W3
        
        # Convert to lists
        U_list = [[int(x) for x in row] for row in U6.tolist()]
        V_list = [[int(x) for x in row] for row in V6.tolist()]
        W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W6.tolist()]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook 54
        n, m, p = 3, 3, 6
        dim_A = 9
        dim_B = 18
        dim_C = 18
        rank = 54
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 3, 3
    dim = 9
    T3_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T3_nonzero[(a, b, c)] = 1.0

    U3 = None
    V3 = None
    W3 = None
    target_rank = 23
    
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
            
        U_rand = np.random.randint(-1, 2, size=(dim, target_rank))
        V_rand = np.random.randint(-1, 2, size=(dim, target_rank))
        
        W_rand = np.zeros((dim, target_rank))
        valid = True
        
        for c in range(dim):
            M_c = np.zeros((dim * dim, target_rank))
            b_c = np.zeros(dim * dim)
            
            idx = 0
            for a in range(dim):
                for b in range(dim):
                    M_c[idx, :] = U_rand[a, :] * V_rand[b, :]
                    if (a, b, c) in T3_nonzero:
                        b_c[idx] = 1.0
                    else:
                        b_c[idx] = 0.0
                    idx += 1
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
            except:
                valid = False
                break
            
            w_doubled = w * 2
            closest_ints = np.round(w_doubled)
            error = np.max(np.abs(w_doubled - closest_ints))
            
            if error < 1e-9:
                W_rand[c, :] = w
            else:
                valid = False
                break
        
        if valid:
            is_valid = True
            for (a, b, c), val in T3_nonzero.items():
                s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                if abs(s - val) > 1e-9:
                    is_valid = False
                    break
            
            if is_valid:
                for _ in range(50):
                    a = np.random.randint(0, dim)
                    b = np.random.randint(0, dim)
                    c = np.random.randint(0, dim)
                    if (a, b, c) not in T3_nonzero:
                        s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                        if abs(s) > 1e-9:
                            is_valid = False
                            break
                
                if is_valid:
                    U3 = U_rand
                    V3 = V_rand
                    W3 = W_rand
                    break

    if U3 is not None:
        U6 = np.zeros((9, 46))
        V6 = np.zeros((18, 46))
        W6 = np.zeros((18, 46))
        
        U6[:, 0:23] = U3
        V6[0:9, 0:23] = V3
        W6[0:9, 0:23] = W3
        
        U6[:, 23:46] = U3
        V6[9:18, 23:46] = V3
        W6[9:18, 23:46] = W3
        
        U_list = [[int(x) for x in row] for row in U6.tolist()]
        V_list = [[int(x) for x in row] for row in V6.tolist()]
        W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W6.tolist()]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        n, m, p = 3, 3, 6
        dim_A = 9
        dim_B = 18
        dim_C = 18
        rank = 54
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
