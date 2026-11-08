import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # 12 terms for A[:,0] * B[0,:]
    # 21 terms for A[:,1:3] * B[1:3,:] (3 blocks of 2x2x2 Strassen)
    
    R = 33
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # C[i, k] += A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # Split k into 3 groups of 2
    for l in range(3):
        k_start = 2 * l
        k_end = 2 * l + 2
        
        # Strassen for 2x2x2
        # A2 = A[:, 1:3] (2x2)
        # B2_l = B[1:3, k_start:k_end] (2x2)
        
        # Map A2 indices to local 0,1
        # A2[0,0] -> A[0,1], A2[0,1] -> A[0,2]
        # A2[1,0] -> A[1,1], A2[1,1] -> A[1,2]
        
        # Map B2_l indices to local 0,1
        # B2_l[0,0] -> B[1, k_start], B2_l[0,1] -> B[1, k_start+1]
        # B2_l[1,0] -> B[2, k_start], B2_l[1,1] -> B[2, k_start+1]
        
        # Strassen Multiplications:
        # M1 = (A2[0,0] + A2[1,1]) * (B2_l[0,0] + B2_l[1,1])
        # M2 = (A2[1,0] + A2[1,1]) * B2_l[0,0]
        # M3 = A2[0,0] * (B2_l[0,1] - B2_l[1,1])
        # M4 = A2[1,1] * (B2_l[1,0] - B2_l[0,0])
        # M5 = (A2[0,0] + A2[0,1]) * B2_l[1,1]
        # M6 = (A2[0,1] - A2[0,0]) * (B2_l[0,0] + B2_l[1,0])
        # M7 = (A2[1,0] - A2[1,1]) * (B2_l[0,1] + B2_l[1,1])
        
        # We implement each M as a separate term in the tensor decomposition.
        # A term is defined by U, V, W vectors.
        # U has entries for A, V for B, W for C.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            # u_coeffs: dict mapping (i, j) to coeff
            # v_coeffs: dict mapping (j, k) to coeff
            # w_coeffs: dict mapping (k, i) to coeff
            
            for (i, j), c_u in u_coeffs.items():
                U[i * m + j][col] += c_u
            for (j, k), c_v in v_coeffs.items():
                V[j * p + k][col] += c_v
            for (k, i), c_w in w_coeffs.items():
                W[k * n + i][col] += c_w
            col += 1

        # M1
        add_strassen_term(
            {(0, 1): 1, (1, 2): 1},
            {(1, k_start): 1, (2, k_end-1): 1},
            {(k_start, 0): 1, (k_end-1, 1): 1} # Wait, W indices are (k, i)
        )
        # Correction: W index is k*n + i.
        # C2_l[0,0] is C[0, k_start]. W index: k_start*n + 0.
        # C2_l[0,1] is C[0, k_start+1]. W index: (k_start+1)*n + 0.
        # C2_l[1,0] is C[1, k_start]. W index: k_start*n + 1.
        # C2_l[1,1] is C[1, k_start+1]. W index: (k_start+1)*n + 1.
        
        # Let's redefine the W coefficients carefully for Strassen outputs.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add contributions to W for each M.
        
        # Reset col logic? No, we just add to the same col index.
        # The function add_strassen_term increments col.
        # So we call it 7 times.
        
        # M1
        # U: A[0,1] + A[1,2]
        # V: B[1, k_start] + B[2, k_start+1]
        # W: C[0, k_start] + C[1, k_start+1]
        add_strassen_term(
            {(0, 1): 1, (1, 2): 1},
            {(1, k_start): 1, (2, k_start+1): 1},
            {(k_start, 0): 1, (k_start+1, 1): 1}
        )
        
        # M2
        # U: A[1,1] + A[1,2]
        # V: B[1, k_start]
        # W: C[1, k_start]
        add_strassen_term(
            {(1, 1): 1, (1, 2): 1},
            {(1, k_start): 1},
            {(k_start, 1): 1}
        )
        
        # M3
        # U: A[0,1]
        # V: B[1, k_start+1] - B[2, k_start+1]
        # W: C[0, k_start+1]
        add_strassen_term(
            {(0, 1): 1},
            {(1, k_start+1): 1, (2, k_start+1): -1},
            {(k_start+1, 0): 1}
        )
        
        # M4
        # U: A[1,2]
        # V: B[2, k_start] - B[1, k_start]
        # W: C[0, k_start] + C[1, k_start]
        add_strassen_term(
            {(1, 2): 1},
            {(2, k_start): 1, (1, k_start): -1},
            {(k_start, 0): 1, (k_start, 1): 1}
        )
        
        # M5
        # U: A[0,1] + A[0,2]
        # V: B[2, k_start+1]
        # W: -C[0, k_start] + C[0, k_start+1]
        add_strassen_term(
            {(0, 1): 1, (0, 2): 1},
            {(2, k_start+1): 1},
            {(k_start, 0): -1, (k_start+1, 0): 1}
        )
        
        # M6
        # U: A[0,2] - A[0,1]
        # V: B[1, k_start] + B[2, k_start]
        # W: C[1, k_start+1]
        add_strassen_term(
            {(0, 2): 1, (0, 1): -1},
            {(1, k_start): 1, (2, k_start): 1},
            {(k_start+1, 1): 1}
        )
        
        # M7
        # U: A[1,1] - A[1,2]
        # V: B[1, k_start+1] + B[2, k_start+1]
        # W: C[0, k_start] + C[1, k_start+1]
        add_strassen_term(
            {(1, 1): 1, (1, 2): -1},
            {(1, k_start+1): 1, (2, k_start+1): 1},
            {(k_start, 0): 1, (k_start+1, 1): 1}
        )
        
    # Verify
    def verify(U, V, W, R, num_tests=5):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for _ in range(num_tests):
            A = np.random.randint(-2, 3, size=(n, m))
            B = np.random.randint(-2, 3, size=(m, p))
            
            C_direct = A @ B
            
            U_reshaped = U_arr.reshape(n, m, R)
            V_reshaped = V_arr.reshape(m, p, R)
            W_reshaped = W_arr.reshape(p, n, R)
            
            LA = np.einsum('ij,ijr->ir', A, U_reshaped)
            LB = np.einsum('jk,jkr->kr', B, V_reshaped)
            
            C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
            
            if not np.array_equal(C_direct, C_decomp):
                return False
                
        return True

    if verify(U, V, W, R, num_tests=10):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_sb[u_idx][col] = 1
                    V_sb[v_idx][col] = 1
                    W_sb[w_idx][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
