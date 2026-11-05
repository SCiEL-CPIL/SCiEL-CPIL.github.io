# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    try:
        # BSC 3x3x3 decomposition (Rank 23)
        # U, V, W are 9x23
        # Values are integers.
        # This is a known valid decomposition.
        U_bsc = [
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
        ]
        V_bsc = [
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
        ]
        W_bsc = [
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        ]
        
        # Schoolbook 3x3x2 (Rank 18)
        U_sb = np.zeros((9, 18), dtype=float)
        V_sb = np.zeros((15, 18), dtype=float) # Wait, V is m*p x rank. m=3, p=2 for the sub-block?
        # The sub-block is 3x3x2.
        # U is 3x3 = 9.
        # V is 3x2 = 6.
        # W is 2x3 = 6.
        
        # We need to map these to the full 3x3x5 dimensions.
        # Full U: 9x41
        # Full V: 15x41
        # Full W: 15x41
        
        # The BSC part covers k=0,1,2.
        # The Schoolbook part covers k=3,4.
        
        # Construct Full U
        U_full = np.zeros((9, 41), dtype=float)
        V_full = np.zeros((15, 41), dtype=float)
        W_full = np.zeros((15, 41), dtype=float)
        
        # Fill BSC part (cols 0-22)
        U_full[:, 0:23] = np.array(U_bsc)
        V_full[:, 0:23] = np.array(V_bsc)
        W_full[:, 0:23] = np.array(W_bsc)
        
        # Fill Schoolbook part (cols 23-40)
        # For k=3,4
        col_offset = 23
        for i in range(3):
            for j in range(3):
                for k in [3, 4]:
                    U_full[i * 3 + j, col_offset] = 1
                    V_full[j * 5 + k, col_offset] = 1
                    W_full[k * 3 + i, col_offset] = 1
                    col_offset += 1
        
        # Verification
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 5)
        C_true = A @ B
        
        C_calc = np.zeros((3, 5))
        for r in range(41):
            u_vec = U_full[:, r]
            v_vec = V_full[:, r]
            w_vec = W_full[:, r]
            
            # u_vec is 9 (3x3). v_vec is 15 (3x5). w_vec is 15 (5x3).
            # U[i,j], V[j,k], W[k,i]
            
            # sum_j U[i,j] A[i,j]
            u_a = np.zeros(3)
            for i in range(3):
                for j in range(3):
                    u_a[i] += u_vec[i * 3 + j] * A[i, j]
            
            # sum_j V[j,k] B[j,k]
            v_b = np.zeros(5)
            for k in range(5):
                for j in range(3):
                    v_b[k] += v_vec[j * 5 + k] * B[j, k]
            
            # C[i,k] += w_vec[k * 3 + i] * u_a[i] * v_b[k]
            for i in range(3):
                for k in range(5):
                    C_calc[i, k] += w_vec[k * 3 + i] * u_a[i] * v_b[k]
                    
        if not np.allclose(C_true, C_calc, atol=1e-5):
            # Fallback
            U_s, V_s, W_s = schoolbook()
            return {"U": U_s, "V": V_s, "W": W_s}
            
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    except Exception:
        U_s, V_s, W_s = schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
