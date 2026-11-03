import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build the columns one by one
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_start, j_start, k_start):
        """
        Adds a Strassen decomposition for a 2x2x2 block starting at (i_start, j_start, k_start).
        i_start, j_start, k_start are the minimum indices for the block.
        The block uses i in {i_start, i_start+1}, j in {j_start, j_start+1}, k in {k_start, k_start+1}.
        """
        # Local indices 0,1 map to global:
        # i0 = i_start, i1 = i_start+1
        # j0 = j_start, j1 = j_start+1
        # k0 = k_start, k1 = k_start+1
        
        # U indices (row = i*m + j)
        u_i0_j0 = i_start * m + j_start
        u_i0_j1 = i_start * m + (j_start + 1)
        u_i1_j0 = (i_start + 1) * m + j_start
        u_i1_j1 = (i_start + 1) * m + (j_start + 1)
        
        # V indices (row = j*p + k)
        v_j0_k0 = j_start * p + k_start
        v_j0_k1 = j_start * p + (k_start + 1)
        v_j1_k0 = (j_start + 1) * p + k_start
        v_j1_k1 = (j_start + 1) * p + (k_start + 1)
        
        # W indices (row = k*n + i)
        w_k0_i0 = k_start * n + i_start
        w_k0_i1 = k_start * n + (i_start + 1)
        w_k1_i0 = (k_start + 1) * n + i_start
        w_k1_i1 = (k_start + 1) * n + (i_start + 1)
        
        # Strassen Multiplications
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B11 + B01)
        
        # Results
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 columns
        
        # 1
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i0_j0] = 1; U[u_i1_j1] = 1
        V[v_j0_k0] = 1; V[v_j1_k1] = 1
        W[w_k0_i0] = 1; W[w_k1_i1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # 2
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i1_j0] = 1; U[u_i1_j1] = 1
        V[v_j0_k0] = 1
        W[w_k0_i1] = 1; W[w_k1_i1] = -1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # 3
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i0_j0] = 1
        V[v_j1_k0] = 1; V[v_j1_k1] = -1
        W[w_k0_i0] = 1; W[w_k1_i0] = 1 # Wait, C01 and C11?
        # Let's re-verify W indices for M3
        # M3 contributes to C01 and C11.
        # C01 is i=i_start, k=k_start+1 -> w_k1_i0
        # C11 is i=i_start+1, k=k_start+1 -> w_k1_i1
        # My previous W assignment: W[w_k0_i0] and W[w_k1_i0] was wrong.
        # Correct:
        W = [0] * (p * n)
        W[w_k1_i0] = 1; W[w_k1_i1] = 1
        U_cols[-1] = U; V_cols[-1] = V; W_cols[-1] = W
        
        # 4
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i1_j1] = 1
        V[v_j0_k1] = 1; V[v_j0_k0] = -1
        # M4 contributes to C00 and C10.
        # C00: i=i_start, k=k_start -> w_k0_i0
        # C10: i=i_start+1, k=k_start -> w_k0_i1
        W[w_k0_i0] = 1; W[w_k0_i1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # 5
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i0_j0] = 1; U[u_i0_j1] = 1
        V[v_j1_k1] = 1
        # M5 contributes to -C00 and C01.
        # C00: w_k0_i0
        # C01: w_k1_i0
        W[w_k0_i0] = -1; W[w_k1_i0] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # 6
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i1_j0] = 1; U[u_i0_j0] = -1
        V[v_j0_k0] = 1; V[v_j1_k0] = 1
        # M6 contributes to C11.
        # C11: w_k1_i1
        W[w_k1_i1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # 7
        U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
        U[u_i0_j1] = 1; U[u_i1_j1] = -1
        V[v_j1_k1] = 1; V[v_j0_k1] = 1
        # M7 contributes to C00.
        # C00: w_k0_i0
        W[w_k0_i0] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    # Add Block 1: i=0, j=0, k=0
    add_strassen_block(0, 0, 0)
    
    # Add Block 2: i=1, j=2, k=2
    add_strassen_block(1, 2, 2)
    
    # Add Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if in Block 1
                if 0 <= i <= 1 and 0 <= j <= 1 and 0 <= k <= 1:
                    continue
                # Check if in Block 2
                if 1 <= i <= 2 and 2 <= j <= 3 and 2 <= k <= 3:
                    continue
                
                U = [0] * (n * m); V = [0] * (m * p); W = [0] * (p * n)
                U[i * m + j] = 1
                V[j * p + k] = 1
                W[k * n + i] = 1
                U_cols.append(U); V_cols.append(V); W_cols.append(W)
    
    rank = len(U_cols)
    
    # Convert to list of lists (rows)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            if U_cols[c][r] != 0:
                U[r][c] = U_cols[c][r]
        for r in range(m * p):
            if V_cols[c][r] != 0:
                V[r][c] = V_cols[c][r]
        for r in range(p * n):
            if W_cols[c][r] != 0:
                W[r][c] = W_cols[c][r]
                
    return {"U": U, "V": V, "W": W}
