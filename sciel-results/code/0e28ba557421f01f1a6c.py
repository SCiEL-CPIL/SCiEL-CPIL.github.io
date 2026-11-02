import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition of rank 43.
    # U: 10 x 43
    # V: 25 x 43
    # W: 10 x 43
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a local decomposition
    def add_decomp(U_loc, V_loc, W_loc, i_off, j_off, k_off, start_col):
        nonlocal col
        n_l = len(W_loc) // 2 # Wait, W_loc shape is (p_l * n_l, r_l). n_l is global n? No, local n.
        # We need to know local n, m, p to map indices.
        # U_loc shape: (n_l * m_l, r_l)
        # V_loc shape: (m_l * p_l, r_l)
        # W_loc shape: (p_l * n_l, r_l)
        # We can infer n_l, m_l, p_l from dimensions?
        # Better to pass them explicitly.
        pass

    # We will hardcode the blocks.
    
    # Block 1: M_{2,1,5} for j=0
    # i in 0..1, j=0, k in 0..4
    for i in range(2):
        for k in range(5):
            U[i * 5 + 0][col] = 1
            V[0 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    # col is now 5
    
    # Block 2: M_{2,4,5} for j in 1..4
    # Split into two M_{2,2,5}
    # Block 2a: j in 1..2
    # Block 2b: j in 3..4
    
    def add_m225(j_start, k_start, c_start):
        # Adds M_{2,2,5} decomposition starting at j_start, k_start, c_start
        # Returns new c_start
        # M_{2,2,5} = M_{2,2,2} (k in 0..1) + M_{2,2,3} (k in 2..4)
        
        # M_{2,2,2} Strassen
        # Local i in 0..1, j in 0..1, k in 0..1
        # Global I = i, J = j_start + j, K = k_start + k
        
        # Strassen terms
        # 1: A[0,0](B[0,0]+B[1,1])
        U[0 * 5 + (j_start + 0)][c_start] = 1
        V[(j_start + 0) * 5 + (k_start + 0)][c_start] = 1
        V[(j_start + 1) * 5 + (k_start + 1)][c_start] = 1
        W[(k_start + 0) * 2 + 0][c_start] = 1
        W[(k_start + 1) * 2 + 0][c_start] = 1
        W[(k_start + 0) * 2 + 1][c_start] = 1
        W[(k_start + 1) * 2 + 1][c_start] = 1
        # C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1 ?
        # Strassen:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Let's map Strassen M's to W contributions.
        # M1: A[0,0] * (B[0,0] + B[1,1])
        # Contribution to C:
        # C[0,0] += M1
        # C[0,1] += 0
        # C[1,0] += 0
        # C[1,1] += M1
        # So W[0,0] += 1, W[1,1] += 1.
        # Wait, W index is k*n + i.
        # C[0,0] -> k=0, i=0 -> W[0]
        # C[0,1] -> k=1, i=0 -> W[2]
        # C[1,0] -> k=0, i=1 -> W[1]
        # C[1,1] -> k=1, i=1 -> W[3]
        
        # M1:
        # U[0, j_start] = 1
        # V[j_start, k_start] = 1, V[j_start+1, k_start+1] = 1
        # W[k_start, 0] += 1, W[k_start+1, 1] += 1
        
        # M2: A[1,0] * (B[0,0] + B[0,1])
        # U[1, j_start] = 1
        # V[j_start, k_start] = 1, V[j_start, k_start+1] = 1
        # C[1,0] += M2 -> W[k_start, 1] += 1
        # C[1,1] -= M2 -> W[k_start+1, 1] -= 1
        
        # M3: A[0,1] * (B[1,0] - B[1,1])
        # U[0, j_start+1] = 1
        # V[j_start+1, k_start] = 1, V[j_start+1, k_start+1] = -1
        # C[0,1] += M3 -> W[k_start+1, 0] += 1
        # C[1,1] += M3 -> W[k_start+1, 1] += 1
        
        # M4: A[1,1] * (B[1,0] + B[1,1])
        # U[1, j_start+1] = 1
        # V[j_start+1, k_start] = 1, V[j_start+1, k_start+1] = 1
        # C[0,0] += M4 -> W[k_start, 0] += 1
        # C[1,0] += M4 -> W[k_start, 1] += 1
        # C[1,1] += M4 -> W[k_start+1, 1] += 1
        
        # M5: (A[0,0] + A[1,1]) * (B[0,1] - B[1,0])
        # U[0, j_start] = 1, U[1, j_start+1] = 1
        # V[j_start, k_start+1] = 1, V[j_start+1, k_start] = -1
        # C[0,0] -= M5 -> W[k_start, 0] -= 1
        # C[0,1] += M5 -> W[k_start+1, 0] += 1
        # C[1,1] += M5 -> W[k_start+1, 1] += 1
        
        # M6: (A[0,1] + A[1,1]) * (B[0,0] - B[0,1])
        # U[0, j_start+1] = 1, U[1, j_start+1] = 1
        # V[j_start, k_start] = 1, V[j_start, k_start+1] = -1
        # C[1,1] += M6 -> W[k_start+1, 1] += 1
        
        # M7: (A[0,0] + A[0,1]) * (B[1,1] - B[1,0])
        # U[0, j_start] = 1, U[0, j_start+1] = 1
        # V[j_start+1, k_start+1] = 1, V[j_start+1, k_start] = -1
        # C[0,0] += M7 -> W[k_start, 0] += 1
        # C[1,1] += M7 -> W[k_start+1, 1] += 1
        
        # We need to accumulate these into U, V, W.
        # Since we are writing to specific columns, we can just add.
        
        # M1
        U[0 * 5 + (j_start + 0)][c_start] += 1
        V[(j_start + 0) * 5 + (k_start + 0)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 1)][c_start] += 1
        W[(k_start + 0) * 2 + 0][c_start] += 1
        W[(k_start + 1) * 2 + 1][c_start] += 1
        c_start += 1
        
        # M2
        U[1 * 5 + (j_start + 0)][c_start] += 1
        V[(j_start + 0) * 5 + (k_start + 0)][c_start] += 1
        V[(j_start + 0) * 5 + (k_start + 1)][c_start] += 1
        W[(k_start + 0) * 2 + 1][c_start] += 1
        W[(k_start + 1) * 2 + 1][c_start] -= 1
        c_start += 1
        
        # M3
        U[0 * 5 + (j_start + 1)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 0)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 1)][c_start] -= 1
        W[(k_start + 1) * 2 + 0][c_start] += 1
        W[(k_start + 1) * 2 + 1][c_start] += 1
        c_start += 1
        
        # M4
        U[1 * 5 + (j_start + 1)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 0)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 1)][c_start] += 1
        W[(k_start + 0) * 2 + 0][c_start] += 1
        W[(k_start + 0) * 2 + 1][c_start] += 1
        W[(k_start + 1) * 2 + 1][c_start] += 1
        c_start += 1
        
        # M5
        U[0 * 5 + (j_start + 0)][c_start] += 1
        U[1 * 5 + (j_start + 1)][c_start] += 1
        V[(j_start + 0) * 5 + (k_start + 1)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 0)][c_start] -= 1
        W[(k_start + 0) * 2 + 0][c_start] -= 1
        W[(k_start + 1) * 2 + 0][c_start] += 1
        W[(k_start + 1) * 2 + 1][c_start] += 1
        c_start += 1
        
        # M6
        U[0 * 5 + (j_start + 1)][c_start] += 1
        U[1 * 5 + (j_start + 1)][c_start] += 1
        V[(j_start + 0) * 5 + (k_start + 0)][c_start] += 1
        V[(j_start + 0) * 5 + (k_start + 1)][c_start] -= 1
        W[(k_start + 1) * 2 + 1][c_start] += 1
        c_start += 1
        
        # M7
        U[0 * 5 + (j_start + 0)][c_start] += 1
        U[0 * 5 + (j_start + 1)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 1)][c_start] += 1
        V[(j_start + 1) * 5 + (k_start + 0)][c_start] -= 1
        W[(k_start + 0) * 2 + 0][c_start] += 1
        W[(k_start + 1) * 2 + 1][c_start] += 1
        c_start += 1
        
        # M_{2,2,3} Schoolbook
        # i in 0..1, j in 0..1, k in 2..4
        for i in range(2):
            for j in range(2):
                for k in range(2, 5):
                    U[i * 5 + (j_start + j)][c_start] += 1
                    V[(j_start + j) * 5 + (k_start + k)][c_start] += 1
                    W[(k_start + k) * 2 + i][c_start] += 1
                    c_start += 1
                    
        return c_start

    col = add_m225(1, 0, col)
    col = add_m225(3, 0, col)
    
    return {"U": U, "V": V, "W": W}
