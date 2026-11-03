"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    deadline = time.time() + 120.0
    
    # --- 1. Define Schoolbook Decomposition ---
    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # --- 2. Define Strassen 2x2x2 Decomposition ---
    def strassen_222_decomp():
        # Rank 7 decomposition for 2x2x2 matrix multiplication
        # U: 4x7, V: 4x7, W: 4x7
        # Indices:
        # U: row = i*2 + j (i,j in 0..1)
        # V: row = j*2 + k (j,k in 0..1)
        # W: row = k*2 + i (k,i in 0..1) -> Note: W[k*n+i] with n=2
        
        # Standard Strassen factors:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        U = [
            [1, 0, 1, 0, 1, -1, 1], # A11
            [0, 0, 0, 1, 0, 0, 1], # A12
            [0, 1, 0, 0, 0, 1, 0], # A21
            [1, 1, 0, 1, 0, 0, -1] # A22
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0], # B11
            [0, 0, 1, 1, 1, 0, 0], # B12
            [1, 1, -1, 0, 0, 0, 1], # B21
            [1, 0, 0, 0, 1, 0, 1]  # B22
        ]
        W = [
            # k=0, i=0 (C11): M1 + M4 - M5 - M7
            [1, 0, 0, 1, -1, 0, -1],
            # k=0, i=1 (C21): M2 + M4
            [0, 1, 0, 1, 0, 0, 0],
            # k=1, i=0 (C12): M3 + M5
            [0, 0, 1, 0, 1, 0, 0],
            # k=1, i=1 (C22): M1 - M2 + M3 + M7
            [1, -1, 1, 0, 0, 0, 1]
        ]
        return U, V, W

    # --- 3. Helper to embed and combine decompositions ---
    def combine_decomps(decomps):
        """
        Combine a list of decompositions.
        Each decomposition is (n, m, p, U, V, W, offset_i, offset_j, offset_k)
        Returns global U, V, W.
        """
        if not decomps:
            return [], [], []
        
        global_n = max(n for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        global_m = max(m for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        global_p = max(p for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        
        # Determine total rank
        total_rank = sum(len(U) for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        
        G_U = [[0] * total_rank for _ in range(global_n * global_m)]
        G_V = [[0] * total_rank for _ in range(global_m * global_p)]
        G_W = [[0] * total_rank for _ in range(global_p * global_n)]
        
        rank_offset = 0
        for (n, m, p, U, V, W, oi, oj, ok) in decomps:
            for c in range(len(U)):
                r_c = rank_offset + c
                # U mapping: local row (i*m + j) -> global row ((oi+i)*global_m + (oj+j))
                for i in range(n):
                    for j in range(m):
                        local_u_idx = i * m + j
                        global_u_idx = (oi + i) * global_m + (oj + j)
                        G_U[global_u_idx][r_c] = U[local_u_idx][c]
                        
                # V mapping: local row (j*p + k) -> global row ((oj+j)*global_p + (ok+k))
                for j in range(m):
                    for k in range(p):
                        local_v_idx = j * p + k
                        global_v_idx = (oj + j) * global_p + (ok + k)
                        G_V[global_v_idx][r_c] = V[local_v_idx][c]
                        
                # W mapping: local row (k*n + i) -> global row ((ok+k)*global_n + (oi+i))
                for k in range(p):
                    for i in range(n):
                        local_w_idx = k * n + i
                        global_w_idx = (ok + k) * global_n + (oi + i)
                        G_W[global_w_idx][r_c] = W[local_w_idx][c]
                        
            rank_offset += len(U)
            
        return G_U, G_V, G_W

    # --- 4. Strategy: Block Decomposition ---
    # Dimensions: 3 x 4 x 7
    # Split n=3 into 2+1
    # Split m=4 into 2+2
    # Split p=7 into 2+2+2+1
    
    # We will iterate over all combinations of these blocks.
    # For each block, if it's 2x2x2, use Strassen (rank 7).
    # If it's smaller, use schoolbook.
    
    blocks = []
    
    # n splits: [0:2], [2:3] -> (0, 2), (2, 1)
    n_splits = [(0, 2), (2, 1)]
    # m splits: [0:2], [2:4] -> (0, 2), (2, 2)
    m_splits = [(0, 2), (2, 2)]
    # p splits: [0:2], [2:4], [4:6], [6:7] -> (0,2), (2,2), (4,2), (6,1)
    p_splits = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    decomps = []
    
    for (oi, ni) in n_splits:
        for (oj, mj) in m_splits:
            for (ok, pk) in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    U, V, W = strassen_222_decomp()
                else:
                    U, V, W = schoolbook_decomp(ni, mj, pk)
                
                decomps.append((ni, mj, pk, U, V, W, oi, oj, ok))
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    if time.time() > deadline - 10:
        break
    
    U, V, W = combine_decomps(decomps)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
