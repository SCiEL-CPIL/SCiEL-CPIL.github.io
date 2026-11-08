import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Returns a rank-minimal (for the known Strassen-based approach) tensor decomposition
    for multiplying a 4x4 matrix by a 4x9 matrix.
    
    The decomposition is constructed by splitting the 4x9 matrix B into two 4x4 blocks 
    and one 4x1 block.
    - The two 4x4 blocks are multiplied using the Strassen algorithm (rank 49 each).
    - The 4x1 block is multiplied using the schoolbook algorithm (rank 16).
    - Total rank: 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, with 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen terms for 2x2 * 2x2
    # u: coeffs for A, v: coeffs for B, w: coeffs for C
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # m1 = (a+d)(e+h)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # m2 = (c+d)e
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # m3 = a(f-h)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # m4 = d(g-e)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # m5 = (a+b)h
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # m6 = (c-a)(e+f)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # m7 = (b-d)(g+h)
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 49 terms for the Strassen decomposition of a 4x4 * 4x4 block.
        """
        if time.time() - start_time > DEADLINE:
            return
            
        # Helper to get global index in U (size 16)
        def u_idx(i, j):
            return i * m + j
            
        # Helper to get global index in V (size 36)
        def v_idx(j, k):
            return j * p + k
            
        # Helper to get global index in W (size 36)
        def w_idx(k, i):
            return k * n + i
            
        # Define the 7 block-level Strassen multiplications
        # Each is defined by:
        # A_side: list of (block_row, block_col, coeff)
        # B_side: list of (block_row, block_col, coeff)
        # C_side: dict { (block_row, block_col): coeff }
        
        block_mults = [
            # M1 = (A11+A22)(B11+B22) -> C11, C22
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
            # M2 = (A21+A22)B11 -> C21, -C22
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
            # M3 = A11(B12-B22) -> C12, C22
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
            # M4 = A22(B21-B11) -> C11, C21
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
            # M5 = (A11+A12)B22 -> -C11, C12
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
            # M6 = (A21-A11)(B11+B12) -> C22
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
            # M7 = (A12-A22)(B21+B22) -> C11
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
        ]
        
        for A_side, B_side, C_side in block_mults:
            # For each of the 7 meta-Strassen terms
            for u_meta, v_meta, w_meta in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Construct U vector
                # u_meta keys are local 2x2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                for local_idx, meta_coeff in u_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for br, bc, block_coeff in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[u_idx(gi, gj)] += meta_coeff * block_coeff
                        
                # Construct V vector
                for local_idx, meta_coeff in v_meta.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, block_coeff in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[v_idx(gj, gk)] += meta_coeff * block_coeff
                        
                # Construct W vector
                for local_idx, meta_coeff in w_meta.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[w_idx(gk, gi)] += meta_coeff * block_coeff
                        
                add_term(u_vec, v_vec, w_vec)

    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8) - Schoolbook
    for i in range(4):
        if time.time() - start_time > DEADLINE:
            break
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    # Build final matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
