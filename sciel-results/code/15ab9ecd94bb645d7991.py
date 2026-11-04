import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    # Pre-compute indices for the 60 tensor entries
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    idx_u = np.zeros(n * m * p, dtype=np.int32)
    idx_v = np.zeros(n * m * p, dtype=np.int32)
    idx_w = np.zeros(n * m * p, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u[idx] = i * m + j
                idx_v[idx] = j * p + k
                idx_w[idx] = k * n + i
                idx += 1

    def check_decomposition_fast(U, V, W):
        """
        Vectorized check using pre-computed indices.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        vals = U[idx_u] * V[idx_v] * W[idx_w]
        sums = vals.sum(axis=1)
        return np.all(sums == 1.0)

    def get_strassen_54():
        """
        Construct the rank 54 decomposition using Strassen for 2x2 blocks.
        Partition m=5 into {0,1}, {2,3}, {4}.
        Partition p=6 into {0,1}, {2,3}, {4,5}.
        n=2 is handled by the 2x2 Strassen blocks.
        """
        rank = 54
        U = np.zeros((nm, rank), dtype=np.int8)
        V = np.zeros((mp, rank), dtype=np.int8)
        W = np.zeros((pn, rank), dtype=np.int8)
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            # 1: A00 B00 -> C00
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            # 2: A00 B01 -> C01
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # 3: A10 B00 -> C10
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # 4: A10 B01 -> C11
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        # Apply Strassen blocks for j pairs (0,1) and (2,3)
        # j_base = 0, k_base in {0, 2, 4}
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        
        # j_base = 2, k_base in {0, 2, 4}
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        # Apply naive blocks for j=4
        # j_base = 4, k_base in {0, 2, 4}
        add_naive_2112_block(0, 4, 0)
        add_naive_2112_block(0, 4, 2)
        add_naive_2112_block(0, 4, 4)
        
        if col != rank:
             # Fallback to schoolbook if construction failed
             rank_fb = n * m * p
             U_fb = np.zeros((nm, rank_fb))
             V_fb = np.zeros((mp, rank_fb))
             W_fb = np.zeros((pn, rank_fb))
             c = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U_fb[i * m + j, c] = 1
                         V_fb[j * p + k, c] = 1
                         W_fb[k * n + i, c] = 1
                         c += 1
             return U_fb, V_fb, W_fb
             
        return U, V, W

    # Get the deterministic rank 54 solution
    U_base, V_base, W_base = get_strassen_54()
    
    # Verify it
    if check_decomposition_fast(U_base, V_base, W_base):
        best_U, best_V, best_W = U_base, V_base, W_base
        best_rank = 54
    else:
        # Fallback to schoolbook if Strassen construction is buggy
        rank_fb = n * m * p
        U_fb = np.zeros((nm, rank_fb))
        V_fb = np.zeros((mp, rank_fb))
        W_fb = np.zeros((pn, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        best_U, best_V, best_W = U_fb, V_fb, W_fb
        best_rank = 60

    # Try to find a lower rank using random search
    # We search for ranks below the current best
    for R in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 2:
            break
            
        # Time slice per rank
        start_rank_time = time.time()
        time_per_rank = 1.0
        
        while time.time() - start_rank_time < time_per_rank:
            # Generate random integer matrices
            # Using small integers {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            W = np.random.randint(-1, 2, size=(pn, R))
            
            if check_decomposition_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                # If we find a lower rank, we should probably continue searching for even lower?
                # Or just break? The task asks for rank-minimal.
                # Given the difficulty, finding one is a big jump.
                # We can continue to see if we can go lower.
                break
        
        if time.time() > deadline - 5:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
