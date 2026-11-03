"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition as baseline (rank 90)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Winograd-style block decomposition
    # Partitions:
    # i (rows of A, rows of C): 3 -> {0,1}, {2}
    # j (cols of A, rows of B): 5 -> {0,1}, {2,3}, {4}
    # k (cols of B, cols of C): 6 -> {0,1}, {2,3}, {4,5}
    
    i_blocks = [
        [0, 1],
        [2]
    ]
    j_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    k_blocks = [
        [0, 1],
        [2, 3],
        [4, 5]
    ]
    
    def strassen_222_terms(i0, i1, j0, j1, k0, k1):
        """
        Returns 7 rank-1 tensor terms for the 2x2x2 matrix multiplication tensor.
        Each term is (u_dict, v_dict, w_dict) where:
          u_dict maps (i,j) -> coeff
          v_dict maps (j,k) -> coeff
          w_dict maps (k,i) -> coeff
        such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k in {0,1}.
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = { (i0,j0): 1, (i1,j1): 1 }
        v1 = { (j0,k0): 1, (j1,k1): 1 }
        w1 = { (k0,i0): 1, (k1,i1): 1 }
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = { (i1,j0): 1, (i1,j1): 1 }
        v2 = { (j0,k0): 1 }
        w2 = { (k0,i1): 1, (k1,i1): -1 }
        terms.append((u2, v2, w2))
        
        # M3 = A00(B10-B11)
        u3 = { (i0,j0): 1 }
        v3 = { (j1,k0): 1, (j1,k1): -1 }
        w3 = { (k0,i0): 1, (k1,i0): -1 }
        terms.append((u3, v3, w3))
        
        # M4 = A11(B01-B00)
        u4 = { (i1,j1): 1 }
        v4 = { (j0,k1): 1, (j0,k0): -1 }
        w4 = { (k1,i1): 1, (k0,i1): -1 }
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = { (i0,j0): 1, (i0,j1): 1 }
        v5 = { (j1,k1): 1 }
        w5 = { (k1,i0): 1, (k1,i1): 1 }
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = { (i1,j0): 1, (i0,j0): -1 }
        v6 = { (j0,k0): 1, (j0,k1): 1 }
        w6 = { (k0,i1): 1, (k1,i1): 1, (k0,i0): -1, (k1,i0): -1 }
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = { (i0,j1): 1, (i1,j1): -1 }
        v7 = { (j1,k0): 1, (j1,k1): 1 }
        w7 = { (k0,i0): 1, (k1,i0): 1, (k0,i1): -1, (k1,i1): -1 }
        terms.append((u7, v7, w7))
        
        return terms

    # Accumulate all terms
    all_terms = []
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                i_size = len(i_blk)
                j_size = len(j_blk)
                k_size = len(k_blk)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    terms = strassen_222_terms(i0, i1, j0, j1, k0, k1)
                    all_terms.extend(terms)
                else:
                    # Schoolbook
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                u = { (i,j): 1 }
                                v = { (j,k): 1 }
                                w = { (k,i): 1 }
                                all_terms.append((u, v, w))
    
    # Now build U, V, W matrices
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verify
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if is_valid:
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using random search with structured initialization
    # We'll try various ranks and see if we can find a valid decomposition
    
    def try_random_decomposition(rank, time_budget=10.0):
        """Try to find a random decomposition of given rank."""
        start_time = time.time()
        np.random.seed(42)
        
        for attempt in range(100):
            if time.time() - start_time > time_budget or time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U_rand = np.random.randn(n * m, rank) * 0.1
            V_rand = np.random.randn(m * p, rank) * 0.1
            W_rand = np.random.randn(p * n, rank) * 0.1
            
            # Try to optimize
            for step in range(50):
                if time.time() - start_time > time_budget or time.time() > deadline - 5:
                    break
                
                # Compute gradient of loss = sum of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
                # This is complex, so we'll use a simple heuristic: try to make the product close to 1
                
                # Check current error
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(rank):
                                s += U_rand[i * m + j, c] * V_rand[j * p + k, c] * W_rand[k * n + i, c]
                            err = abs(s - 1)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    # Found a valid decomposition!
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    
                    # Round to nearest 0.5 if close
                    for i in range(len(U_list)):
                        for j in range(len(U_list[i])):
                            val = U_list[i][j]
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) < 1e-3:
                                U_list[i][j] = nearest
                    for i in range(len(V_list)):
                        for j in range(len(V_list[i])):
                            val = V_list[i][j]
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) < 1e-3:
                                V_list[i][j] = nearest
                    for i in range(len(W_list)):
                        for j in range(len(W_list[i])):
                            val = W_list[i][j]
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) < 1e-3:
                                W_list[i][j] = nearest
                    
                    # Verify after rounding
                    is_valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(rank):
                                    s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                                if abs(s - 1) > 1e-6:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # Simple gradient-free update
                U_rand += np.random.randn(n * m, rank) * 0.01
                V_rand += np.random.randn(m * p, rank) * 0.01
                W_rand += np.random.randn(p * n, rank) * 0.01
        
        return None
    
    # Try ranks from 70 down to 60
    for rank in range(70, 59, -1):
        if time.time() > deadline - 10:
            break
        result = try_random_decomposition(rank, time_budget=5.0)
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    return best_result

# EVOLVE-BLOCK-END
