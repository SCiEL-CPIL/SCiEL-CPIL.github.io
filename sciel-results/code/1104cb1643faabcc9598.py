import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    
    def build_decomp(arrangement):
        """Build a Strassen-based decomposition with a given arrangement.
        arrangement: list of (type, j_base, k_base) tuples
        type: 'strassen' or 'naive2112' or 'naive211'
        """
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # m0: (A[i,j]+A[i+1,j+1])(B[j,k]+B[j+1,k+1])
            U[i_base * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            # m1: (A[i+1,j]+A[i+1,j+1])B[j,k]
            U[(i_base+1) * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] -= 1
            col += 1
            
            # m2: A[i,j](B[j,k+1]-B[j+1,k+1])
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            # m3: A[i+1,j+1](B[j+1,k]-B[j,k])
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            
            # m4: (A[i,j]+A[i,j+1])B[j+1,k+1]
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            
            # m5: (A[i+1,j]-A[i,j])(B[j,k]+B[j,k+1])
            U[(i_base+1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            # m6: (A[i,j+1]-A[i+1,j+1])(B[j+1,k]+B[j+1,k+1])
            U[i_base * m + (j_base+1)][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] -= 1
            V[(j_base+1) * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
        
        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            # 2x1 * 1x2 naive: 4 terms
            U[i_base * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
        
        def add_naive_211_block(i_base, j_base, k_base):
            nonlocal col
            # 2x1 * 1x1 naive: 2 terms
            U[i_base * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
        
        for (blk_type, j_base, k_base) in arrangement:
            if blk_type == 'strassen':
                add_strassen_block(0, j_base, k_base)
            elif blk_type == 'naive2112':
                add_naive_2112_block(0, j_base, k_base)
            elif blk_type == 'naive211':
                add_naive_211_block(0, j_base, k_base)
        
        return np.array(U[:n*m], dtype=float)[:, :col], np.array(V[:m*p], dtype=float)[:, :col], np.array(W[:p*n], dtype=float)[:, :col]
    
    # Try multiple arrangements - different groupings of the 5 columns of A/B
    # The key structural choice: which pairs of columns get Strassen (7 terms) vs naive (4 terms)
    # 5 columns: need to partition into groups of 2 (Strassen) and 1 (naive)
    # Possible: 2+2+1 (3 Strassen + 1 naive col) = 3*7 + 4 = 25 per k-pair
    # Or: 2+1+2 = same
    # Or: 1+2+2 = same
    # For k: 6 rows of B, partition into 3 pairs: (0,1),(2,3),(4,5)
    
    arrangements = []
    
    # Arrangement A: current program - Strassen on (0,1),(2,3), naive on col 4
    for k_pair in [(0,1), (2,3), (4,5)]:
        arrangements.append([
            ('strassen', 0, k_pair[0]),
            ('strassen', 2, k_pair[0]),
            ('naive2112', 4, k_pair[0]),
        ])
    
    # Arrangement B: Strassen on (0,1),(3,4), naive on col 2
    for k_pair in [(0,1), (2,3), (4,5)]:
        arrangements.append([
            ('strassen', 0, k_pair[0]),
            ('strassen', 3, k_pair[0]),
            ('naive2112', 2, k_pair[0]),
        ])
    
    # Arrangement C: Strassen on (2,3),(3,4), naive on col 0
    for k_pair in [(0,1), (2,3), (4,5)]:
        arrangements.append([
            ('strassen', 2, k_pair[0]),
            ('strassen', 3, k_pair[0]),
            ('naive2112', 0, k_pair[0]),
        ])
    
    # Arrangement D: Strassen on (0,1),(1,2), naive on col 3 (overlapping - skip)
    # Arrangement E: Strassen on (1,2),(3,4), naive on col 0
    for k_pair in [(0,1), (2,3), (4,5)]:
        arrangements.append([
            ('strassen', 1, k_pair[0]),
            ('strassen', 3, k_pair[0]),
            ('naive2112', 0, k_pair[0]),
        ])
    
    # Arrangement F: Strassen on (0,1),(2,3), naive on col 4 - different k order
    for k_pair in [(0,1), (2,3), (4,5)]:
        arrangements.append([
            ('strassen', 0, k_pair[0]),
            ('strassen', 2, k_pair[0]),
            ('naive2112', 4, k_pair[0]),
        ])
    
    # Try each arrangement, find the one with lowest rank after reduction
    best_result = None
    best_rank = 999
    
    deadline = START_TIME + TIME_LIMIT - 5.0
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        
        U_arr, V_arr, W_arr = build_decomp(arr)
        col = U_arr.shape[1]
        
        if col >= best_rank:
            continue
        
        # Build tensor coefficient matrix
        T_size = n * m * p
        M = np.zeros((T_size, col))
        for r in range(col):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        M[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        
        if rank_M < col:
            # Greedy elimination
            keep = list(range(col))
            
            for _pass in range(8):
                if time.time() > deadline:
                    break
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for idx in list(keep):
                        test_keep = [k for k in keep if k != idx]
                        if len(test_keep) == 0:
                            break
                        M_test = M[:, test_keep]
                        if np.linalg.matrix_rank(M_test, tol=1e-10) == rank_M:
                            keep = test_keep
                            changed = True
                            break
            
            new_rank = len(keep)
            
            if new_rank < best_rank:
                U_new = U_arr[:, keep]
                V_new = V_arr[:, keep]
                W_new = W_arr[:, keep]
                
                # Verify
                verified = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r] for r in range(new_rank))
                            if abs(val - 1.0) > 1e-9:
                                verified = False
                                break
                        if not verified:
                            break
                    if not verified:
                        break
                
                if verified:
                    best_rank = new_rank
                    best_result = (U_new, V_new, W_new)
        
        # Also check if original col is already minimal
        if col < best_rank and rank_M == col:
            # Verify original
            verified = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(col))
                        if abs(val - 1.0) > 1e-9:
                            verified = False
                            break
                    if not verified:
                        break
            if verified:
                best_rank = col
                best_result = (U_arr, V_arr, W_arr)
    
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                half = round(val * 2) / 2
                if abs(val - half) < 1e-8:
                    clean_row.append(half if half != int(half) else int(half))
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    if best_result is not None:
        U_new, V_new, W_new = best_result
        return {"U": to_clean_list(U_new), "V": to_clean_list(V_new), "W": to_clean_list(W_new)}
    
    # Fallback: standard rank-54 decomposition
    U = [[0] * 54 for _ in range(n * m)]
    V = [[0] * 54 for _ in range(m * p)]
    W = [[0] * 54 for _ in range(p * n)]
    col = 0
    
    def add_strassen_block_fallback(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
    
    def add_naive_2112_fallback(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
    
    add_strassen_block_fallback(0, 0, 0)
    add_strassen_block_fallback(0, 0, 2)
    add_strassen_block_fallback(0, 0, 4)
    add_strassen_block_fallback(0, 2, 0)
    add_strassen_block_fallback(0, 2, 2)
    add_strassen_block_fallback(0, 2, 4)
    add_naive_2112_fallback(0, 4, 0)
    add_naive_2112_fallback(0, 4, 2)
    add_naive_2112_fallback(0, 4, 4)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    return {"U": to_clean_list(U_arr), "V": to_clean_list(V_arr), "W": to_clean_list(W_arr)}
