"""Matrix multiplication tensor decomposition for M_{3,4,7} using Strassen block."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    def get_trivial_block(n, m, p):
        """Returns U, V, W for M_{n,m,p} with trivial rank n*m*p"""
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
        return U, V, W, rank
    
    def verify_full_fast(U, V, W, rank, n, m, p):
        """Verify full decomposition for M_{n,m,p} using vectorized numpy"""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        valid_constraints = []
        invalid_constraints = []
        
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and k_b == k_c:
                        valid_constraints.append((a, b, c))
                    else:
                        invalid_constraints.append((a, b, c))
        
        if valid_constraints:
            va = np.array([x[0] for x in valid_constraints])
            vb = np.array([x[1] for x in valid_constraints])
            vc = np.array([x[2] for x in valid_constraints])
            products = U_arr[va] * V_arr[vb] * W_arr[vc]
            sums = products.sum(axis=1)
            if np.max(np.abs(sums - 1.0)) > 1e-9:
                return False
        
        if invalid_constraints:
            ia = np.array([x[0] for x in invalid_constraints])
            ib = np.array([x[1] for x in invalid_constraints])
            ic = np.array([x[2] for x in invalid_constraints])
            products = U_arr[ia] * V_arr[ib] * W_arr[ic]
            sums = products.sum(axis=1)
            if np.max(np.abs(sums)) > 1e-9:
                return False
        
        return True
    
    def get_strassen_block(n, m, p, i0, j0, k0):
        """
        Return the 7 Strassen columns for the 2x2x2 block at positions (i0,i0+1), (j0,j0+1), (k0,k0+1)
        These columns should replace the 8 trivial columns for that block.
        """
        # Strassen for 2x2 matrices:
        # m1 = (A00 + A11)(B00 + B11)
        # m2 = (A10 + A11)B00
        # m3 = A00(B01 - B11)
        # m4 = A11(B10 - B00)
        # m5 = (A01 + A10)(B10 + B11)
        # m6 = (A01 - A10)(B00 + B01)
        # m7 = (A11 - A00)(B11 + B01)
        #
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        U_block = []
        V_block = []
        W_block = []
        
        # m1 = (A00 + A11)(B00 + B11)
        U_block.append([1, 0, 1, 0])  # A00, A01, A10, A11 -> [i0,j0], [i0,j0+1], [i0+1,j0], [i0+1,j0+1]
        V_block.append([1, 0, 0, 1])  # B00, B01, B10, B11 -> [j0,k0], [j0,k0+1], [j0+1,k0], [j0+1,k0+1]
        W_block.append([1, 0, 0, 1])  # C00, C01, C10, C11 -> [k0,i0], [k0+1,i0], [k0,i0+1], [k0+1,i0+1]
        
        # m2 = (A10 + A11)B00
        U_block.append([0, 0, 1, 1])
        V_block.append([1, 0, 0, 0])
        W_block.append([0, 0, 1, 0])  # C10
        
        # m3 = A00(B01 - B11)
        U_block.append([1, 0, 0, 0])
        V_block.append([0, 1, 0, -1])
        W_block.append([0, 1, 0, 1])  # C01 + C11
        
        # m4 = A11(B10 - B00)
        U_block.append([0, 0, 0, 1])
        V_block.append([0, 0, 1, -1])
        W_block.append([1, 0, 1, 0])  # C00 + C10
        
        # m5 = (A01 + A10)(B10 + B11)
        U_block.append([0, 1, 1, 0])
        V_block.append([0, 0, 1, 1])
        W_block.append([0, 1, -1, 0])  # C01 - C10
        
        # m6 = (A01 - A10)(B00 + B01)
        U_block.append([0, 1, -1, 0])
        V_block.append([1, 1, 0, 0])
        W_block.append([0, 0, 0, 1])  # C11
        
        # m7 = (A11 - A00)(B11 + B01)
        U_block.append([-1, 0, 0, 1])
        V_block.append([0, 1, 0, 1])
        W_block.append([1, 0, 0, 1])  # C00 + C11
        
        return U_block, V_block, W_block
    
    # Start with trivial decomposition
    U_triv, V_triv, W_triv, rank_triv = get_trivial_block(n, m, p)
    
    best_total_rank = rank_triv
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    
    # Try applying Strassen to the 2x2x2 block at (i0=0, j0=0, k0=0)
    i0, j0, k0 = 0, 0, 0
    
    # The 8 trivial columns to remove for this block
    # Columns are indexed by (i, j, k) in order: i in range(n), j in range(m), k in range(p)
    # Column index for (i,j,k) is i*m*p + j*p + k
    trivial_cols = []
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                col_idx = i * m * p + j * p + k
                trivial_cols.append(col_idx)
    
    # Remove these 8 columns
    U_str = [row[:] for row in U_triv]
    V_str = [row[:] for row in V_triv]
    W_str = [row[:] for row in W_triv]
    
    for col in sorted(trivial_cols, reverse=True):
        for i in range(len(U_str)):
            U_str[i].pop(col)
        for i in range(len(V_str)):
            V_str[i].pop(col)
        for i in range(len(W_str)):
            W_str[i].pop(col)
    
    # Add the 7 Strassen columns
    U_s, V_s, W_s = get_strassen_block(n, m, p, i0, j0, k0)
    
    for r in range(7):
        # U: rows are (i, j) with i in [0,2], j in [0,3]
        # The 2x2 block covers i in {0,1}, j in {0,1}
        for di in range(2):
            for dj in range(2):
                row_idx = (i0 + di) * m + (j0 + dj)
                U_str[row_idx].append(U_s[r][di * 2 + dj])
        
        # V: rows are (j, k) with j in [0,3], k in [0,6]
        # The 2x2 block covers j in {0,1}, k in {0,1}
        for dj in range(2):
            for dk in range(2):
                row_idx = (j0 + dj) * p + (k0 + dk)
                V_str[row_idx].append(V_s[r][dj * 2 + dk])
        
        # W: rows are (k, i) with k in [0,6], i in [0,2]
        # The 2x2 block covers k in {0,1}, i in {0,1}
        for dk in range(2):
            for di in range(2):
                row_idx = (k0 + dk) * n + (i0 + di)
                W_str[row_idx].append(W_s[r][dk * 2 + di])
    
    new_rank = rank_triv - 8 + 7
    
    # Verify
    if verify_full_fast(U_str, V_str, W_str, new_rank, n, m, p):
        if new_rank < best_total_rank:
            best_total_rank = new_rank
            best_result = {"U": U_str, "V": V_str, "W": W_str}
    
    # Try other 2x2x2 blocks as well
    # (i0, j0, k0) where i0 in {0,1}, j0 in {0,1,2}, k0 in {0,1,2,3,4,5}
    for i0_try in range(n - 1):
        for j0_try in range(m - 1):
            for k0_try in range(p - 1):
                if time.time() > deadline - 10:
                    break
                
                # Check if this block overlaps with the already-replaced block
                if i0_try in [i0, i0+1] and j0_try in [j0, j0+1] and k0_try in [k0, k0+1]:
                    continue
                
                U_try = [row[:] for row in U_triv]
                V_try = [row[:] for row in V_triv]
                W_try = [row[:] for row in W_triv]
                
                trivial_cols_try = []
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            i = i0_try + di
                            j = j0_try + dj
                            k = k0_try + dk
                            col_idx = i * m * p + j * p + k
                            trivial_cols_try.append(col_idx)
                
                for col in sorted(trivial_cols_try, reverse=True):
                    for i in range(len(U_try)):
                        U_try[i].pop(col)
                    for i in range(len(V_try)):
                        V_try[i].pop(col)
                    for i in range(len(W_try)):
                        W_try[i].pop(col)
                
                U_s2, V_s2, W_s2 = get_strassen_block(n, m, p, i0_try, j0_try, k0_try)
                
                for r in range(7):
                    for di in range(2):
                        for dj in range(2):
                            row_idx = (i0_try + di) * m + (j0_try + dj)
                            U_try[row_idx].append(U_s2[r][di * 2 + dj])
                    
                    for dj in range(2):
                        for dk in range(2):
                            row_idx = (j0_try + dj) * p + (k0_try + dk)
                            V_try[row_idx].append(V_s2[r][dj * 2 + dk])
                    
                    for dk in range(2):
                        for di in range(2):
                            row_idx = (k0_try + dk) * n + (i0_try + di)
                            W_try[row_idx].append(W_s2[r][dk * 2 + di])
                
                new_rank_try = rank_triv - 8 + 7
                
                if verify_full_fast(U_try, V_try, W_try, new_rank_try, n, m, p):
                    if new_rank_try < best_total_rank:
                        best_total_rank = new_rank_try
                        best_result = {"U": U_try, "V": V_try, "W": W_try}
                
                if time.time() > deadline - 5:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
