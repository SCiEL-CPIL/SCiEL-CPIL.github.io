import numpy as np
import time
from fractions import Fraction
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # We'll use a block decomposition approach with Strassen's algorithm
    # for 2x2 blocks and schoolbook for 1x1 blocks.
    
    # The key insight: we can partition the 3x5x7 multiplication into blocks
    # and use optimal algorithms for each block.
    
    # For a 3x5x7 multiplication:
    # - Partition rows (i) into sizes: [2, 1]
    # - Partition cols (j) into sizes: [2, 2, 1]  
    # - Partition cols (k) into sizes: [2, 2, 2, 1]
    
    # Block costs:
    # - 2x2x2 block: Strassen rank 7
    # - 2x2x1 block: rank 4 (schoolbook)
    # - 2x1x2 block: rank 4 (schoolbook)
    # - 1x2x2 block: rank 4 (schoolbook)
    # - 1x1x2 block: rank 2 (schoolbook)
    # - 2x1x1 block: rank 2 (schoolbook)
    # - 1x2x1 block: rank 2 (schoolbook)
    # - 1x1x1 block: rank 1 (schoolbook)
    
    # Let's enumerate all possible partitions and find the minimum rank
    
    def block_rank(n_b, m_b, p_b):
        """Return the rank for multiplying n_b x m_b by m_b x p_b."""
        if n_b == 2 and m_b == 2 and p_b == 2:
            return 7  # Strassen
        else:
            return n_b * m_b * p_b  # Schoolbook
    
    def total_rank_for_partition(pi, pj, pk):
        """Compute total rank for given partitions of i, j, k dimensions."""
        total = 0
        i_start = 0
        for n_b in pi:
            j_start = 0
            for m_b in pj:
                k_start = 0
                for p_b in pk:
                    total += block_rank(n_b, m_b, p_b)
                    k_start += p_b
                j_start += m_b
            i_start += n_b
        return total
    
    # Generate all possible partitions of n, m, p into parts of size 1 or 2
    def all_partitions(d, max_part=2):
        """Generate all partitions of d into parts of size 1 or max_part."""
        if d == 0:
            yield []
        elif d >= 1:
            for rest in all_partitions(d - 1, max_part):
                yield [1] + rest
        if d >= max_part:
            for rest in all_partitions(d - max_part, max_part):
                yield [max_part] + rest
    
    best_rank = n * m * p  # schoolbook
    best_pi = [1] * n
    best_pj = [1] * m
    best_pk = [1] * p
    
    for pi in all_partitions(n):
        for pj in all_partitions(m):
            for pk in all_partitions(p):
                r = total_rank_for_partition(pi, pj, pk)
                if r < best_rank:
                    best_rank = r
                    best_pi = pi
                    best_pj = pj
                    best_pk = pk
    
    # Now build the factor matrices for the best partition
    rank_total = best_rank
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    rank_offset = 0
    
    # Helper to add Strassen for 2x2x2 block
    def add_strassen(rank_offset, i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        # Strassen's 7 products for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # U factors: for A entries (local indices 0,1,2,3 = a,b,c,d)
        u_loc = [[0]*7 for _ in range(4)]
        u_loc[0][0] = 1; u_loc[3][0] = 1  # a+d
        u_loc[2][1] = 1; u_loc[3][1] = 1  # c+d
        u_loc[0][2] = 1                   # a
        u_loc[3][3] = 1                   # d
        u_loc[0][4] = 1; u_loc[1][4] = 1  # a+b
        u_loc[2][5] = 1; u_loc[0][5] = -1 # c-a
        u_loc[1][6] = 1; u_loc[3][6] = -1 # b-d
        
        # V factors: for B entries (local indices 0,1,2,3 = e,f,g,h)
        v_loc = [[0]*7 for _ in range(4)]
        v_loc[0][0] = 1; v_loc[3][0] = 1  # e+h
        v_loc[0][1] = 1                   # e
        v_loc[1][2] = 1; v_loc[3][2] = -1 # f-h
        v_loc[2][3] = 1; v_loc[0][3] = -1 # g-e
        v_loc[3][4] = 1                   # h
        v_loc[0][5] = 1; v_loc[1][5] = 1  # e+f
        v_loc[2][6] = 1; v_loc[3][6] = 1  # g+h
        
        # W factors: for C entries (local indices 0,1,2,3 = C00,C01,C10,C11)
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        w_loc = [[0]*7 for _ in range(4)]
        w_loc[0][0] = 1; w_loc[0][3] = -1; w_loc[0][4] = -1; w_loc[0][6] = 1
        w_loc[1][2] = 1; w_loc[1][4] = 1
        w_loc[2][1] = 1; w_loc[2][3] = 1
        w_loc[3][0] = 1; w_loc[3][1] = -1; w_loc[3][2] = 1; w_loc[3][5] = -1
        
        # Map local to global indices
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]
    
    # Helper to add schoolbook for any block
    def add_schoolbook(rank_offset, n_b, m_b, p_b, i0, j0, k0):
        """Add schoolbook decomposition for A[i0:i0+n_b, j0:j0+m_b] * B[j0:j0+m_b, k0:k0+p_b]"""
        block_rank = n_b * m_b * p_b
        for i_l in range(n_b):
            for j_l in range(m_b):
                for k_l in range(p_b):
                    r = (i_l * m_b + j_l) * p_b + k_l
                    u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                    U[u_g_idx][rank_offset + r] += 1
                    
                    v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                    V[v_g_idx][rank_offset + r] += 1
                    
                    w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                    W[w_g_idx][rank_offset + r] += 1
    
    # Build the decomposition
    i_start = 0
    for n_b in best_pi:
        j_start = 0
        for m_b in best_pj:
            k_start = 0
            for p_b in best_pk:
                if n_b == 2 and m_b == 2 and p_b == 2:
                    add_strassen(rank_offset, i_start, j_start, k_start)
                else:
                    add_schoolbook(rank_offset, n_b, m_b, p_b, i_start, j_start, k_start)
                rank_offset += block_rank(n_b, m_b, p_b)
                k_start += p_b
            j_start += m_b
        i_start += n_b
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank_total):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][idx] = 1
                    V_s[j*p+k][idx] = 1
                    W_s[k*n+i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
