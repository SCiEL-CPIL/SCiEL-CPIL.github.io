import time
import itertools
from functools import lru_cache

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    n, m, p = DIMENSIONS
    
    # Generate all partitions of a number into k positive integers
    def get_partitions(k, num):
        if k == 1:
            yield [num]
        elif k > num:
            return
        else:
            for first in range(1, num - k + 2):
                for rest in get_partitions(k - 1, num - first):
                    yield [first] + rest

    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    best_rank = n * m * p
    best_part = (None, None, None)
    
    # Max possible k is min(n, m, p)
    max_k = min(n, m, p)
    
    for k in range(1, max_k + 1):
        parts_n = list(get_partitions(k, n))
        parts_m = list(get_partitions(k, m))
        parts_p = list(get_partitions(k, p))
        
        if not parts_n or not parts_m or not parts_p:
            continue
            
        for pn in parts_n:
            for pm in parts_m:
                for pp in parts_p:
                    rank = 0
                    for a, b, c in zip(pn, pm, pp):
                        rank += block_rank(a, b, c)
                    if rank < best_rank:
                        best_rank = rank
                        best_part = (pn, pm, pp)
                        
    pn, pm, pp = best_part
    
    total_rank = best_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Calculate offsets
    i_offsets = [0]
    for s in pn:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in pm:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in pp:
        k_offsets.append(k_offsets[-1] + s)
        
    for i_idx in range(len(pn)):
        i_start = i_offsets[i_idx]
        a_size = pn[i_idx]
        for j_idx in range(len(pm)):
            j_start = j_offsets[j_idx]
            b_size = pm[j_idx]
            for k_idx in range(len(pp)):
                k_start = k_offsets[k_idx]
                c_size = pp[k_idx]
                
                if a_size == 2 and b_size == 2 and c_size == 2:
                    # Strassen
                    # M1
                    U[i_start * m + j_start][col] = 1
                    U[(i_start+1) * m + (j_start+1)][col] = 1
                    V[j_start * p + k_start][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = 1
                    W[k_start * n + i_start][col] = 1
                    W[(k_start+1) * n + (i_start+1)][col] = 1
                    col += 1
                    # M2
                    U[(i_start+1) * m + j_start][col] = 1
                    U[(i_start+1) * m + (j_start+1)][col] = 1
                    V[j_start * p + k_start][col] = 1
                    W[k_start * n + (i_start+1)][col] = 1
                    col += 1
                    # M3
                    U[i_start * m + j_start][col] = 1
                    V[(j_start+1) * p + k_start][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = -1
                    W[k_start * n + i_start][col] = 1
                    W[(k_start+1) * n + i_start][col] = 1
                    col += 1
                    # M4
                    U[(i_start+1) * m + (j_start+1)][col] = 1
                    V[j_start * p + (k_start+1)][col] = 1
                    V[j_start * p + k_start][col] = -1
                    W[k_start * n + i_start][col] = 1
                    W[(k_start+1) * n + (i_start+1)][col] = 1
                    col += 1
                    # M5
                    U[i_start * m + j_start][col] = 1
                    U[i_start * m + (j_start+1)][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = 1
                    W[k_start * n + i_start][col] = -1
                    W[(k_start+1) * n + i_start][col] = 1
                    col += 1
                    # M6
                    U[(i_start+1) * m + j_start][col] = 1
                    U[i_start * m + j_start][col] = -1
                    V[j_start * p + k_start][col] = 1
                    V[j_start * p + (k_start+1)][col] = 1
                    W[(k_start+1) * n + (i_start+1)][col] = 1
                    col += 1
                    # M7
                    U[i_start * m + (j_start+1)][col] = 1
                    U[(i_start+1) * m + (j_start+1)][col] = -1
                    V[(j_start+1) * p + k_start][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = 1
                    W[k_start * n + i_start][col] = 1
                    col += 1
                else:
                    # Schoolbook
                    for ii in range(a_size):
                        for jj in range(b_size):
                            for kk in range(c_size):
                                I = i_start + ii
                                J = j_start + jj
                                K = k_start + kk
                                U[I * m + J][col] = 1
                                V[J * p + K][col] = 1
                                W[K * n + I][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
