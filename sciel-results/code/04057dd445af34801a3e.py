import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Trivial decomposition as fallback
    U_triv = [[0] * trivial_rank for _ in range(n * m)]
    V_triv = [[0] * trivial_rank for _ in range(m * p)]
    W_triv = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1

    best_rank = trivial_rank
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}

    # Check for rank 41 using random search with small integers
    # We try to find U, V, W of rank 41 such that the tensor matches.
    # Given the sparsity, random search is unlikely to succeed quickly,
    # but we try a few hundred attempts.
    
    def check_decomposition(U, V, W, rank):
        """
        Verify that the decomposition exactly matches the matrix multiplication tensor.
        Conditions:
        1. For all i, j, k: sum_t U[3i+j][t] * V[7j+k][t] * W[2k+i][t] == 1
        2. For all i, j, k, i2, j2, k2 where (i,j) != (i2,j2) or (j,k) != (j2,k2) or (k,i) != (k2,i2):
           sum_t U[3i+j][t] * V[7j+k][t] * W[2k+i][t] == 0
           Note: The tensor is zero if the indices don't align for multiplication.
           Specifically, T[(i,j), (j',k), (k',i')] is 1 if i=i', j=j', k=k', else 0.
        """
        # Check the 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for t in range(rank):
                        s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                    if s != 1:
                        return False
        
        # Check the 0s (Sample a subset for speed, or all if fast)
        # Total combinations: 6 * 21 * 14 = 1764.
        # We can iterate through all and check if the indices are not a valid (i,j,k) triple.
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        v_idx = j2 * p + k
                        for i2 in range(n):
                            for k2 in range(p):
                                w_idx = k2 * n + i2
                                # Check if this is a valid multiplication term
                                if i == i2 and j == j2 and k == k2:
                                    continue # Already checked as 1
                                # Check if sum is 0
                                s = 0
                                for t in range(rank):
                                    s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                                if s != 0:
                                    return False
        return True

    # Try rank 41
    target_rank = 41
    if time.time() < deadline - 10:
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers
            # Use -1, 0, 1 to keep numbers small
            U_r = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V_r = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W_r = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick check on the 1s first
            valid_ones = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for t in range(target_rank):
                            s += U_r[u_idx][t] * V_r[v_idx][t] * W_r[w_idx][t]
                        if s != 1:
                            valid_ones = False
                            break
                    if not valid_ones:
                        break
                if not valid_ones:
                    break
            
            if not valid_ones:
                continue
                
            # Full check
            if check_decomposition(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break

    return best_result
