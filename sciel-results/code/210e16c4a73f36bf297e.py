"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_block_decomposition(i_groups, j_groups, k_groups):
        """
        Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W}

    def count_rank(i_groups, j_groups, k_groups):
        """Count the total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try multiple partitions
    partitions = [
        # Original partition (rank 51)
        ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
        # Partition from directive idea: i={0,1},{2}, j={0,1,2,3},{4}, k={0,1,2,3},{4,5}
        ([[0, 1], [2]], [[0, 1, 2, 3], [4]], [[0, 1, 2, 3], [4, 5]]),
        # Another partition
        ([[0, 1], [2]], [[0, 1, 2], [3, 4]], [[0, 1, 2], [3, 4, 5]]),
        # Fine-grained partition to get more 2x2x2 blocks
        ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
        # Try: i={0},{1},{2}, j={0,1},{2,3},{4}, k={0,1},{2,3},{4,5}
        ([[0], [1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
        # Try: i={0,1},{2}, j={0},{1,2},{3,4}, k={0,1},{2,3},{4,5}
        ([[0, 1], [2]], [[0], [1, 2], [3, 4]], [[0, 1], [2, 3], [4, 5]]),
        # Try: i={0},{1},{2}, j={0,1},{2,3},{4}, k={0},{1,2},{3,4},{5}
        ([[0], [1], [2]], [[0, 1], [2, 3], [4]], [[0], [1, 2], [3, 4], [5]]),
        # Try: i={0,1},{2}, j={0,1},{2},{3,4}, k={0,1},{2,3},{4,5}
        ([[0, 1], [2]], [[0, 1], [2], [3, 4]], [[0, 1], [2, 3], [4, 5]]),
        # Try: i={0},{1},{2}, j={0},{1},{2},{3,4}, k={0,1},{2,3},{4,5}
        ([[0], [1], [2]], [[0], [1], [2], [3, 4]], [[0, 1], [2, 3], [4, 5]]),
        # Try: i={0,1},{2}, j={0,1},{2,3},{4}, k={0},{1,2},{3,4},{5}
        ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0], [1, 2], [3, 4], [5]]),
    ]

    # Sort partitions by estimated rank (ascending)
    ranked_partitions = []
    for ig, jg, kg in partitions:
        r = count_rank(ig, jg, kg)
        ranked_partitions.append((r, ig, jg, kg))
    ranked_partitions.sort(key=lambda x: x[0])

    for r, ig, jg, kg in ranked_partitions:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        result = build_block_decomposition(ig, jg, kg)
        if verify(result["U"], result["V"], result["W"], r):
            best_result = result
            best_rank = r
            if best_rank <= 45:  # Good enough, stop searching
                break

    return best_result
# EVOLVE-BLOCK-END
