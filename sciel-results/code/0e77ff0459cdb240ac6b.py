"""Rewritten program for matrix multiplication 3,5,6 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook_decomp():
        """Return the schoolbook decomposition of rank n*m*p = 90."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact using numpy."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U3[:, :, c]  # (n, m)
            vc = V3[:, :, c]  # (m, p)
            wc = W3[:, :, c]  # (p, n)
            tmp = uc[:, :, None] * vc[None, :, :]  # (n, m, p)
            wc_3d = np.zeros((n, 1, p))
            for k in range(p):
                for i in range(n):
                    wc_3d[i, 0, k] = wc[k, i]
            T += tmp * wc_3d
        
        if np.any(np.abs(T - 1.0) > 1e-9):
            return False
        return True

    def strassen_2x2x2():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
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

    def build_strassen_block(i_start, j_start, k_start, U, V, W, col):
        """Build a Strassen block for 2x2x2 starting at global positions."""
        strassen_terms = strassen_2x2x2()
        for (u_pat, v_pat, w_pat) in strassen_terms:
            for (li, lj), val_u in u_pat.items():
                gi = i_start + li
                gj = j_start + lj
                U[gi * m + gj, col] = val_u
            for (lj, lk), val_v in v_pat.items():
                gj = j_start + lj
                gk = k_start + lk
                V[gj * p + gk, col] = val_v
            for (lk, li), val_w in w_pat.items():
                gk = k_start + lk
                gi = i_start + li
                W[gk * n + gi, col] = val_w
            col += 1
        return col

    def build_schoolbook_block(i_start, j_start, k_start, si, sj, sk, U, V, W, col):
        """Build a schoolbook block for si x sj x sk."""
        for li in range(si):
            for lj in range(sj):
                for lk in range(sk):
                    gi = i_start + li
                    gj = j_start + lj
                    gk = k_start + lk
                    U[gi * m + gj, col] = 1
                    V[gj * p + gk, col] = 1
                    W[gk * n + gi, col] = 1
                    col += 1
        return col

    def get_block_rank(si, sj, sk):
        """Get the rank for a block of size si x sj x sk."""
        if si == 2 and sj == 2 and sk == 2:
            return 7
        else:
            return si * sj * sk

    def generate_partitions(dim):
        """Generate all partitions of {0..dim-1} into contiguous groups of size 1 or 2."""
        results = []
        def helper(start, current):
            if start == dim:
                results.append(current[:])
                return
            if start + 1 <= dim:
                current.append([start])
                helper(start + 1, current)
                current.pop()
            if start + 2 <= dim:
                current.append([start, start + 1])
                helper(start + 2, current)
                current.pop()
        helper(0, [])
        return results

    def find_best_partition():
        """Find the partition with minimum rank."""
        best_rank = n * m * p
        best_i = None
        best_j = None
        best_k = None
        
        i_partitions = generate_partitions(n)
        j_partitions = generate_partitions(m)
        k_partitions = generate_partitions(p)
        
        for ip in i_partitions:
            if time.time() > deadline:
                break
            for jp in j_partitions:
                if time.time() > deadline:
                    break
                for kp in k_partitions:
                    if time.time() > deadline:
                        break
                    total_rank = 0
                    for ig in ip:
                        for jg in jp:
                            for kg in kp:
                                total_rank += get_block_rank(len(ig), len(jg), len(kg))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_i = ip
                        best_j = jp
                        best_k = kp
        
        return best_i, best_j, best_k, best_rank

    def build_decomposition(i_part, j_part, k_part):
        """Build the decomposition for the given partition."""
        total_rank = 0
        for ig in i_part:
            for jg in j_part:
                for kg in k_part:
                    total_rank += get_block_rank(len(ig), len(jg), len(kg))
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        i_offset = 0
        for ig in i_part:
            j_offset = 0
            for jg in j_part:
                k_offset = 0
                for kg in k_part:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        col = build_strassen_block(i_offset, j_offset, k_offset, U, V, W, col)
                    else:
                        col = build_schoolbook_block(i_offset, j_offset, k_offset, si, sj, sk, U, V, W, col)
                    k_offset += sk
                j_offset += sj
            i_offset += si
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    # Start with schoolbook as fallback
    best_result = schoolbook_decomp()
    best_rank = n * m * p

    # Find the best partition
    best_i, best_j, best_k, best_rank_found = find_best_partition()
    
    if best_i is not None and best_rank_found < best_rank:
        result = build_decomposition(best_i, best_j, best_k)
        r = result.pop("rank")
        if verify(result["U"], result["V"], result["W"], r):
            best_result = result
            best_rank = r

    return best_result
# EVOLVE-BLOCK-END
