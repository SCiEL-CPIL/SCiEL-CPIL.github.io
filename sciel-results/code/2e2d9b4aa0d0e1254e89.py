"""Improved Strassen-based matrix multiplication tensor decomposition for 3x5x6."""

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6

    # Schoolbook baseline
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

    def strassen_222_terms(i0, i1, j0, j1, k0, k1):
        """Return 7 rank-1 tensor terms for 2x2x2 Strassen decomposition."""
        terms = []
        u1 = {(i0,j0): 1, (i1,j1): 1}
        v1 = {(j0,k0): 1, (j1,k1): 1}
        w1 = {(k0,i0): 1, (k1,i1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(i1,j0): 1, (i1,j1): 1}
        v2 = {(j0,k0): 1}
        w2 = {(k0,i1): 1, (k1,i1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(i0,j0): 1}
        v3 = {(j1,k0): 1, (j1,k1): -1}
        w3 = {(k0,i0): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(i1,j1): 1}
        v4 = {(j0,k1): 1, (j0,k0): -1}
        w4 = {(k1,i1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(i0,j0): 1, (i0,j1): 1}
        v5 = {(j1,k1): 1}
        w5 = {(k1,i1): 1, (k0,i0): -1}
        terms.append((u5, v5, w5))
        
        u6 = {(i1,j0): 1, (i0,j0): -1}
        v6 = {(j0,k0): 1, (j0,k1): 1}
        w6 = {(k1,i1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(i0,j1): 1, (i1,j1): -1}
        v7 = {(j1,k0): 1, (j1,k1): 1}
        w7 = {(k0,i0): 1, (k1,i1): -1}
        terms.append((u7, v7, w7))
        
        return terms

    def schoolbook_terms(i_list, j_list, k_list):
        """Return rank-1 tensor terms for schoolbook decomposition of a block."""
        terms = []
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms

    def generate_partitions(size):
        """Generate all set partitions of {0, 1, ..., size-1}."""
        if size == 0:
            yield []
            return
        if size == 1:
            yield [[0]]
            return
        
        def _gen(remaining, parts):
            if not remaining:
                yield [list(p) for p in parts]
                return
            first = remaining[0]
            rest = remaining[1:]
            
            # Try adding first to each existing part
            for i in range(len(parts)):
                parts[i].append(first)
                yield from _gen(rest, parts)
                parts[i].pop()
            
            # Try making a new part with just first
            parts.append([first])
            yield from _gen(rest, parts)
            parts.pop()
        
        yield from _gen(list(range(size)), [])

    # Enumerate all block partitions and find the one with minimum rank
    best_partition = None
    best_partition_rank = rank_schoolbook
    
    i_part_list = list(generate_partitions(n))
    j_part_list = list(generate_partitions(m))
    k_part_list = list(generate_partitions(p))
    
    for i_parts in i_part_list:
        for j_parts in j_part_list:
            for k_parts in k_part_list:
                rank = 0
                for i_blk in i_parts:
                    for j_blk in j_parts:
                        for k_blk in k_parts:
                            if len(i_blk) == 2 and len(j_blk) == 2 and len(k_blk) == 2:
                                rank += 7
                            else:
                                rank += len(i_blk) * len(j_blk) * len(k_blk)
                
                if rank < best_partition_rank:
                    best_partition_rank = rank
                    best_partition = (i_parts, j_parts, k_parts)
    
    if best_partition is not None and best_partition_rank < best_rank:
        i_parts, j_parts, k_parts = best_partition
        all_terms = []
        for i_blk in i_parts:
            for j_blk in j_parts:
                for k_blk in k_parts:
                    if len(i_blk) == 2 and len(j_blk) == 2 and len(k_blk) == 2:
                        i0, i1 = i_blk
                        j0, j1 = j_blk
                        k0, k1 = k_blk
                        terms = strassen_222_terms(i0, i1, j0, j1, k0, k1)
                        all_terms.extend(terms)
                    else:
                        terms = schoolbook_terms(i_blk, j_blk, k_blk)
                        all_terms.extend(terms)
        
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
        
        # Verify correctness
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
            return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
