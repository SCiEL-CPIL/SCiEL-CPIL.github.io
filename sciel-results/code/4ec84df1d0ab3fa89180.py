import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

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
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
        terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}))
        terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}))
        terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
        terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}))
        terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
        terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
        return terms

    def build_decomposition(i_groups, j_groups, k_groups):
        """Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere."""
        strassen_terms = strassen_2x2x2_terms()
        
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
                            gi, gj, gk = ig[li], lj if False else jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

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

    # Generate candidate partitions
    def generate_partitions(dim, min_group=1, max_group=None):
        """Generate all ways to partition a range into groups of size 1 or 2."""
        if max_group is None:
            max_group = min(2, dim)
        
        results = []
        
        def backtrack(remaining, current):
            if not remaining:
                results.append(list(current))
                return
            if time.time() > deadline:
                return
            
            # Try group of size 1
            if remaining >= 1:
                current.append([remaining[0]])
                backtrack(remaining[1:], current)
                current.pop()
            
            # Try group of size 2
            if remaining >= 2:
                current.append([remaining[0], remaining[1]])
                backtrack(remaining[2:], current)
                current.pop()
        
        backtrack(list(range(dim)), [])
        return results

    # Try the known good partition first
    best_result = None
    best_rank = n * m * p

    # Known good partition: i=[0,1],[2]; j=[0,1],[2,3],[4]; k=[0,1],[2,3],[4,5]
    i_groups_1 = [[0, 1], [2]]
    j_groups_1 = [[0, 1], [2, 3], [4]]
    k_groups_1 = [[0, 1], [2, 3], [4, 5]]
    
    result1 = build_decomposition(i_groups_1, j_groups_1, k_groups_1)
    rank1 = result1.pop("rank")
    if verify(result1["U"], result1["V"], result1["W"], rank1):
        best_result = result1
        best_rank = rank1

    # Try other partitions if time permits
    if time.time() < deadline - 30:
        # Try different j partitions
        j_alternatives = [
            [[0, 1], [2, 3], [4]],
            [[0, 1], [2], [3, 4]],
            [[0], [1, 2], [3, 4]],
            [[0, 1, 2], [3, 4]],
            [[0], [1], [2, 3], [4]],
        ]
        
        for j_alt in j_alternatives:
            if time.time() > deadline - 10:
                break
            result = build_decomposition(i_groups_1, j_alt, k_groups_1)
            rank = result.pop("rank")
            if rank < best_rank:
                if verify(result["U"], result["V"], result["W"], rank):
                    best_result = result
                    best_rank = rank

    # Try different i partitions
    if time.time() < deadline - 20:
        i_alternatives = [
            [[0, 1], [2]],
            [[0], [1, 2]],
        ]
        
        for i_alt in i_alternatives:
            if time.time() > deadline - 10:
                break
            for j_alt in [j_groups_1, [[0, 1], [2], [3, 4]], [[0], [1, 2], [3, 4]]]:
                if time.time() > deadline - 5:
                    break
                result = build_decomposition(i_alt, j_alt, k_groups_1)
                rank = result.pop("rank")
                if rank < best_rank:
                    if verify(result["U"], result1["V"] if False else result["V"], result["W"], rank):
                        best_result = result
                        best_rank = rank

    # Fallback to schoolbook if nothing worked
    if best_result is None:
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result
# EVOLVE-BLOCK-END
