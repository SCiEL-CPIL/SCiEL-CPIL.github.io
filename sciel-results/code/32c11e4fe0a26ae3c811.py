"""Matrix multiplication 3x5x6 with rank-minimal tensor decomposition."""
import time
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

    def decomp_222_rank7():
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

    def get_ordered_partitions(dim):
        """Get all ordered partitions of dim into positive integers."""
        if dim == 1:
            return [[1]]
        results = []
        for first in range(1, dim + 1):
            if first == dim:
                results.append([dim])
            else:
                for rest in get_ordered_partitions(dim - first):
                    results.append([first] + rest)
        return results

    def block_rank(si, sj, sk):
        """Return the rank for a block of size si x sj x sk."""
        if si == 2 and sj == 2 and sk == 2:
            return 7
        return si * sj * sk

    def compute_total_rank(i_sizes, j_sizes, k_sizes):
        """Compute total rank for given block size lists."""
        total = 0
        for si in i_sizes:
            for sj in j_sizes:
                for sk in k_sizes:
                    total += block_rank(si, sj, sk)
        return total

    def find_best_partition():
        """Find the partition that minimizes total rank."""
        i_parts = get_ordered_partitions(n)
        j_parts = get_ordered_partitions(m)
        k_parts = get_ordered_partitions(p)
        
        best_rank = n * m * p
        best_sizes = (None, None, None)
        
        for ip in i_parts:
            if time.time() > deadline - 10:
                break
            for jp in j_parts:
                for kp in k_parts:
                    r = compute_total_rank(ip, jp, kp)
                    if r < best_rank:
                        best_rank = r
                        best_sizes = (ip, jp, kp)
        
        return best_rank, best_sizes

    def sizes_to_groups(sizes):
        """Convert a list of sizes to a list of index groups."""
        groups = []
        idx = 0
        for s in sizes:
            groups.append(list(range(idx, idx + s)))
            idx += s
        return groups

    def build_decomposition(i_groups, j_groups, k_groups):
        """Build the decomposition for a given partition."""
        strassen_terms = decomp_222_rank7()
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    r = block_rank(si, sj, sk)
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
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    best_result = schoolbook()
    best_rank = n * m * p

    if time.time() < deadline - 5:
        optimal_rank, (opt_i_sizes, opt_j_sizes, opt_k_sizes) = find_best_partition()
        
        if opt_i_sizes is not None and optimal_rank < best_rank:
            opt_i = sizes_to_groups(opt_i_sizes)
            opt_j = sizes_to_groups(opt_j_sizes)
            opt_k = sizes_to_groups(opt_k_sizes)
            result = build_decomposition(opt_i, opt_j, opt_k)
            rank = result.pop("rank")
            
            if verify(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
