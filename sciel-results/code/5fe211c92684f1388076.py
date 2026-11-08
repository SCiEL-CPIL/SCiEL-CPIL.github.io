"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

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
        """Verify that the decomposition is exact using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # Build the tensor check efficiently
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j]
                    v_row = V_np[j * p + k]
                    w_row = W_np[k * n + i]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_dict, v_dict, w_dict).
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # Term 2: M2 = (A10+A11)B00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        # Term 3: M3 = A00(B01-B11)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        # Term 4: M4 = A11(B10-B00)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        # Term 5: M5 = (A00+A01)B11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        # Term 6: M6 = (A10-A00)(B00+B01)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # Term 7: M7 = (A01-A11)(B10+B11)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms

    def all_set_partitions(N):
        """
        Generate all set partitions of {0,...,N-1} into blocks of size 1 or 2.
        Returns list of lists of frozensets.
        """
        if N == 0:
            return [[]]
        if N == 1:
            return [[frozenset([0])]]
        if N == 2:
            return [[frozenset([0]), frozenset([1])], frozenset([0,1])]
        
        results = []
        # 0 is alone
        for rest in all_set_partitions(N - 1):
            results.append([frozenset([0])] + rest)
        # 0 is paired with each other element
        for j in range(1, N):
            # Remove j from the rest
            remaining = list(range(1, N))
            remaining.remove(j)
            # Generate partitions of remaining
            def gen_partitions_of_set(S):
                if not S:
                    return [[]]
                first = S[0]
                rest = S[1:]
                res = []
                # first alone
                for r in gen_partitions_of_set(rest):
                    res.append([frozenset([first])] + r)
                # first paired with each in rest
                for k in rest:
                    rem2 = list(rest)
                    rem2.remove(k)
                    for r in gen_partitions_of_set(rem2):
                        res.append([frozenset([first, k])] + r)
                return res
            for r in gen_partitions_of_set(remaining):
                results.append([frozenset([0, j])] + r)
        return results

    def build_block_decomposition(i_groups, j_groups, k_groups):
        """
        Build a block decomposition using Strassen for 2x2x2 blocks
        and schoolbook for all other blocks.
        i_groups, j_groups, k_groups are lists of frozensets.
        """
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
                # Sort the groups for consistent indexing
                ig_sorted = sorted(ig)
                jg_sorted = sorted(jg)
                kg_sorted = sorted(kg)
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig_sorted[li], jg_sorted[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg_sorted[lj], kg_sorted[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg_sorted[lk], ig_sorted[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                ig_sorted = sorted(ig)
                jg_sorted = sorted(jg)
                kg_sorted = sorted(kg)
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig_sorted[li], jg_sorted[lj], kg_sorted[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def compute_rank_for_partition(i_groups, j_groups, k_groups):
        """Compute the rank for a given partition without building the full matrices."""
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total_rank += 7
                    else:
                        total_rank += si * sj * sk
        return total_rank

    # Generate all set partitions for each dimension
    i_partitions = all_set_partitions(n)
    j_partitions = all_set_partitions(m)
    k_partitions = all_set_partitions(p)
    
    best_rank = n * m * p
    best_partition = None
    
    # First pass: find the best rank by enumeration
    for ip in i_partitions:
        if time.time() > deadline:
            break
        for jp in j_partitions:
            if time.time() > deadline:
                break
            for kp in k_partitions:
                if time.time() > deadline:
                    break
                rank = compute_rank_for_partition(ip, jp, kp)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (ip, jp, kp)
    
    # Build the best decomposition
    if best_partition is not None:
        ip, jp, kp = best_partition
        result = build_block_decomposition(ip, jp, kp)
        result.pop("rank")
        
        # Verify
        if verify(result["U"], result["V"], result["W"], best_rank):
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
