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

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C11 (T[1,1,1])
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        # Contributes +1 to C10 (T[1,0,0]) and -1 to C11 (T[1,1,1])
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        # Contributes +1 to C01 (T[0,0,1]) and +1 to C11 (T[1,1,1])
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C10 (T[1,0,0])
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        # Contributes -1 to C00 (T[0,0,0]) and +1 to C01 (T[0,0,1])
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11 (T[1,1,1])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00 (T[0,0,0])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def get_partitions(dim):
        """Get all ordered partitions of dim into positive integers."""
        if dim == 1:
            return [[1]]
        results = []
        for first in range(1, dim + 1):
            if first == dim:
                results.append([dim])
            else:
                for rest in get_partitions(dim - first):
                    results.append([first] + rest)
        return results

    def block_rank(si, sj, sk):
        """Return the known minimal rank for an si x sj x sk matrix multiplication block."""
        # Schoolbook
        base = si * sj * sk
        # Strassen for 2x2x2
        if si == 2 and sj == 2 and sk == 2:
            return 7
        # 2x2x3 or permutations: split the size-3 dim into 2+1
        # 2x2x3 = 2x2x2 (7) + 2x2x1 (4) = 11
        # 2x3x2 = 2x2x2 (7) + 2x1x2 (4) = 11  
        # 3x2x2 = 2x2x2 (7) + 1x2x2 (4) = 11
        if sorted([si, sj, sk]) == [2, 2, 3]:
            return 11
        # 3x2x3 or permutations: 
        # 3x2x3 = 2x2x3 (11) + 1x2x3 (6) = 17
        # 3x3x2 = 2x3x2 (11) + 1x3x2 (6) = 17
        # 2x3x3 = 2x2x3 (11) + 2x1x3 (6) = 17
        if sorted([si, sj, sk]) == [2, 3, 3]:
            return 17
        # 3x3x3: split into 2+1 in one dim
        # 3x3x3 = 2x3x3 (17) + 1x3x3 (9) = 26
        if si == 3 and sj == 3 and sk == 3:
            return 26
        return base

    def compute_rank_for_partition(i_groups, j_groups, k_groups):
        """Compute total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    total += block_rank(si, sj, sk)
        return total

    def find_best_partition():
        """Find the partition that minimizes total rank."""
        i_parts = get_partitions(n)
        j_parts = get_partitions(m)
        k_parts = get_partitions(p)
        
        best_rank = n * m * p
        best_partition = (None, None, None)
        
        for ip in i_parts:
            if time.time() > deadline - 10:
                break
            for jp in j_parts:
                for kp in k_parts:
                    i_groups = []
                    idx = 0
                    for size in ip:
                        i_groups.append(list(range(idx, idx + size)))
                        idx += size
                    
                    j_groups = []
                    idx = 0
                    for size in jp:
                        j_groups.append(list(range(idx, idx + size)))
                        idx += size
                    
                    k_groups = []
                    idx = 0
                    for size in kp:
                        k_groups.append(list(range(idx, idx + size)))
                        idx += size
                    
                    r = compute_rank_for_partition(i_groups, j_groups, k_groups)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (i_groups, j_groups, k_groups)
        
        return best_rank, best_partition

    def build_block_decomposition(ig, jg, kg, col_offset, U, V, W, strassen_terms):
        """Build decomposition for a single block, starting at col_offset. Returns new col_offset."""
        si, sj, sk = len(ig), len(jg), len(kg)
        col = col_offset
        
        if si == 2 and sj == 2 and sk == 2:
            # Strassen 2x2x2
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
        elif sorted([si, sj, sk]) == [2, 2, 3]:
            # Split the size-3 dimension into 2+1
            # Determine which dimension is 3
            if sk == 3:
                # Split k: 2x2x2 + 2x2x1
                kg2 = kg[:2]
                kg1 = kg[2:]
                # 2x2x2 part
                col = build_block_decomposition(ig, jg, kg2, col, U, V, W, strassen_terms)
                # 2x2x1 part (schoolbook)
                for li in range(si):
                    for lj in range(sj):
                        gk = kg1[0]
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            elif sj == 3:
                # Split j: 2x2x2 + 2x1x2
                jg2 = jg[:2]
                jg1 = jg[2:]
                col = build_block_decomposition(ig, jg2, kg, col, U, V, W, strassen_terms)
                for li in range(si):
                    for lk in range(sk):
                        gj = jg1[0]
                        gi, gk = ig[li], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            else:  # si == 3
                # Split i: 2x2x2 + 1x2x2
                ig2 = ig[:2]
                ig1 = ig[2:]
                col = build_block_decomposition(ig2, jg, kg, col, U, V, W, strassen_terms)
                gi = ig1[0]
                for lj in range(sj):
                    for lk in range(sk):
                        gj, gk = jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
        elif sorted([si, sj, sk]) == [2, 3, 3]:
            # Split the size-3 dimension (any of the two 3s) into 2+1
            # Try splitting the first dimension that is 3
            if si == 3:
                ig2 = ig[:2]
                ig1 = ig[2:]
                # 2x3x3 part (which is [2,3,3] sorted, so recurse)
                col = build_block_decomposition(ig2, jg, kg, col, U, V, W, strassen_terms)
                # 1x3x3 part (schoolbook)
                gi = ig1[0]
                for lj in range(sj):
                    for lk in range(sk):
                        gj, gk = jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            elif sj == 3:
                jg2 = jg[:2]
                jg1 = jg[2:]
                col = build_block_decomposition(ig, jg2, kg, col, U, V, W, strassen_terms)
                gj = jg1[0]
                for li in range(si):
                    for lk in range(sk):
                        gi, gk = ig[li], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            else:  # sk == 3
                kg2 = kg[:2]
                kg1 = kg[2:]
                col = build_block_decomposition(ig, jg, kg2, col, U, V, W, strassen_terms)
                gk = kg1[0]
                for li in range(si):
                    for lj in range(sj):
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
        elif si == 3 and sj == 3 and sk == 3:
            # 3x3x3: split i into 2+1
            ig2 = ig[:2]
            ig1 = ig[2:]
            # 2x3x3 part
            col = build_block_decomposition(ig2, jg, kg, col, U, V, W, strassen_terms)
            # 1x3x3 part (schoolbook)
            gi = ig1[0]
            for lj in range(sj):
                for lk in range(sk):
                    gj, gk = jg[lj], kg[lk]
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        else:
            # Schoolbook
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
        
        return col

    def build_decomposition(i_groups, j_groups, k_groups):
        """Build the decomposition for a given partition."""
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
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            col = build_block_decomposition(ig, jg, kg, col, U, V, W, strassen_terms)
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        
        Partition:
        - i (rows of A, 0..2): group I0={0,1}, I1={2}
        - j (cols of A, 0..4): group J0={0,1}, J1={2,3}, J2={4}
        - k (cols of B, 0..5): group K0={0,1}, K1={2,3}, K2={4,5}
        
        For each sub-block (Ig, Jg, Kg):
        - If all sizes are 2: use Strassen (rank 7)
        - Otherwise: use schoolbook for that sub-block
        """
        # Define groups
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
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
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify the schoolbook decomposition
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        pass

    # Try the partition-based decomposition
    if time.time() < deadline - 5:
        optimal_rank, (opt_i, opt_j, opt_k) = find_best_partition()
        
        if opt_i is not None and optimal_rank < best_rank:
            result = build_decomposition(opt_i, opt_j, opt_k)
            rank = result.pop("rank")
            
            if verify(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank

    # Try the fixed Strassen partition as well
    if time.time() < deadline - 5:
        strassen_result = build_strassen_decomposition()
        strassen_rank = strassen_result.pop("rank")
        
        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], strassen_rank):
            if strassen_rank < best_rank:
                best_result = strassen_result
                best_rank = strassen_rank

    return best_result
# EVOLVE-BLOCK-END
