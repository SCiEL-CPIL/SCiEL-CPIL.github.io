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

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i*m + j
                    v_r = j*p + k
                    w_r = k*n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

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

    def strassen_2x2x2_terms():
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

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
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
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def recursive_strassen_rank(i_size, j_size, k_size):
        """
        Compute the rank needed for an i_size x j_size x k_size block
        using recursive Strassen decomposition.
        Base cases: if any dimension is 1, rank = product.
        If all dimensions >= 2, split each into two halves (pad if odd) and use Strassen.
        """
        if time.time() > deadline:
            return i_size * j_size * k_size
        
        # Base case: any dimension is 1
        if i_size == 1 or j_size == 1 or k_size == 1:
            return i_size * j_size * k_size
        
        # If all dimensions are exactly 2, use Strassen directly
        if i_size == 2 and j_size == 2 and k_size == 2:
            return 7
        
        # For larger blocks, try splitting
        # Split each dimension into two parts as evenly as possible
        i1 = i_size // 2
        i2 = i_size - i1
        j1 = j_size // 2
        j2 = j_size - j1
        k1 = k_size // 2
        k2 = k_size - k1
        
        # Recursively compute ranks for the 8 sub-blocks
        r11 = recursive_strassen_rank(i1, j1, k1)
        r12 = recursive_strassen_rank(i1, j1, k2)
        r13 = recursive_strassen_rank(i1, j2, k1)
        r14 = recursive_strassen_rank(i1, j2, k2)
        r21 = recursive_strassen_rank(i2, j1, k1)
        r22 = recursive_strassen_rank(i2, j1, k2)
        r23 = recursive_strassen_rank(i2, j2, k1)
        r24 = recursive_strassen_rank(i2, j2, k2)
        
        # Strassen combines 8 sub-problems into 7 multiplications
        # But we need to be careful: the 7 Strassen products each involve
        # sub-blocks of specific sizes
        # M1 = (A11+A22)(B11+B22) -> rank of (i1+i2)x(j1+j2)x(k1+k2) block... 
        # Actually, this is more complex. Let's use a simpler approach:
        # Just use the block decomposition where each 2x2x2 sub-block uses Strassen
        # and everything else uses schoolbook.
        
        # Simpler: for this specific problem (3,5,6), let's try different groupings
        return i_size * j_size * k_size

    def build_recursive_strassen():
        """
        Build decomposition using recursive Strassen.
        For 3x5x6: split into blocks where possible 2x2x2 sub-blocks use Strassen.
        """
        # For 3x5x6, the best approach is to find all possible 2x2x2 sub-blocks
        # and apply Strassen to them, with schoolbook for the rest.
        # 
        # i: 0,1,2 -> groups: [0,1], [2]
        # j: 0,1,2,3,4 -> groups: [0,1], [2,3], [4]
        # k: 0,1,2,3,4,5 -> groups: [0,1], [2,3], [4,5]
        #
        # This gives blocks:
        # [0,1]x[0,1]x[0,1] -> 2x2x2 -> Strassen (7)
        # [0,1]x[0,1]x[2,3] -> 2x2x2 -> Strassen (7)
        # [0,1]x[0,1]x[4,5] -> 2x2x2 -> Strassen (7)
        # [0,1]x[2,3]x[0,1] -> 2x2x2 -> Strassen (7)
        # [0,1]x[2,3]x[2,3] -> 2x2x2 -> Strassen (7)
        # [0,1]x[2,3]x[4,5] -> 2x2x2 -> Strassen (7)
        # [0,1]x[4]x[0,1] -> 2x1x2 -> schoolbook (4)
        # [0,1]x[4]x[2,3] -> 2x1x2 -> schoolbook (4)
        # [0,1]x[4]x[4,5] -> 2x1x2 -> schoolbook (4)
        # [2]x[0,1]x[0,1] -> 1x2x2 -> schoolbook (4)
        # [2]x[0,1]x[2,3] -> 1x2x2 -> schoolbook (4)
        # [2]x[0,1]x[4,5] -> 1x2x2 -> schoolbook (4)
        # [2]x[2,3]x[0,1] -> 1x2x2 -> schoolbook (4)
        # [2]x[2,3]x[2,3] -> 1x2x2 -> schoolbook (4)
        # [2]x[2,3]x[4,5] -> 1x2x2 -> schoolbook (4)
        # [2]x[4]x[0,1] -> 1x1x2 -> schoolbook (2)
        # [2]x[4]x[2,3] -> 1x1x2 -> schoolbook (2)
        # [2]x[4]x[4,5] -> 1x1x2 -> schoolbook (2)
        #
        # Total: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
        #
        # Alternative: try different groupings to get more 2x2x2 blocks
        # j: 0,1,2,3,4 -> [0,1], [2,3], [4] gives 3 groups
        # What about [0], [1,2], [3,4]?
        # Then:
        # [0,1]x[1,2]x[0,1] -> 2x2x2 -> 7
        # [0,1]x[1,2]x[2,3] -> 2x2x2 -> 7
        # [0,1]x[1,2]x[4,5] -> 2x2x2 -> 7
        # [0,1]x[0]x[0,1] -> 2x1x2 -> 4
        # [0,1]x[0]x[2,3] -> 2x1x2 -> 4
        # [0,1]x[0]x[4,5] -> 2x1x2 -> 4
        # [0,1]x[3,4]x[0,1] -> 2x2x2 -> 7
        # [0,1]x[3,4]x[2,3] -> 2x2x2 -> 7
        # [0,1]x[3,4]x[4,5] -> 2x2x2 -> 7
        # [2]x[1,2]x[0,1] -> 1x2x2 -> 4
        # [2]x[1,2]x[2,3] -> 1x2x2 -> 4
        # [2]x[1,2]x[4,5] -> 1x2x2 -> 4
        # [2]x[0]x[0,1] -> 1x1x2 -> 2
        # [2]x[0]x[2,3] -> 1x1x2 -> 2
        # [2]x[0]x[4,5] -> 1x1x2 -> 2
        # [2]x[3,4]x[0,1] -> 1x2x2 -> 4
        # [2]x[3,4]x[2,3] -> 1x2x2 -> 4
        # [2]x[3,4]x[4,5] -> 1x2x2 -> 4
        #
        # Total: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84 (same)
        
        # The key insight: we can't get below 84 with simple block Strassen for 3x5x6
        # because the odd dimensions prevent more 2x2x2 blocks.
        # 
        # However, we can try to find linear dependencies among the terms to reduce rank.
        
        return build_strassen_decomposition()

    def try_all_groupings():
        """
        Try all reasonable groupings of the three dimensions and use Strassen
        where possible, then look for linear dependencies.
        """
        strassen_terms = strassen_2x2x2_terms()
        best_result = None
        best_rank = n * m * p  # schoolbook
        
        # Generate all possible splits for each dimension
        def get_splits(size):
            """Generate all ways to split indices 0..size-1 into contiguous groups."""
            if size <= 1:
                return [list(range(size))]
            
            splits = []
            # Single group
            splits.append([list(range(size))])
            
            # Two groups
            for split in range(1, size):
                splits.append([list(range(split)), list(range(split, size))])
            
            # Three groups
            for s1 in range(1, size - 1):
                for s2 in range(s1 + 1, size):
                    splits.append([
                        list(range(s1)),
                        list(range(s1, s2)),
                        list(range(s2, size))
                    ])
            
            return splits
        
        i_splits = get_splits(n)
        j_splits = get_splits(m)
        k_splits = get_splits(p)
        
        for i_g in i_splits:
            for j_g in j_splits:
                for k_g in k_splits:
                    if time.time() > deadline - 5:
                        return best_result
                    
                    # Calculate rank for this grouping
                    total_rank = 0
                    block_ranks = []
                    for ig in i_g:
                        for jg in j_g:
                            for kg in k_g:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    r = 7
                                else:
                                    r = si * sj * sk
                                total_rank += r
                                block_ranks.append(r)
                    
                    if total_rank >= best_rank:
                        continue
                    
                    # Build the decomposition
                    U = [[0] * total_rank for _ in range(n * m)]
                    V = [[0] * total_rank for _ in range(m * p)]
                    W = [[0] * total_rank for _ in range(p * n)]
                    
                    col = 0
                    
                    for ig in i_g:
                        for jg in j_g:
                            for kg in k_g:
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
                    
                    # Verify using numpy for speed
                    U_np = np.array(U, dtype=float)
                    V_np = np.array(V, dtype=float)
                    W_np = np.array(W, dtype=float)
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_row = U_np[i*m+j, :]
                                v_row = V_np[j*p+k, :]
                                w_row = W_np[k*n+i, :]
                                val = np.sum(u_row * v_row * w_row)
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = total_rank
        
        return best_result

    def reduce_rank_by_dependency(U, V, W, max_iter=500):
        """
        Try to reduce rank by finding linear dependencies among the terms.
        For each column c, check if removing it still gives a valid decomposition.
        Also try: for each pair of columns, check if their sum/difference can be removed.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = len(U_np[0])
        
        # Precompute the tensor values for each (i,j,k)
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 for all i,j,k
        
        def is_valid(U_sub, V_sub, W_sub):
            """Check if the decomposition is valid."""
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U_sub[i*m+j, :]
                        v_row = V_sub[j*p+k, :]
                        w_row = W_sub[k*n+i, :]
                        val = np.sum(u_row * v_row * w_row)
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        # Try removing one column at a time
        changed = True
        while changed and max_iter > 0:
            changed = False
            max_iter -= 1
            
            rank = len(U_np[0])
            for c in range(rank - 1, -1, -1):
                if time.time() > deadline - 2:
                    return U, V, W
                
                # Remove column c
                cols = [cc for cc in range(rank) if cc != c]
                U_sub = U_np[:, cols]
                V_sub = V_np[:, cols]
                W_sub = W_np[:, cols]
                
                if is_valid(U_sub, V_sub, W_sub):
                    # Found a lower rank decomposition
                    U_np = U_sub
                    V_np = V_sub
                    W_np = W_sub
                    changed = True
                    break
            
            if not changed:
                break
        
        # Convert back to lists
        U_out = U_np.tolist()
        V_out = V_np.tolist()
        W_out = W_np.tolist()
        
        return U_out, V_out, W_out

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try Strassen-based decomposition
    strassen_result = build_strassen_decomposition()
    strassen_rank = strassen_result.pop("rank")
    
    if verify_sparse(strassen_result["U"], strassen_result["V"], strassen_result["W"], strassen_rank):
        if strassen_rank < best_rank:
            best_result = strassen_result
            best_rank = strassen_rank

    # Try all groupings to find the best block decomposition
    if time.time() < deadline - 10:
        all_groupings_result = try_all_groupings()
        if all_groupings_result is not None:
            ag_rank = len(all_groupings_result["U"][0])
            if ag_rank < best_rank:
                best_result = all_groupings_result
                best_rank = ag_rank

    # Reduce rank by finding linear dependencies
    if time.time() < deadline - 2 and best_rank > 50:
        U_red, V_red, W_red = reduce_rank_by_dependency(
            best_result["U"], best_result["V"], best_result["W"], max_iter=300
        )
        new_rank = len(U_red[0])
        if new_rank < best_rank:
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = new_rank

    return best_result
# EVOLVE-BLOCK-END
