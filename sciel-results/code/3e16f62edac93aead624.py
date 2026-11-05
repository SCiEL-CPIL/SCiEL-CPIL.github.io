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
    nm, mp, pn = n * m, m * p, p * n

    def _strassen_2x2x2_terms():
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

    def _verify_fast(U, V, W):
        """Vectorized verification using einsum."""
        U_np = np.asarray(U, dtype=float)
        V_np = np.asarray(V, dtype=float)
        W_np = np.asarray(W, dtype=float)
        r = U_np.shape[1]
        M_all = np.einsum('cr,ar,br->cab', W_np, U_np, V_np)
        expected = np.zeros((pn, nm, mp), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[k*n+i, i*m+j, j*p+k] = 1.0
        return np.max(np.abs(M_all - expected)) < 1e-9

    def _verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i * m + j
                    v_r = j * p + k
                    w_r = k * n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def _schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_block_decomp(i_groups, j_groups, k_groups):
        """Build decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere."""
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
        
        U = [[0] * total_rank for _ in range(nm)]
        V = [[0] * total_rank for _ in range(mp)]
        W = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        strassen_terms = _strassen_2x2x2_terms()
        
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

    def _generate_contiguous_partitions(dim):
        """Generate all contiguous partitions of [0..dim) into chunks of size 1 or 2."""
        results = []
        def rec(start, cur):
            if start == dim:
                results.append(list(cur))
                return
            if start + 1 <= dim:
                cur.append([start])
                rec(start + 1, cur)
                cur.pop()
            if start + 2 <= dim:
                cur.append([start, start + 1])
                rec(start + 2, cur)
                cur.pop()
        rec(0, [])
        return results

    def _prune_columns(U, V, W, rank, max_iters=2000):
        """Iteratively remove redundant columns via random and systematic search."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            if rank == 0:
                break
            
            found = False
            for _ in range(min(10, rank)):
                col = random.randint(0, rank - 1)
                
                cand_U = []
                for row in U:
                    cand_row = row[:col] + row[col+1:]
                    cand_U.append(cand_row)
                
                cand_V = []
                for row in V:
                    cand_row = row[:col] + row[col+1:]
                    cand_V.append(cand_row)
                
                cand_W = []
                for row in W:
                    cand_row = row[:col] + row[col+1:]
                    cand_W.append(cand_row)
                
                if _verify_sparse(cand_U, cand_V, cand_W, rank - 1):
                    U, V, W = cand_U, cand_V, cand_W
                    rank -= 1
                    found = True
                    break
            
            if not found and attempt % 100 == 0:
                for col in range(rank):
                    cand_U = []
                    for row in U:
                        cand_row = row[:col] + row[col+1:]
                        cand_U.append(cand_row)
                    cand_V = []
                    for row in V:
                        cand_row = row[:col] + row[col+1:]
                        cand_V.append(cand_row)
                    cand_W = []
                    for row in W:
                        cand_row = row[:col] + row[col+1:]
                        cand_W.append(cand_row)
                    if _verify_sparse(cand_U, cand_V, cand_W, rank - 1):
                        U, V, W = cand_U, cand_V, cand_W
                        rank -= 1
                        break
                    if time.time() > deadline - 5:
                        break
        
        return U, V, W, rank

    def _svd_based_decomposition(target_rank):
        """
        Use SVD on tensor matricizations to find low-rank approximations,
        then refine to exact rational values.
        """
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            U_rand = np.random.randn(nm, target_rank) * 0.1
            V_rand = np.random.randn(mp, target_rank) * 0.1
            W_rand = np.random.randn(pn, target_rank) * 0.1
            
            current_err = float('inf')
            U_cur = U_rand.copy()
            V_cur = V_rand.copy()
            W_cur = W_rand.copy()
            
            for step in range(5000):
                if time.time() > deadline - 10:
                    break
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_cur[i*m+j] * V_cur[j*p+k] * W_cur[k*n+i])
                            err += (val - 1.0) ** 2
                
                if err < current_err:
                    current_err = err
                    if err < 1e-12:
                        break
                    if err < best_err:
                        best_err = err
                        best_U = U_cur.copy()
                        best_V = V_cur.copy()
                        best_W = W_cur.copy()
                
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = U_cur[row, col]
                    U_cur[row, col] = old_val + np.random.randn() * 0.05
                elif mat_choice == 'V':
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = V_cur[row, col]
                    V_cur[row, col] = old_val + np.random.randn() * 0.05
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = W_cur[row, col]
                    W_cur[row, col] = old_val + np.random.randn() * 0.05
            
            if best_err < 1e-10:
                break
        
        if best_err > 1e-8:
            return None
        
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        return U_list, V_list, W_list

    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = n * m * p

    # Phase 1: Block decomposition search
    i_splits = _generate_contiguous_partitions(n)
    j_splits = _generate_contiguous_partitions(m)
    k_splits = _generate_contiguous_partitions(p)
    
    for i_g in i_splits:
        for j_g in j_splits:
            for k_g in k_splits:
                if time.time() > deadline:
                    break
                
                total_rank = 0
                for ig in i_g:
                    for jg in j_g:
                        for kg in k_g:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                total_rank += 7
                            else:
                                total_rank += si * sj * sk
                
                if total_rank >= best_rank:
                    continue
                
                result = _build_block_decomp(i_g, j_g, k_g)
                
                if _verify_sparse(result["U"], result["V"], result["W"], total_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = total_rank

    # Phase 2: Prune columns from best block decomposition
    if best_rank < n * m * p and time.time() < deadline - 15:
        pruned_U, pruned_V, pruned_W, pruned_rank = _prune_columns(
            best_result["U"], best_result["V"], best_result["W"], best_rank
        )
        if pruned_rank < best_rank:
            best_result = {"U": pruned_U, "V": pruned_V, "W": pruned_W}
            best_rank = pruned_rank

    # Phase 3: SVD-based search for lower ranks
    for target_rank in range(max(1, best_rank - 5), best_rank):
        if time.time() > deadline - 20:
            break
        
        result = _svd_based_decomposition(target_rank)
        if result is None:
            continue
        
        U_list, V_list, W_list = result
        
        if _verify_sparse(U_list, V_list, W_list, target_rank):
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break

    # Phase 4: Final aggressive pruning
    if best_rank < n * m * p and time.time() < deadline - 5:
        pruned_U, pruned_V, pruned_W, pruned_rank = _prune_columns(
            best_result["U"], best_result["V"], best_result["W"], best_rank
        )
        if pruned_rank < best_rank:
            best_result = {"U": pruned_U, "V": pruned_V, "W": pruned_W}
            best_rank = pruned_rank

    return best_result
# EVOLVE-BLOCK-END
