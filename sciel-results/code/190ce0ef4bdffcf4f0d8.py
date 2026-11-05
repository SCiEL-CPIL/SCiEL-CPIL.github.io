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

    def _verify_numpy(U, V, W, rank):
        """Fast vectorized verification using einsum."""
        U_np = np.asarray(U, dtype=float)
        V_np = np.asarray(V, dtype=float)
        W_np = np.asarray(W, dtype=float)
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

    def _prune_columns_fast(U, V, W, rank, max_iters=5000):
        """Iteratively remove redundant columns using numpy-accelerated checks."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        iterations = 0
        while iterations < max_iters:
            if time.time() > deadline - 5:
                break
            if rank == 0:
                break
            iterations += 1
            
            found = False
            # Try random columns first for speed
            cols_to_try = random.sample(range(rank), min(rank, 20))
            # Add systematic sweep every 50 iterations
            if iterations % 50 == 0:
                cols_to_try = list(range(rank))
            
            for col in cols_to_try:
                # Quick check: if column is all zeros, remove it
                if all(U[r][col] == 0 for r in range(nm)) and \
                   all(V[r][col] == 0 for r in range(mp)) and \
                   all(W[r][col] == 0 for r in range(pn)):
                    U = [row[:col] + row[col+1:] for row in U]
                    V = [row[:col] + row[col+1:] for row in V]
                    W = [row[:col] + row[col+1:] for row in W]
                    rank -= 1
                    found = True
                    break
                
                # Check if removing this column preserves exactness
                # Use numpy for speed
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                
                U_cand = np.delete(U_np, col, axis=1)
                V_cand = np.delete(V_np, col, axis=1)
                W_cand = np.delete(W_np, col, axis=1)
                
                M_all = np.einsum('cr,ar,br->cab', W_cand, U_cand, V_cand)
                expected = np.zeros((pn, nm, mp), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected[k*n+i, i*m+j, j*p+k] = 1.0
                
                if np.max(np.abs(M_all - expected)) < 1e-9:
                    U = U_cand.tolist()
                    V = V_cand.tolist()
                    W = W_cand.tolist()
                    rank -= 1
                    found = True
                    break
            
            if not found:
                # Try a few more random attempts before giving up on this pass
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    col = random.randint(0, rank - 1)
                    U_np = np.array(U, dtype=float)
                    V_np = np.array(V, dtype=float)
                    W_np = np.array(W, dtype=float)
                    
                    U_cand = np.delete(U_np, col, axis=1)
                    V_cand = np.delete(V_np, col, axis=1)
                    W_cand = np.delete(W_np, col, axis=1)
                    
                    M_all = np.einsum('cr,ar,br->cab', W_cand, U_cand, V_cand)
                    expected = np.zeros((pn, nm, mp), dtype=float)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                expected[k*n+i, i*m+j, j*p+k] = 1.0
                    
                    if np.max(np.abs(M_all - expected)) < 1e-9:
                        U = U_cand.tolist()
                        V = V_cand.tolist()
                        W = W_cand.tolist()
                        rank -= 1
                        found = True
                        break
                
                if not found:
                    break
        
        return U, V, W, rank

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

    # Phase 2: Aggressive pruning from best block decomposition
    if best_rank < n * m * p and time.time() < deadline - 20:
        pruned_U, pruned_V, pruned_W, pruned_rank = _prune_columns_fast(
            best_result["U"], best_result["V"], best_result["W"], best_rank
        )
        if pruned_rank < best_rank:
            best_result = {"U": pruned_U, "V": pruned_V, "W": pruned_W}
            best_rank = pruned_rank

    # Phase 3: Try to find even lower rank by perturbing and re-pruning
    if best_rank < n * m * p and time.time() < deadline - 10:
        U_np = np.array(best_result["U"], dtype=float)
        V_np = np.array(best_result["V"], dtype=float)
        W_np = np.array(best_result["W"], dtype=float)
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(123 + trial)
            # Small perturbation to escape local minima
            U_pert = U_np + np.random.randn(*U_np.shape) * 1e-4
            V_pert = V_np + np.random.randn(*V_np.shape) * 1e-4
            W_pert = W_np + np.random.randn(*W_np.shape) * 1e-4
            
            # Round to nearest rational with small denominator
            U_round = np.round(U_pert * 8) / 8
            V_round = np.round(V_pert * 8) / 8
            W_round = np.round(W_pert * 8) / 8
            
            U_list = U_round.tolist()
            V_list = V_round.tolist()
            W_list = W_round.tolist()
            
            if _verify_sparse(U_list, V_list, W_list, best_rank):
                pruned_U2, pruned_V2, pruned_W2, pruned_rank2 = _prune_columns_fast(
                    U_list, V_list, W_list, best_rank
                )
                if pruned_rank2 < best_rank:
                    best_result = {"U": pruned_U2, "V": pruned_V2, "W": pruned_W2}
                    best_rank = pruned_rank2
                    break

    # Phase 4: Final aggressive pruning
    if best_rank < n * m * p and time.time() < deadline - 5:
        pruned_U, pruned_V, pruned_W, pruned_rank = _prune_columns_fast(
            best_result["U"], best_result["V"], best_result["W"], best_rank
        )
        if pruned_rank < best_rank:
            best_result = {"U": pruned_U, "V": pruned_V, "W": pruned_W}
            best_rank = pruned_rank

    return best_result
# EVOLVE-BLOCK-END
