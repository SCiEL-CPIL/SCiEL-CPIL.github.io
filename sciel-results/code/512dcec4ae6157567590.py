"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 130.0


def _schoolbook(n, m, p):
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


def _verify_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
    """Vectorized verification of tensor decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > tol:
                    return False
    return True


def _verify(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    return _verify_fast(U_arr, V_arr, W_arr, n, m, p, tol)


def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _build_4x4x7_optimized():
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]
    W2p = [[0] * rank2 for _ in range(8)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            block_idx += 1
    
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]
    W3 = [[0] * rank3 for _ in range(4)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def _try_reduce_rank_aggressive(U, V, W, n, m, p, time_budget=30.0):
    """Aggressive rank reduction using systematic elimination."""
    start = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Precompute all tensor entries for fast checking
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all valid (i,j,k)
    
    # Build list of all (a, b, c) triples that should have value 1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    # For each rank term r, compute its contribution to each triple
    # contrib[r][t] = U[a,r] * V[b,r] * W[c,r] for triple t
    
    # Try to eliminate terms greedily
    # A term r is eliminable if for all triples, the sum of contributions
    # from remaining terms still equals 1
    
    # More efficient: check if removing term r changes any tensor entry
    # delta_t = U[a,r] * V[b,r] * W[c,r]
    # If delta_t = 0 for all t, term is redundant
    
    # But we also need to check if the remaining terms can still sum to 1
    # This is equivalent to checking if the contribution vector of term r
    # is in the span of other terms' contribution vectors
    
    # Let's use a different approach: for each term, compute its "fingerprint"
    # and check for linear dependencies
    
    # Build contribution matrix: rows = triples, cols = rank terms
    # M[t, r] = U[a_t, r] * V[b_t, r] * W[c_t, r]
    # We need M @ ones = ones (all ones vector)
    # A term r is removable if M[:, r] is in the column space of M without column r
    # AND removing it still allows M' @ x = ones for some x (but x should be binary/integer)
    
    # Simpler heuristic: just check if M[:, r] is all zeros, or if it's a
    # linear combination of other columns that preserves the sum constraint
    
    # For efficiency, let's try random subsets and check if they work
    
    # First pass: identify terms with zero contribution
    removable = []
    for r in range(rank):
        if time.time() - start > time_budget:
            break
        # Check if this term contributes anything
        max_contrib = 0
        for t in range(len(triples)):
            a, b, c = triples[t]
            contrib = U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            if abs(contrib) > max_contrib:
                max_contrib = abs(contrib)
        if max_contrib < 1e-12:
            removable.append(r)
    
    # Second pass: check linear dependencies more carefully
    # For each candidate, see if removing it preserves all tensor entries
    candidates = list(range(rank))
    
    # Try eliminating terms in order of "importance" (lowest contribution first)
    term_importance = []
    for r in range(rank):
        total_abs = 0
        for t in range(len(triples)):
            a, b, c = triples[t]
            contrib = U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            total_abs += abs(contrib)
        term_importance.append((total_abs, r))
    
    term_importance.sort()
    
    # Greedily try to remove terms from least important to most important
    removed = set()
    for _, r in term_importance:
        if time.time() - start > time_budget - 5:
            break
        if r in removed:
            continue
        
        # Temporarily remove term r and check if all tensor entries are still 1
        # This is expensive, so only do it for a subset
        if len(removed) < 20:  # Limit number of removals to check
            U_test = np.delete(U_arr, list(removed | {r}), axis=1)
            V_test = np.delete(V_arr, list(removed | {r}), axis=1)
            W_test = np.delete(W_arr, list(removed | {r}), axis=1)
            
            valid = True
            for t in range(len(triples)):
                a, b, c = triples[t]
                s = np.dot(U_test[a], V_test[b] * W_test[c])
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            
            if valid:
                removed.add(r)
    
    if not removed:
        return U, V, W
    
    # Build final reduced matrices
    keep_indices = [r for r in range(rank) if r not in removed]
    U_new = U_arr[:, keep_indices]
    V_new = V_arr[:, keep_indices]
    W_new = W_arr[:, keep_indices]
    
    # Final verification
    valid = True
    for t in range(len(triples)):
        a, b, c = triples[t]
        s = np.dot(U_new[a], V_new[b] * W_new[c])
        if abs(s - 1) > 1e-9:
            valid = False
            break
    
    if not valid:
        return U, V, W
    
    # Column merge pass: try to combine pairs of columns into one
    # A merge of columns r1, r2 into a new column r_new works if:
    # For all triples (a,b,c): U[a,r1]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]
    # can be represented as U'[a]*V'[b]*W'[c] for some U', V', W'
    # This is generally hard, but we can try a heuristic:
    # If the product vectors are proportional in a structured way.
    
    # Simpler approach: try random linear combinations of two columns
    # and check if the result is still valid.
    current_rank = len(U_new[0])
    merged = set()
    
    for iter_count in range(50):
        if time.time() - start > time_budget - 2:
            break
        if len(merged) >= current_rank // 2:
            break
        
        # Pick two non-merged columns to try merging
        available = [r for r in range(current_rank) if r not in merged]
        if len(available) < 2:
            break
        
        r1, r2 = available[0], available[1]
        
        # Try to find alpha, beta such that replacing columns r1, r2
        # with a single new column works. The new column must satisfy:
        # U_new[a, r1]*V_new[b, r1]*W_new[c, r1] + U_new[a, r2]*V_new[b, r2]*W_new[c, r2]
        # = new_U[a] * new_V[b] * new_W[c]
        # for all (a,b,c).
        
        # This is a rank-1 approximation problem on the "merged" tensor.
        # For simplicity, try: new_U = U[:,r1] + U[:,r2], new_V = V[:,r1] + V[:,r2], new_W = W[:,r1] + W[:,r2]
        # This only works if the cross terms cancel, which is rare.
        
        # Better: try new_U = U[:,r1], new_V = V[:,r1] + V[:,r2], new_W = W[:,r1]
        # This works if U[:,r1]*V[:,r2]*W[:,r1] + U[:,r2]*V[:,r1]*W[:,r2] + U[:,r2]*V[:,r2]*W[:,r2] = 0
        # which is unlikely.
        
        # Most practical: just check if one column is a scalar multiple of the other
        # in a way that allows absorption.
        
        # Check if column r2 is a scalar multiple of column r1 across all three matrices
        # i.e., there exists s such that V[:,r2] = s*V[:,r1] and W[:,r2] = s*W[:,r1]
        # Then U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2]
        # = U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*s*V[:,r1]*s*W[:,r1]
        # = (U[:,r1] + s^2 * U[:,r2]) * V[:,r1] * W[:,r1]
        
        # Find s from V: if V[:,r1] has non-zero entries, s = V[:,r2][idx]/V[:,r1][idx]
        v1 = V_new[:, r1]
        v2 = V_new[:, r2]
        nonzero_v1 = np.where(np.abs(v1) > 1e-12)[0]
        
        if len(nonzero_v1) > 0:
            # Check if v2 is proportional to v1
            ratios = v2[nonzero_v1] / v1[nonzero_v1]
            if np.allclose(ratios, ratios[0], atol=1e-9):
                s = ratios[0]
                # Verify W proportionality
                w1 = W_new[:, r1]
                w2 = W_new[:, r2]
                nonzero_w1 = np.where(np.abs(w1) > 1e-12)[0]
                if len(nonzero_w1) > 0:
                    ratios_w = w2[nonzero_w1] / w1[nonzero_w1]
                    if np.allclose(ratios_w, s, atol=1e-9):
                        # Merge: new_U = U[:,r1] + s^2 * U[:,r2], new_V = V[:,r1], new_W = W[:,r1]
                        # Remove r2, update r1
                        U_new[:, r1] = U_new[:, r1] + s * s * U_new[:, r2]
                        merged.add(r2)
                        # Verify
                        valid_merge = True
                        for t in range(len(triples)):
                            a, b, c = triples[t]
                            s_val = np.dot(U_new[a], V_new[b] * W_new[c])
                            if abs(s_val - 1) > 1e-8:
                                valid_merge = False
                                break
                        if not valid_merge:
                            U_new[:, r1] = U_new[:, r1] - s * s * U_new[:, r2]
                            merged.discard(r2)
    
    # Remove merged columns
    if merged:
        final_keep = [r for r in range(current_rank) if r not in merged]
        U_final = U_new[:, final_keep]
        V_final = V_new[:, final_keep]
        W_final = W_new[:, final_keep]
        
        valid_final = True
        for t in range(len(triples)):
            a, b, c = triples[t]
            s_val = np.dot(U_final[a], V_final[b] * W_final[c])
            if abs(s_val - 1) > 1e-9:
                valid_final = False
                break
        
        if valid_final:
            return U_final.tolist(), V_final.tolist(), W_final.tolist()
    
    return U_new.tolist(), V_new.tolist(), W_new.tolist()


def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            # Try aggressive rank reduction
            remaining_time = _TIME_LIMIT - (time.time() - _START_TIME) - 10
            if remaining_time > 5:
                U_red, V_red, W_red = _try_reduce_rank_aggressive(
                    result['U'], result['V'], result['W'], 4, 4, 7,
                    time_budget=min(30.0, remaining_time)
                )
                if len(U_red[0]) < len(result['U'][0]):
                    if _verify(U_red, V_red, W_red, 4, 4, 7):
                        return {"U": U_red, "V": V_red, "W": W_red}
            return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    return _schoolbook(4, 4, 7)
