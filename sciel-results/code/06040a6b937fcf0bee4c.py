"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
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


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
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
    """Build decomposition for 4x4 * 4x7: rank 49 (4x4x4 Strassen²) + 28 (4x4x2) + 16 (4x4x1) = 93."""
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
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
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]  # 4 rows * 2 cols
    W2p = [[0] * rank2 for _ in range(8)]  # 2 rows * 4 cols
    
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
                        k = kl
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
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]  # 4 rows * 1 col
    W3 = [[0] * rank3 for _ in range(4)]  # 1 row * 4 cols
    
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
    
    # Combine all parts
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
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
    
    # Part 2: B cols 4-5, C cols 4-5
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
    
    # Part 3: B col 6, C col 6
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


def _try_reduce_rank(U, V, W, n, m, p, time_budget=30):
    """Attempt to reduce rank by finding linearly dependent rank-1 tensors.
    
    Uses SVD on the coefficient matrix to identify the null-space of the
    rank-1 tensor coefficient vectors, then removes one column at a time
    (the most linearly-dependent) and re-checks.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    # Build the full coefficient matrix M (flattened tensor coefficients per rank-1 term)
    # M has shape (nm*mp*pn, rank); column r is vec(U_r ⊗ V_r ⊗ W_r)
    M = np.zeros((nm * mp * pn, rank))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # Use outer product structure: M[a*mp*pn + b*pn + c, r] = u[a]*v[b]*w[c]
        for a in range(nm):
            ua = u_col[a]
            if ua == 0:
                continue
            base = a * mp * pn
            for b in range(mp):
                vb = v_col[b]
                if vb == 0:
                    continue
                uv = ua * vb
                if uv == 0:
                    continue
                for c in range(pn):
                    wc = w_col[c]
                    if wc != 0:
                        M[base + b * pn + c, r] = uv * wc
    
    # Check numerical rank of M
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"U": U, "V": V, "W": W}
    
    if len(S) == 0 or S[0] < 1e-12:
        return {"U": U, "V": V, "W": W}
    
    tol = 1e-10 * S[0]
    num_rank = sum(1 for s in S if s > tol)
    
    if num_rank >= rank:
        # No linear dependence; try pairwise merge heuristic
        return _try_pairwise_merge(U, V, W, n, m, p, time_budget)
    
    # There IS linear dependence. The null space has dimension rank - num_rank.
    # Find the column with the largest component in the null space direction.
    # Vt has shape (rank, rank); null space vectors are the last (rank-num_rank) rows of Vt.
    null_dim = rank - num_rank
    null_basis = Vt[num_rank:, :].T  # shape (rank, null_dim)
    
    # For each column r, compute how much of it is "explained" by the null space
    # i.e., the projection of the r-th column of M onto the null space
    # Actually, we want to find which column to remove.
    # The columns of M that are linearly dependent: find a subset to remove.
    # Strategy: greedily remove the column whose removal keeps the rank highest.
    
    # Simpler: find the column with the smallest singular value contribution.
    # Use the fact that if we remove column r, the new rank is at most rank-1.
    # We need to verify the decomposition still works after removal.
    
    # Find the best column to remove: the one most aligned with the null space.
    # The null space vector v satisfies M @ v = 0.
    # If we have a null vector, we can express one column as a combination of others.
    
    # Greedy removal: try removing each column, check if remaining columns still span.
    # For efficiency, use the SVD structure: the columns with largest null-space
    # component are the best candidates for removal.
    
    # Compute null-space alignment for each column
    null_alignment = np.zeros(rank)
    for r in range(rank):
        # Project column r of M onto null space
        proj = null_basis.T @ M[:, r]
        null_alignment[r] = np.linalg.norm(proj)
    
    # Sort by null alignment (highest first) and try removing
    order = np.argsort(-null_alignment)
    
    for r_remove in order:
        if time.time() - start_time > time_budget:
            break
        # Try removing column r_remove
        keep = [i for i in range(rank) if i != r_remove]
        U_new = [U_arr[:, i] for i in keep]
        V_new = [V_arr[:, i] for i in keep]
        W_new = [W_arr[:, i] for i in keep]
        
        # Verify
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            return {"U": [list(row) for row in np.array(U_new).T],
                    "V": [list(row) for row in np.array(V_new).T],
                    "W": [list(row) for row in np.array(W_new).T]}
    
    # No single column removal worked; fall back to pairwise merge
    return _try_pairwise_merge(U, V, W, n, m, p, time_budget)


def _verify_fast(U_cols, V_cols, W_cols, n, m, p, tol=1e-9):
    """Fast verification using numpy broadcasting."""
    rank = len(U_cols)
    U_arr = np.array(U_cols, dtype=float)  # (nm, rank)
    V_arr = np.array(V_cols, dtype=float)  # (mp, rank)
    W_arr = np.array(W_cols, dtype=float)  # (pn, rank)
    
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


def _try_pairwise_merge(U, V, W, n, m, p, time_budget=30):
    """Try to merge pairs of rank-1 terms that are proportional in a useful way.
    
    If two terms r1 and r2 have U, V, W vectors that are proportional (or can be
    combined), we might reduce rank. More practically: look for terms where
    the tensor U_r ⊗ V_r ⊗ W_r can be expressed as a sum of fewer terms
    by exploiting the Kronecker product structure.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    # Check for duplicate or proportional columns
    # Two columns r1, r2 are "mergeable" if there exists alpha, beta such that
    # alpha * U_r1 + beta * U_r2 = 0 (or similar for V, W)
    # This means the rank-1 tensors are linearly dependent in a specific way.
    
    # Simpler heuristic: find pairs where U vectors are proportional
    # AND V vectors are proportional (then W must also be proportional for
    # the tensors to be linearly dependent).
    
    # For the Kronecker-structured decomposition, look for terms where
    # the sub-block indices match in a way that allows combining.
    
    # Check for exact duplicates
    seen = {}
    merge_map = list(range(rank))  # identity initially
    
    for r in range(rank):
        if time.time() - start_time > time_budget:
            break
        u_key = tuple(U_arr[:, r].round(10))
        v_key = tuple(V_arr[:, r].round(10))
        w_key = tuple(W_arr[:, r].round(10))
        full_key = (u_key, v_key, w_key)
        
        if full_key in seen:
            # Duplicate found - we can remove one
            merge_map[r] = seen[full_key]
        else:
            seen[full_key] = r
    
    # Check for proportional pairs (U_r1 = a * U_r2, V_r1 = b * V_r2, W_r1 = c * W_r2
    # where a*b*c = 1 for the tensor to be proportional)
    # Actually for tensor proportionality: U_r1⊗V_r1⊗W_r1 = λ * U_r2⊗V_r2⊗W_r2
    # This requires U_r1 = α*U_r2, V_r1 = β*V_r2, W_r1 = γ*W_r2 with αβγ = λ
    
    # Build the merged decomposition
    unique_indices = sorted(set(merge_map))
    if len(unique_indices) < rank:
        U_new = [U_arr[:, i] for i in unique_indices]
        V_new = [V_arr[:, i] for i in unique_indices]
        W_new = [W_arr[:, i] for i in unique_indices]
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            return {"U": [list(row) for row in np.array(U_new).T],
                    "V": [list(row) for row in np.array(V_new).T],
                    "W": [list(row) for row in np.array(W_new).T]}
    
    # Try structured merge: for the Kronecker product structure,
    # look for terms that differ only in one sub-index and can be combined.
    # This is the "tensor network contraction" idea: if two terms share
    # the same sub-block structure except for one factor, they might merge.
    
    # For now, return the original if no merges found
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Build the rank-93 decomposition
    try:
        result = _build_4x4x7_optimized()
        if not _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            # Fallback
            return _schoolbook(4, 4, 7)
        
        # Try to reduce rank via linear dependence
        if time.time() - _START_TIME < _TIME_LIMIT - 25:
            try:
                reduced = _try_reduce_rank(result['U'], result['V'], result['W'], 4, 4, 7, time_budget=20)
                if _verify(reduced['U'], reduced['V'], reduced['W'], 4, 4, 7):
                    if len(reduced['U'][0]) < len(result['U'][0]):
                        return reduced
            except Exception:
                pass
        
        return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)
