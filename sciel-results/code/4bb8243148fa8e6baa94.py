"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Compute T[i,j,kp,jp,k,ip] = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
    # Reshape to 3D tensors
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i, j, jp, k, kp, ip] = sum_c U3[i,j,c] * V3[jp,k,c] * W3[kp,ip,c]
    # Use einsum: 'ijc,jkc,kpc->ijkjp'... let me think about indices
    # U[i,j,c], V[jp,k,c], W[kp,ip,c]
    # We want T[i,j,kp,jp,k,ip]
    T = np.einsum('ijc,jkc,kpc->ijkjp', U3, V3, W3)
    # Wait, that's not right. Let me be more careful.
    # U3[i,j,c], V3[jp,k,c], W3[kp,ip,c]
    # T[i,j,kp,jp,k,ip] = sum_c U3[i,j,c]*V3[jp,k,c]*W3[kp,ip,c]
    T = np.einsum('ijc,jkc,kpc->ijkjp', U3, V3, W3)
    # Hmm, the output indices are i,j,k,jp but I need i,j,kp,jp,k,ip
    # Let me redo: np.einsum('ijc,jkc,kpc->ijkjp', ...) gives indices i,j,k,jp
    # But I have 6 indices: i,j,kp,jp,k,ip
    # U3 has indices (i,j,c), V3 has (jp,k,c), W3 has (kp,ip,c)
    # So: np.einsum('ijc,jkc,kpc->ijkjp', U3, V3, W3) 
    # Output: i from U3, j from U3, k from V3, j from U3... no
    # Let me use explicit indices:
    # U3[i,j,c] * V3[jp,k,c] * W3[kp,ip,c] -> T[i,j,kp,jp,k,ip]
    T = np.einsum('ijc,jkc,kpc->ijkjp', U3, V3, W3)
    # This is wrong. Let me just do it properly:
    T = np.einsum('ijc,jkc,kpc->ijkjp', U3, V3, W3)
    
    # Actually, let me just construct it directly for correctness
    # T[i,j,kp,jp,k,ip]
    T = np.zeros((n, m, p, m, p, n))
    for c in range(rank):
        T += U_arr[:, c].reshape(n, m)[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] \
           * V_arr[:, c].reshape(m, p)[np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis] \
           * W_arr[:, c].reshape(p, n)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    
    # Check against identity tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T[i, j, k, j, k, i]
                if abs(actual - expected) > tol:
                    return False
                # Check all other entries should be 0
                # But that's too slow. Let's check a subset
    return True

def _verify_exact(U, V, W, n, m, p, tol=1e-8):
    """Exact verification of tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build full tensor T[i,j,kp,jp,k,ip]
    # T = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
    # We need T[i,j,k,j,k,i] = 1 and T[i,j,kp,jp,k,ip] = 0 otherwise
    
    # Efficient: compute for each c, the outer product
    # For verification, check all 48 diagonal entries and sample off-diagonal
    
    # Check diagonal: T[i,j,k,j,k,i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check off-diagonal: T[i,j,kp,jp,k,ip] = 0 when (i,jp,kp) != (i,j,k) or similar
    # This is expensive, so check a good sample
    # Actually, for correctness we need all entries. Let's be smart.
    # The tensor has n*m*p*m*p*n = 2*4*6*4*6*2 = 2304 entries
    # Only 48 are non-zero (the diagonal)
    # Check all off-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for jp in range(m):
                    for kp in range(p):
                        for ip in range(n):
                            if i==ip and j==jp and k==kp:
                                continue
                            s = 0.0
                            for c in range(rank):
                                s += U_arr[i*m+j, c] * V_arr[jp*p+k, c] * W_arr[kp*n+ip, c]
                            if abs(s) > tol:
                                return False
    return True

def _try_round_to_rational(U, V, W, n, m, p):
    """Try to round entries to integer or half-integer values and verify."""
    for denom in [1, 2, 4, 8, 16]:
        U_r = np.round(np.array(U) * denom) / denom
        V_r = np.round(np.array(V) * denom) / denom
        W_r = np.round(np.array(W) * denom) / denom
        
        if _verify_exact(U_r, V_r, W_r, n, m, p, tol=1e-6):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def _greedy_set_cover(n, m, p, max_rank, deadline, seed=42):
    """
    Greedy set-cover approach: find sparse rank-1 terms that span the tensor.
    
    The tensor T has entries T[i,j,kp,jp,k,ip] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    A rank-1 term (u,v,w) contributes u[i*m+j]*v[jp*p+k]*w[kp*n+ip] to T[i,j,kp,jp,k,ip]
    
    Strategy: generate candidate sparse rank-1 terms, greedily select those that
    best reduce the residual.
    """
    np.random.seed(seed)
    
    # Target tensor entries (flattened)
    # Index: (i,j,kp,jp,k,ip) -> flat index
    def tensor_idx(i, j, kp, jp, k, ip):
        return ((i*m + j)*p + kp)*m*p*n + (jp*p + k)*n + ip
        # Actually let's just use a dict or 6D array
    
    # Build target as dict
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i,j,k,j,k,i)] = 1.0
    
    # Generate candidate sparse rank-1 terms
    # Each term is (support_u, support_v, support_w, signs)
    # For efficiency, limit support sizes
    
    candidates = []
    
    # Type 1: single-entry terms (schoolbook-like)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {i*m+j: 1}
                v = {j*p+k: 1}
                w = {k*n+i: 1}
                candidates.append((u, v, w))
    
    # Type 2: terms covering multiple entries in U
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2:
                    continue
                for k in range(p):
                    u = {i*m+j1: 1, i*m+j2: 1}
                    v = {j1*p+k: 1, j2*p+k: 1}
                    w = {k*n+i: 1}
                    candidates.append((u, v, w))
    
    # Type 3: terms covering multiple entries in V
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(p):
                    if k1 == k2:
                        continue
                    u = {i*m+j: 1}
                    v = {j*p+k1: 1, j*p+k2: 1}
                    w = {k1*n+i: 1, k2*n+i: 1}
                    candidates.append((u, v, w))
    
    # Type 4: terms covering multiple in W
    for j in range(m):
        for k in range(p):
            for i1 in range(n):
                for i2 in range(n):
                    if i1 == i2:
                        continue
                    u = {i1*m+j: 1, i2*m+j: 1}
                    v = {j*p+k: 1}
                    w = {k*n+i1: 1, k*n+i2: 1}
                    candidates.append((u, v, w))
    
    # Type 5: mixed multi-support
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2:
                    continue
                for k1 in range(p):
                    for k2 in range(p):
                        if k1 == k2:
                            continue
                        u = {i*m+j1: 1, i*m+j2: 1}
                        v = {j1*p+k1: 1, j2*p+k2: 1}
                        w = {k1*n+i: 1, k2*n+i: 1}
                        candidates.append((u, v, w))
    
    # Type 6: full 2x2x2 blocks
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2:
                    continue
                for k1 in range(p):
                    for k2 in range(p):
                        if k1 == k2:
                            continue
                        u = {i*m+j1: 1, i*m+j2: 1}
                        v = {j1*p+k1: 1, j1*p+k2: 1, j2*p+k1: 1, j2*p+k2: 1}
                        w = {k1*n+i: 1, k2*n+i: 1}
                        candidates.append((u, v, w))
    
    # Now greedily select
    residual = {key: val for key, val in target.items()}
    selected = []
    
    # Score each candidate by how many residual entries it can help
    def score_candidate(u, v, w):
        """Score = number of target entries where this term has non-zero contribution."""
        score = 0
        for i in range(n):
            for j in range(m):
                if i*m+j not in u:
                    continue
                for jp in range(m):
                    for k in range(p):
                        if jp*p+k not in v:
                            continue
                        for kp in range(p):
                            for ip in range(n):
                                if kp*n+ip not in w:
                                    continue
                                key = (i, j, kp, jp, k, ip)
                                if key in residual and residual[key] != 0:
                                    score += 1
        return score
    
    # Precompute scores
    scored = []
    for idx, (u, v, w) in enumerate(candidates):
        s = score_candidate(u, v, w)
        if s > 0:
            scored.append((s, idx, u, v, w))
    
    scored.sort(key=lambda x: -x[0])
    
    # Greedy selection with coefficient solving
    U_cols = []
    V_cols = []
    W_cols = []
    
    used_candidates = set()
    
    while residual and len(selected) < max_rank and time.time() < deadline - 5:
        # Find best candidate
        best_score = 0
        best_idx = -1
        best_uvw = None
        
        for s, idx, u, v, w in scored:
            if idx in used_candidates:
                continue
            if s > best_score:
                best_score = s
                best_idx = idx
                best_uvw = (u, v, w)
        
        if best_idx == -1 or best_score == 0:
            break
        
        u, v, w = best_uvw
        used_candidates.add(best_idx)
        
        # Determine coefficient: we want to match as many residual entries as possible
        # For each entry in the support of this term, the contribution is coef * u_val * v_val * w_val
        # Since all values are ±1, the contribution is ±coef
        # We want to pick coef to best match the residual
        
        # Collect all (entry, sign) pairs
        entries = []
        for i in range(n):
            for j in range(m):
                if i*m+j not in u:
                    continue
                for jp in range(m):
                    for k in range(p):
                        if jp*p+k not in v:
                            continue
                        for kp in range(p):
                            for ip in range(n):
                                if kp*n+ip not in w:
                                    continue
                                key = (i, j, kp, jp, k, ip)
                                sign = u[i*m+j] * v[jp*p+k] * w[kp*n+ip]
                                entries.append((key, sign))
        
        if not entries:
            continue
        
        # Find best coefficient (from {-1, -0.5, 0.5, 1} or continuous)
        best_coef = 1.0
        best_err = float('inf')
        
        for coef in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            err = 0.0
            for key, sign in entries:
                if key in residual:
                    err += (residual[key] - coef * sign) ** 2
                else:
                    # This entry should be 0, but we're adding coef*sign
                    err += (coef * sign) ** 2
            if err < best_err:
                best_err = err
                best_coef = coef
        
        if best_err >= 1e-10:
            # Try continuous coefficient via least squares
            # Minimize sum (residual[key] - coef * sign)^2 over entries in support
            # + sum (0 - coef * sign)^2 over entries not in target
            a = 0.0
            b = 0.0
            for key, sign in entries:
                a += sign * sign
                if key in residual:
                    b += sign * residual[key]
            if a > 1e-15:
                coef = b / a
                err = 0.0
                for key, sign in entries:
                    if key in residual:
                        err += (residual[key] - coef * sign) ** 2
                    else:
                        err += (coef * sign) ** 2
                if err < best_err:
                    best_err = err
                    best_coef = coef
        
        if best_err > 0.5:
            # This candidate doesn't help much, skip
            continue
        
        # Add this term
        U_cols.append({i*m+j: best_coef * u[i*m+j] for i in range(n) for j in range(m) if i*m+j in u})
        V_cols.append({jp*p+k: best_coef * v[jp*p+k] for jp in range(m) for k in range(p) if jp*p+k in v})
        W_cols.append({kp*n+ip: w[kp*n+ip] for kp in range(p) for ip in range(n) if kp*n+ip in w})
        
        # Wait, the coefficient should be distributed. Let me reconsider.
        # Term c contributes: U_c[i*m+j] * V_c[jp*p+k] * W_c[kp*n+ip]
        # If we set U_c = u, V_c = v, W_c = w * best_coef, then contribution = best_coef * u*v*w
        # Or we can put the coefficient in any one of them.
        
        # Let's put it in W for simplicity
        U_cols[-1] = {i*m+j: u[i*m+j] for i in range(n) for j in range(m) if i*m+j in u}
        V_cols[-1] = {jp*p+k: v[jp*p+k] for jp in range(m) for k in range(p) if jp*p+k in v}
        W_cols[-1] = {kp*n+ip: best_coef * w[kp*n+ip] for kp in range(p) for ip in range(n) if kp*n+ip in w}
        
        selected.append(best_idx)
        
        # Update residual
        for key, sign in entries:
            if key in residual:
                residual[key] -= best_coef * sign
                if abs(residual[key]) < 1e-10:
                    del residual[key]
            else:
                # This created a new non-zero entry
                residual[key] = -best_coef * sign
                if abs(residual[key]) < 1e-10:
                    del residual[key]
        
        if not residual:
            break
    
    if residual:
        return None
    
    # Convert to matrices
    rank = len(U_cols)
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U_mat[idx, c] = val
        for idx, val in V_cols[c].items():
            V_mat[idx, c] = val
        for idx, val in W_cols[c].items():
            W_mat[idx, c] = val
    
    # Verify
    if _verify_exact(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
        # Try to round
        rounded = _try_round_to_rational(U_mat, V_mat, W_mat, n, m, p)
        if rounded is not None:
            return rounded
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def _strassen_based(n, m, p, deadline):
    """
    Use Strassen's algorithm structure.
    2x4x6: Split 4 into 2+2, split 6 into 3+3.
    C = A * B where A is 2x4, B is 4x6.
    A = [A11 A12; A21 A22] where each Ai is 1x2
    B = [B1; B2] where B1 is 2x6, B2 is 2x6
    
    C1 = A11*B1 + A12*B2  (1x6)
    C2 = A21*B1 + A22*B2  (1x6)
    
    Each 1x2 * 2x6 multiplication has rank 2 (schoolbook) or maybe 1?
    No, 1x2 * 2x6: C_k = sum_j A_j * B_jk. This is a bilinear map.
    The tensor for 1x2*2x6 has 1*2*6=12 entries, rank at least max(2,12,6)=6? 
    Actually rank is exactly 2: C_k = A_1*B_1k + A_2*B_2k, so rank 2.
    
    So C1 = A11*B1 + A12*B2: this is two separate 1x2*2x6 multiplications.
    Each has rank 2, so C1 has rank 4, C2 has rank 4, total rank 8.
    But that's just schoolbook for 2x4*4x6 which is 48.
    
    The issue is that A11*B1 and A12*B2 share the same B1 and B2.
    Strassen-style: compute M1 = A11*B1, M2 = A12*B2, then C1 = M1+M2.
    But M1+M2 is not a matrix product, it's an addition.
    
    Actually, for rectangular matrices, Strassen doesn't directly apply in the same way.
    Let me think differently.
    
    2x4 * 4x6:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into j=0,1 and j=2,3:
    C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
    
    Let A_top = A[:, 0:2] (2x2), A_bot = A[:, 2:4] (2x2)
    B_top = B[0:2, :] (2x6), B_bot = B[2:4, :] (2x6)
    
    C = A_top * B_top + A_bot * B_bot
    
    Each 2x2*2x6 multiplication: rank is at most 2*2*6=24, but can be lower.
    2x2*2x6: C[i,k] = sum_{j=0,1} A[i,j]*B[j,k]
    This is two independent 1x2*2x6 multiplications (one for each row of A).
    Each 1x2*2x6 has rank 2. So 2x2*2x6 has rank 4.
    
    So C = A_top*B_top + A_bot*B_bot, each product has rank 4, total rank 8.
    But wait, that's not how tensor rank works. The sum of two rank-r tensors has rank at most 2r,
    but could be less if they share structure.
    
    Actually, the tensor for 2x2*2x6 is:
    T[i,j,k,i',j',k'] = delta(i,i')*delta(j,j')*delta(k,k') for the "multiplication" part
    Wait, I'm confusing myself.
    
    The matrix multiplication tensor for n×m times m×p is a 3-way tensor T[i,j,k] 
    where the bilinear form is sum_{i,j,k} x_i y_j z_k T[i,j,k] = (XY)_k.
    For 2x2*2x6: T has dimensions 2*2, 2*6, 2*6 = 4,12,12.
    T[(i,j),(j',k),(i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
    Wait no. The standard form: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor is T[i,j,k] where the bilinear form is sum_{i,j,k} u_i v_j w_k T[i,j,k]
    and T[i,j,k] = delta(j, j) ... 
    
    I'm overcomplicating this. Let me just use the direct approach.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try greedy set-cover with various max_rank values
    for max_rank in range(30, 48):
        if time.time() > deadline - 10:
            break
        
        for seed in [42, 123, 456]:
            if time.time() > deadline - 5:
                break
            
            result = _greedy_set_cover(n, m, p, max_rank, deadline, seed)
            if result is not None:
                U_list, V_list, W_list = result
                actual_rank = len(U_list[0]) if U_list else 0
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    print(f"Found rank {actual_rank} decomposition via greedy")
                break
    
    return best_result
# EVOLVE-BLOCK-END
