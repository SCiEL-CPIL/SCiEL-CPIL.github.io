import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strategy: Use non-overlapping 2x2x2 Strassen blocks to reduce rank.
    # Each 2x2x2 block: 8 schoolbook terms -> 7 Strassen terms (save 1)
    # We can place up to 4 such blocks in the 3x5x5 tensor
    # Total: 4 blocks * 7 + (75 - 4*8) = 28 + 43 = 71
    
    total_rank = 4 * 7 + (n * m * p - 4 * 8)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Define the 4 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 2, 3),
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # M1 = (a00 + a11)(b00 + b11)
        c = col
        U[i0*m + j0, c] += 1
        U[i1*m + j1, c] += 1
        V[j0*p + k0, c] += 1
        V[j1*p + k1, c] += 1
        W[k0*n + i0, c] += 1
        W[k1*n + i1, c] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        c = col
        U[i1*m + j0, c] += 1
        U[i1*m + j1, c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + i1, c] += 1
        W[k1*n + i1, c] += -1
        col += 1
        
        # M3 = a00(b01 - b11)
        c = col
        U[i0*m + j0, c] += 1
        V[j0*p + k1, c] += 1
        V[j1*p + k1, c] += -1
        W[k1*n + i0, c] += 1
        W[k1*n + i1, c] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        c = col
        U[i1*m + j1, c] += 1
        V[j1*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + i1, c] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        c = col
        U[i0*m + j0, c] += 1
        U[i0*m + j1, c] += 1
        V[j1*p + k1, c] += 1
        W[k0*n + i0, c] += -1
        W[k1*n + i0, c] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        c = col
        U[i1*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + k1, c] += 1
        W[k1*n + i1, c] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        c = col
        U[i0*m + j1, c] += 1
        U[i1*m + j1, c] += -1
        V[j1*p + k0, c] += 1
        V[j1*p + k1, c] += 1
        W[k0*n + i0, c] += 1
        col += 1
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (i0, i1, j0, j1, k0, k1) in blocks:
                    if i0 <= i <= i1 and j0 <= j <= j1 and k0 <= k <= k1:
                        in_block = True
                        break
                if not in_block:
                    U[i*m + j, col] += 1
                    V[j*p + k, col] += 1
                    W[k*n + i, col] += 1
                    col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Rank-merging heuristic: try to combine pairs of rank-1 terms
    def can_merge_and_extract(c1, c2, U_mat, V_mat, W_mat, tol=1e-10):
        """Check if terms c1 and c2 can be merged into a single rank-1 term.
        Returns (True, u_merged, v_merged, w_merged) or (False, None, None, None)."""
        u1 = U_mat[:, c1]
        u2 = U_mat[:, c2]
        v1 = V_mat[:, c1]
        v2 = V_mat[:, c2]
        w1 = W_mat[:, c1]
        w2 = W_mat[:, c2]
        
        # Find indices where both terms have non-zero entries
        nz_u1 = np.nonzero(u1)[0]
        nz_u2 = np.nonzero(u2)[0]
        nz_v1 = np.nonzero(v1)[0]
        nz_v2 = np.nonzero(v2)[0]
        nz_w1 = np.nonzero(w1)[0]
        nz_w2 = np.nonzero(w2)[0]
        
        if len(nz_u1) == 0 or len(nz_v1) == 0 or len(nz_w1) == 0:
            return True, None, None, None
        if len(nz_u2) == 0 or len(nz_v2) == 0 or len(nz_w2) == 0:
            return True, None, None, None
        
        # Test rank-1 condition on the sum tensor
        i_candidates = np.union1d(nz_u1, nz_u2)
        j_candidates = np.union1d(nz_v1, nz_v2)
        k_candidates = np.union1d(nz_w1, nz_w2)
        
        if len(i_candidates) < 2 or len(j_candidates) < 2 or len(k_candidates) < 2:
            return True, None, None, None
        
        # Test rank-1 condition on multiple 2x2x2 sub-tensors
        np.random.seed(42)
        for _ in range(20):
            i1, i2 = np.random.choice(i_candidates, 2, replace=False)
            j1, j2 = np.random.choice(j_candidates, 2, replace=False)
            k1, k2 = np.random.choice(k_candidates, 2, replace=False)
            
            T_i1j1k1 = u1[i1]*v1[j1]*w1[k1] + u2[i1]*v2[j1]*w2[k1]
            T_i1j1k2 = u1[i1]*v1[j1]*w1[k2] + u2[i1]*v2[j1]*w2[k2]
            T_i1j2k1 = u1[i1]*v1[j2]*w1[k1] + u2[i1]*v2[j2]*w2[k1]
            T_i1j2k2 = u1[i1]*v1[j2]*w1[k2] + u2[i1]*v2[j2]*w2[k2]
            T_i2j1k1 = u1[i2]*v1[j1]*w1[k1] + u2[i2]*v2[j1]*w2[k1]
            T_i2j1k2 = u1[i2]*v1[j1]*w1[k2] + u2[i2]*v2[j1]*w2[k2]
            T_i2j2k1 = u1[i2]*v1[j2]*w1[k1] + u2[i2]*v2[j2]*w2[k1]
            T_i2j2k2 = u1[i2]*v1[j2]*w1[k2] + u2[i2]*v2[j2]*w2[k2]
            
            det1 = T_i1j1k1 * T_i2j2k2 - T_i1j2k1 * T_i2j1k2
            det2 = T_i1j1k2 * T_i2j2k1 - T_i1j2k2 * T_i2j1k1
            det3 = T_i1j1k1 * T_i2j1k2 - T_i1j1k2 * T_i2j1k1
            
            if abs(det1) > tol * max(1.0, abs(T_i1j1k1)*abs(T_i2j2k2)):
                return False, None, None, None
            if abs(det2) > tol * max(1.0, abs(T_i1j1k2)*abs(T_i2j2k1)):
                return False, None, None, None
            if abs(det3) > tol * max(1.0, abs(T_i1j1k1)*abs(T_i2j1k2)):
                return False, None, None, None
        
        # If all tests pass, try to find the merged term
        # The merged term (u,v,w) satisfies: u[i]*v[j]*w[k] = T[i,j,k]
        # We can recover u, v, w up to scaling from T
        # Pick a reference point where T is non-zero
        T_ref = None
        ref_i, ref_j, ref_k = None, None, None
        for ii in i_candidates:
            for jj in j_candidates:
                for kk in k_candidates:
                    val = u1[ii]*v1[jj]*w1[kk] + u2[ii]*v2[jj]*w2[kk]
                    if abs(val) > tol:
                        T_ref = val
                        ref_i, ref_j, ref_k = ii, jj, kk
                        break
                if T_ref is not None:
                    break
            if T_ref is not None:
                break
        
        if T_ref is None:
            return False, None, None, None
        
        # Recover u, v, w from T
        # u[i] * v[j] * w[k] = T[i,j,k]
        # Fix v[ref_j] = 1, w[ref_k] = 1, then u[ref_i] = T_ref
        # Then for other i: u[i] = T[i, ref_j, ref_k] / (v[ref_j]*w[ref_k]) = T[i, ref_j, ref_k]
        # For other j: v[j] = T[ref_i, j, ref_k] / (u[ref_i]*w[ref_k]) = T[ref_i, j, ref_k] / T_ref
        # For other k: w[k] = T[ref_i, ref_j, k] / (u[ref_i]*v[ref_j]) = T[ref_i, ref_j, k] / T_ref
        
        u_merged = np.zeros_like(U_mat[:, 0])
        v_merged = np.zeros_like(V_mat[:, 0])
        w_merged = np.zeros_like(W_mat[:, 0])
        
        u_merged[ref_i] = T_ref
        v_merged[ref_j] = 1
        w_merged[ref_k] = 1
        
        for ii in i_candidates:
            if ii != ref_i:
                u_merged[ii] = u1[ii]*v1[ref_j]*w1[ref_k] + u2[ii]*v2[ref_j]*w2[ref_k]
        
        for jj in j_candidates:
            if jj != ref_j:
                v_merged[jj] = u1[ref_i]*v1[jj]*w1[ref_k] + u2[ref_i]*v2[jj]*w2[ref_k]
        
        for kk in k_candidates:
            if kk != ref_k:
                w_merged[kk] = u1[ref_i]*v1[ref_j]*w1[kk] + u2[ref_i]*v2[ref_j]*w2[kk]
        
        # Verify the merged term reproduces the sum
        for ii in i_candidates:
            for jj in j_candidates:
                for kk in k_candidates:
                    expected = u1[ii]*v1[jj]*w1[kk] + u2[ii]*v2[jj]*w2[kk]
                    actual = u_merged[ii]*v_merged[jj]*w_merged[kk]
                    if abs(expected - actual) > tol * max(1.0, abs(expected)):
                        return False, None, None, None
        
        return True, u_merged, v_merged, w_merged
    
    # Try to merge pairs of terms
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = total_rank
    merged = set()
    
    # Try all pairs, prioritizing pairs that might merge
    for c1 in range(current_rank):
        if c1 in merged:
            continue
        if time.time() > deadline:
            break
        for c2 in range(c1 + 1, current_rank):
            if c2 in merged:
                continue
            if time.time() > deadline:
                break
            
            can_merge, u_m, v_m, w_m = can_merge_and_extract(c1, c2, current_U, current_V, current_W)
            if can_merge and u_m is not None:
                # Replace c1 with the merged term, mark c2 as merged
                current_U[:, c1] = u_m
                current_V[:, c1] = v_m
                current_W[:, c1] = w_m
                merged.add(c2)
                current_rank -= 1
                break  # Restart from c1 with new merged set
    
    # Build final result
    final_U = np.zeros((n * m, current_rank))
    final_V = np.zeros((m * p, current_rank))
    final_W = np.zeros((p * n, current_rank))
    
    col = 0
    for c in range(total_rank):
        if c in merged:
            continue
        final_U[:, col] = current_U[:, c]
        final_V[:, col] = current_V[:, c]
        final_W[:, col] = current_W[:, c]
        col += 1
    
    assert col == current_rank, f"Expected {current_rank} columns, got {col}"
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U_mat.shape[1]):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    if not verify(final_U, final_V, final_W, n, m, p):
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
