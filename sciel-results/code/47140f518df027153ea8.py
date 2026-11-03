import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # 4 Strassen blocks + schoolbook
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_rank = 4 * 7 + (n * m * p - 4 * 8)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        c = col
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        c = col
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        c = col
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        c = col
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        c = col
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        col += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
    
    # Schoolbook for remaining
    covered = np.zeros((n, m, p), dtype=bool)
    for i0, j0, k0 in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] += 1
                    V[j*p + k, col] += 1
                    W[k*n + i, col] += 1
                    col += 1
    
    assert col == total_rank
    
    # Fast rank-1 check using SVD on matrix flattening
    def is_rank1_fast(tensor, tol=1e-10):
        """Check if a 3D tensor (n,m,p) is rank 1 using SVD."""
        # Reshape to (n, m*p)
        mat = tensor.reshape(n, m * p)
        if np.all(np.abs(mat) < tol):
            return True, (np.zeros(n), np.zeros(m), np.zeros(p))
        
        # SVD
        try:
            U_svd, s, Vt = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return False, None
        
        if len(s) < 2:
            # Rank 1 by SVD
            sigma = s[0] if len(s) > 0 else 0
            if sigma < tol:
                return True, (np.zeros(n), np.zeros(m), np.zeros(p))
            u = U_svd[:, 0]
            vt = Vt[0, :]
            v = vt.reshape(m, p)
            # Verify
            recon = np.einsum('i,jk->ijk', u, v)
            if np.allclose(tensor, recon, atol=tol, rtol=tol):
                return True, (u, v[:, 0], v[0, :])
            return False, None
        
        # Check if second singular value is negligible
        if s[1] / s[0] < tol:
            u = U_svd[:, 0]
            vt = Vt[0, :]
            v = vt.reshape(m, p)
            recon = np.einsum('i,jk->ijk', u, v)
            if np.allclose(tensor, recon, atol=tol, rtol=tol):
                # Extract b and c from v
                # v = b * c^T, so find b and c
                # Use the fact that v[j,k] = b[j]*c[k]
                # Find non-zero entries to determine b and c
                nz_j = np.nonzero(np.abs(v) > tol)[0]
                if len(nz_j) == 0:
                    return False, None
                j0 = nz_j[0]
                c_vec = v[j0, :]
                # Find if c_vec is proportional to all rows
                b = np.zeros(m)
                b[j0] = 1
                for j in range(m):
                    if np.abs(v[j, :]).max() > tol:
                        ratio = v[j, :] / c_vec
                        # Check if ratio is constant
                        nz_c = np.nonzero(np.abs(c_vec) > tol)[0]
                        if len(nz_c) == 0:
                            return False, None
                        ratios = ratio[nz_c]
                        if np.allclose(ratios, ratios[0], atol=tol, rtol=tol):
                            b[j] = ratios[0]
                        else:
                            return False, None
                return True, (u, b, c_vec)
        
        return False, None
    
    def try_merge_terms(U_mat, V_mat, W_mat, cols):
        """Try to merge multiple terms. Returns (success, new_u, new_v, new_w)."""
        if len(cols) == 0:
            return False, None, None, None
        
        # Form sum tensor
        T = np.zeros((n, m, p))
        for c in cols:
            u_c = U_mat[:, c].reshape(n, m)
            v_c = V_mat[:, c].reshape(m, p)
            w_c = W_mat[:, c].reshape(p, n)
            # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            # But w_c is indexed as W[k*n+i], so w_c[k,i]
            for i in range(n):
                for j in range(m):
                    if u_c[i, j] != 0:
                        for k in range(p):
                            if v_c[j, k] != 0 and w_c[k, i] != 0:
                                T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        is_r1, result = is_rank1_fast(T)
        if is_r1:
            a, b, c_vec = result
            # Reconstruct U, V, W vectors
            new_u = np.zeros(n * m)
            new_v = np.zeros(m * p)
            new_w = np.zeros(p * n)
            
            for i in range(n):
                for j in range(m):
                    new_u[i * m + j] = a[i] * b[j]
            for j in range(m):
                for k in range(p):
                    # Wait, we need to be careful. The tensor is T[i,j,k] = a[i]*b[j]*c[k]
                    # U[i*m+j] * V[j*p+k] * W[k*n+i] = a[i]*b[j]*c[k]
                    # We can set U[i*m+j] = a[i], V[j*p+k] = b[j], W[k*n+i] = c[k]
                    # But this only works if the factorization is consistent
                    # Actually, for a rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k],
                    # we have U[i*m+j] * V[j*p+k] * W[k*n+i] = a[i]*b[j]*c[k]
                    # One valid factorization: U[i*m+j] = a[i], V[j*p+k] = b[j], W[k*n+i] = c[k]
                    # Let's verify: a[i] * b[j] * c[k] = T[i,j,k] ✓
                    pass
            
            # Simpler: just set them directly
            for i in range(n):
                for j in range(m):
                    new_u[i * m + j] = a[i]
            for j in range(m):
                for k in range(p):
                    new_v[j * p + k] = b[j]
            for k in range(p):
                for i in range(n):
                    new_w[k * n + i] = c_vec[k]
            
            return True, new_u, new_v, new_w
        return False, None, None, None
    
    # Attempt rank reduction via merging
    merged_cols = set()
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    active_cols = list(range(total_rank))
    
    # First pass: try all pairs
    max_passes = 15
    for _ in range(max_passes):
        if time.time() > deadline - 20:
            break
        found = False
        new_active = []
        i = 0
        while i < len(active_cols):
            c1 = active_cols[i]
            if c1 in merged_cols:
                i += 1
                continue
            merged_with = []
            for j in range(i+1, len(active_cols)):
                c2 = active_cols[j]
                if c2 in merged_cols:
                    continue
                success, new_u, new_v, new_w = try_merge_terms(current_U, current_V, current_W, [c1, c2])
                if success:
                    current_U[:, c1] = new_u
                    current_V[:, c1] = new_v
                    current_W[:, c1] = new_w
                    merged_cols.add(c2)
                    merged_with.append(c2)
                    found = True
                    break
            if not merged_with:
                new_active.append(c1)
            i += 1
        active_cols = new_active
        if not found:
            break
    
    # Second pass: try triples among remaining active columns
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            found = False
            new_active = []
            i = 0
            while i < len(active_cols):
                c1 = active_cols[i]
                if c1 in merged_cols:
                    i += 1
                    continue
                merged_with = []
                # Try pairs first (already done), now try triples
                for j in range(i+1, len(active_cols)):
                    c2 = active_cols[j]
                    if c2 in merged_cols:
                        continue
                    for k in range(j+1, len(active_cols)):
                        c3 = active_cols[k]
                        if c3 in merged_cols:
                            continue
                        success, new_u, new_v, new_w = try_merge_terms(current_U, current_V, current_W, [c1, c2, c3])
                        if success:
                            current_U[:, c1] = new_u
                            current_V[:, c1] = new_v
                            current_W[:, c1] = new_w
                            merged_cols.add(c2)
                            merged_cols.add(c3)
                            merged_with.extend([c2, c3])
                            found = True
                            break
                    if merged_with:
                        break
                if not merged_with:
                    new_active.append(c1)
                i += 1
            active_cols = new_active
            if not found:
                break
    
    if merged_cols:
        keep = [c for c in range(total_rank) if c not in merged_cols]
        U_final = current_U[:, keep]
        V_final = current_V[:, keep]
        W_final = current_W[:, keep]
    else:
        U_final = U
        V_final = V
        W_final = W
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
