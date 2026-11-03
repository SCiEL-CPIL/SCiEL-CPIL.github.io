"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def strassen_decomp(n, m, p):
        """
        Build rank-84 decomposition using Strassen on 2x2x2 sub-blocks.
        For 3x5x6: partition rows 3=2+1, cols 5=2+2+1, outputs 6=2+2+2
        6 blocks of 2x2x2, each using Strassen rank-7 instead of schoolbook rank-8.
        Remaining 48 products are schoolbook. Total = 6*7 + 48 = 84.
        """
        # Partition:
        # Rows: [0,1] (2 rows), [2] (1 row)
        # Cols: [0,1] (2), [2,3] (2), [4] (1)
        # Outputs: [0,1] (2), [2,3] (2), [4,5] (2)
        
        row_groups = [(0, 1), (2,)]
        col_groups = [(0, 1), (2, 3), (4,)]
        out_groups = [(0, 1), (2, 3), (4, 5)]
        
        U = np.zeros((n * m, 0))
        V = np.zeros((m * p, 0))
        W = np.zeros((p * n, 0))
        
        def add_block(U_list, V_list, W_list, entries):
            """entries: list of (u_idx, v_idx, w_idx, coeff)"""
            for u_idx, v_idx, w_idx, coeff in entries:
                U_list.append((u_idx, coeff))
                V_list.append((v_idx, coeff))
                W_list.append((w_idx, coeff))
        
        U_entries = []
        V_entries = []
        W_entries = []
        
        for (r1, r2) in row_groups:
            for (c1, c2) in col_groups:
                for (k1, k2) in out_groups:
                    # This is a block: rows r1..r2, cols c1..c2, outputs k1..k2
                    # If all sizes are 2, use Strassen; else schoolbook
                    rsz = r2 - r1 + 1
                    csz = c2 - c1 + 1
                    ksz = k2 - k1 + 1
                    
                    if rsz == 2 and csz == 2 and ksz == 2:
                        # Strassen 2x2x2
                        # A = A11 A12; A21 A22  (2x2)
                        # B = B11 B12; B21 B22  (2x2)
                        # C = C11 C12; C21 C22  (2x2)
                        # Strassen:
                        # M1 = (A11+A22)(B11+B22)
                        # M2 = (A21+A22)B11
                        # M3 = A11(B12-B22)
                        # M4 = A22(B21-B11)
                        # M5 = (A11+A12)B22
                        # M6 = (A21-A11)(B11+B12)
                        # M7 = (A12-A22)(B21+B22)
                        # C11 = M1+M4-M5+M7
                        # C12 = M3+M5
                        # C21 = M2+M4
                        # C22 = M1-M2+M3+M6
                        
                        # Map to scalar indices
                        # A[i, j] where i in {r1, r1+1}, j in {c1, c1+1}
                        # B[j, k] where j in {c1, c1+1}, k in {k1, k1+1}
                        # C[i, k] where i in {r1, r1+1}, k in {k1, k1+1}
                        
                        r0, r1_ = r1, r1 + 1
                        c0, c1_ = c1, c1 + 1
                        k0, k1_ = k1, k1 + 1
                        
                        # A11 = A[r0, c0], A12 = A[r0, c1_]
                        # A21 = A[r1_, c0], A22 = A[r1_, c1_]
                        # B11 = B[c0, k0], B12 = B[c0, k1_]
                        # B21 = B[c1_, k0], B22 = B[c1_, k1_]
                        # C11 = C[r0, k0], C12 = C[r0, k1_]
                        # C21 = C[r1_, k0], C22 = C[r1_, k1_]
                        
                        # M1 = (A11+A22)(B11+B22)
                        # = A[r0,c0]*B[c0,k0] + A[r0,c0]*B[c1_,k1_] + A[r1_,c1_]*B[c0,k0] + A[r1_,c1_]*B[c1_,k1_]
                        # But in tensor form, each "product" is a rank-1 term
                        # M1 contributes to C11, C22 with appropriate signs
                        # We need to express M1 as a single rank-1 term in the decomposition
                        # M1 = sum of 4 schoolbook terms, but we want it as ONE factor
                        # In the tensor decomposition, M1 is represented by:
                        # U: entries for A11 and A22 (coeff 1 each)
                        # V: entries for B11 and B22 (coeff 1 each)
                        # W: entries for C11 (coeff 1) and C22 (coeff 1)
                        
                        def strassen_term(Ue, Ve, We, u_coeffs, v_coeffs, w_coeffs):
                            """Add one Strassen product term"""
                            for (u_idx, coeff) in u_coeffs:
                                Ue.append((u_idx, coeff))
                            for (v_idx, coeff) in v_coeffs:
                                Ve.append((v_idx, coeff))
                            for (w_idx, coeff) in w_coeffs:
                                We.append((w_idx, coeff))
                        
                        # M1 = (A11+A22)(B11+B22)
                        # U: A11=1, A22=1
                        # V: B11=1, B22=1
                        # W: C11=1, C22=1
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r0*m+c0, 1), (r1_*m+c1_, 1)],
                                     [(c0*p+k0, 1), (c1_*p+k1_, 1)],
                                     [(k0*n+r0, 1), (k1_*n+r1_, 1)])
                        
                        # M2 = (A21+A22)B11
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r1_*m+c0, 1), (r1_*m+c1_, 1)],
                                     [(c0*p+k0, 1)],
                                     [(k0*n+r1_, 1)])
                        
                        # M3 = A11(B12-B22)
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r0*m+c0, 1)],
                                     [(c0*p+k1_, 1), (c1_*p+k1_, -1)],
                                     [(k1_*n+r0, 1)])
                        
                        # M4 = A22(B21-B11)
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r1_*m+c1_, 1)],
                                     [(c1_*p+k0, 1), (c0*p+k0, -1)],
                                     [(k0*n+r1_, 1)])
                        
                        # M5 = (A11+A12)B22
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r0*m+c0, 1), (r0*m+c1_, 1)],
                                     [(c1_*p+k1_, 1)],
                                     [(k1_*n+r0, -1)])
                        
                        # M6 = (A21-A11)(B11+B12)
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r1_*m+c0, 1), (r0*m+c0, -1)],
                                     [(c0*p+k0, 1), (c0*p+k1_, 1)],
                                     [(k1_*n+r1_, 1)])
                        
                        # M7 = (A12-A22)(B21+B22)
                        strassen_term(U_entries, V_entries, W_entries,
                                     [(r0*m+c1_, 1), (r1_*m+c1_, -1)],
                                     [(c1_*p+k0, 1), (c1_*p+k1_, 1)],
                                     [(k0*n+r0, 1), (k1_*n+r1_, 1)])
                        
                    else:
                        # Schoolbook for this block
                        for i in range(r1, r2+1):
                            for j in range(c1, c2+1):
                                for k in range(k1, k2+1):
                                    U_entries.append((i*m+j, 1))
                                    V_entries.append((j*p+k, 1))
                                    W_entries.append((k*n+i, 1))
        
        rank = len(U_entries)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for col, (u_idx, coeff) in enumerate(U_entries):
            U[u_idx, col] += coeff
        for col, (v_idx, coeff) in enumerate(V_entries):
            V[v_idx, col] += coeff
        for col, (w_idx, coeff) in enumerate(W_entries):
            W[w_idx, col] += coeff
        
        return {'U': U, 'V': V, 'W': W, 'rank': rank}

    def verify(U, V, W, n, m, p, tol=1e-6):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, n, m, p, tol=1e-6):
        """Vectorized verification"""
        r = U.shape[1]
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Use broadcasting
        U_t = U.reshape(n, m, r)  # n x m x r
        V_t = V.reshape(m, p, r)  # m x p x r
        W_t = W.reshape(p, n, r)  # p x n x r
        
        # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # This is a 3-tensor contraction
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
        return np.all(np.abs(T - 1.0) <= tol)

    def als_step(U, V, W, n, m, p, r):
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                w_idx = np.arange(p) * n + i
                w_slice = W[w_idx, :]
                v_slice = V[j*p : (j+1)*p, :]
                A = v_slice * w_slice
                b = np.ones(p)
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_U[i*m+j, :] = u_sol
                except np.linalg.LinAlgError:
                    new_U[i*m+j, :] = U[i*m+j, :]
        
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                u_idx = np.arange(n) * m + j
                u_slice = U[u_idx, :]
                w_idx = k * n + np.arange(n)
                w_slice = W[w_idx, :]
                A = u_slice * w_slice
                b = np.ones(n)
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_V[j*p+k, :] = v_sol
                except np.linalg.LinAlgError:
                    new_V[j*p+k, :] = V[j*p+k, :]
        
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                u_idx = i * m + np.arange(m)
                u_slice = U[u_idx, :]
                v_idx = np.arange(m) * p + k
                v_slice = V[v_idx, :]
                A = u_slice * v_slice
                b = np.ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_W[k*n+i, :] = w_sol
                except np.linalg.LinAlgError:
                    new_W[k*n+i, :] = W[k*n+i, :]
        
        return new_U, new_V, new_W

    def try_rank_als(r, U0=None, V0=None, W0=None, max_iters=30):
        if time.time() > deadline:
            return None
        
        if U0 is not None:
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
        else:
            # Sparse random init: mostly zeros with some nonzeros
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            # Add random nonzeros
            for c in range(r):
                # Each column has a few nonzeros
                u_nz = np.random.choice(n*m, size=min(3, n*m), replace=False)
                v_nz = np.random.choice(m*p, size=min(3, m*p), replace=False)
                w_nz = np.random.choice(p*n, size=min(3, p*n), replace=False)
                U[u_nz, c] = np.random.choice([-1, 1], size=len(u_nz))
                V[v_nz, c] = np.random.choice([-1, 1], size=len(v_nz))
                W[w_nz, c] = np.random.choice([-1, 1], size=len(w_nz))
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U, V, W = als_step(U, V, W, n, m, p, r)
            
            if it % 5 == 0 or it == max_iters - 1:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                    return {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
        
        # Final check with different rounding
        for scale in [2, 4]:
            U_rounded = np.round(U * scale) / scale
            V_rounded = np.round(V * scale) / scale
            W_rounded = np.round(W * scale) / scale
            if verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                return {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
        
        return None

    def try_strassen_based(r, max_iters=20):
        """Start from Strassen-84 and try to reduce rank to r"""
        if time.time() > deadline:
            return None
        if r >= 84:
            return None  # We already have 84
        
        # Get Strassen decomposition
        strassen = strassen_decomp(n, m, p)
        U84 = strassen['U']
        V84 = strassen['V']
        W84 = strassen['W']
        
        # Try to project down to rank r by zeroing random columns
        for attempt in range(3):
            if time.time() > deadline:
                return None
            
            # Randomly select r columns out of 84
            cols = np.random.choice(84, size=r, replace=False)
            U = U84[:, cols].copy()
            V = V84[:, cols].copy()
            W = W84[:, cols].copy()
            
            # ALS to fix up
            res = try_rank_als(r, U, V, W, max_iters=max_iters)
            if res is not None:
                return res
        
        return None

    # Build baseline
    strassen = strassen_decomp(n, m, p)
    best_result = {'U': strassen['U'].tolist(), 'V': strassen['V'].tolist(), 'W': strassen['W'].tolist()}
    best_rank = strassen['rank']
    
    # Verify Strassen
    if not verify_fast(strassen['U'], strassen['V'], strassen['W'], n, m, p):
        # Fallback to schoolbook
        sb = schoolbook()
        best_result = sb
        best_rank = n * m * p
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound+1 up to best_rank-1
    # Focus on ranks near 84 first (most likely to succeed)
    ranks_to_try = list(range(max(lower_bound+1, best_rank - 10), best_rank))
    ranks_to_try += list(range(lower_bound+1, max(lower_bound+1, best_rank - 10)))
    # Deduplicate
    seen = set()
    ranks_to_try = [r for r in ranks_to_try if r not in seen and not seen.add(r)]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = False
        # Try Strassen-based reduction
        res = try_strassen_based(r, max_iters=15)
        if res is not None:
            best_result = {'U': res['U'].tolist(), 'V': res['V'].tolist(), 'W': res['W'].tolist()}
            best_rank = r
            found = True
            break
        
        # Try random ALS
        for attempt in range(2):
            if time.time() > deadline:
                break
            res = try_rank_als(r, max_iters=20)
            if res is not None:
                best_result = {'U': res['U'].tolist(), 'V': res['V'].tolist(), 'W': res['W'].tolist()}
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

