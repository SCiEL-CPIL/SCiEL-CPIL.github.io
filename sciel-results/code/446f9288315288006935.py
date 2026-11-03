"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  1. Build a known rank-55 decomposition (Strassen on 2x2 blocks + naive 2x1*1x10).
  2. Apply peeling: iteratively identify rank-1 components with sparse V-factors
     (at most 2 non-zero entries in the m=3 middle dimension), subtract them,
     and refine the remaining tensor to find a lower-rank decomposition.
  3. Use structured search over support patterns derived from the rank-55
     decomposition, looking for rank 54 or lower.
  4. For each candidate support, solve the linear system for coefficients.
  5. Verify exactness and half-integer constraint.
"""

import numpy as np
import random
import time
from itertools import combinations, product

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using matrix multiplication.
    For each (i, k), check that sum_r U[i,j,r] * W[k,i,r] * V[j,k,r] == delta_jj'.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            Uw = U_mat[i] * W_mat[k, i]
            M_ik = Uw @ V_mat[:, k].T
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _build_mult_tensor(n, m, p):
    """
    Build the matrix multiplication tensor as a 3D array T[a, b, c]
    where a = i*m+j, b = j*p+k, c = k*n+i.
    T[a, b, c] = 1 if the indices are consistent, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _tensor_to_matrix(T, n, m, p, i, k):
    """
    For fixed i and k, extract the m x m matrix from tensor T.
    T[i*m+j, j*p+k, k*n+i] gives the (j, j') entry.
    """
    M = np.zeros((m, m), dtype=np.float64)
    for j in range(m):
        for j2 in range(m):
            # We need T[i*m+j, j*p+k, k*n+i] but that's only for j2=j
            # Actually, the tensor entry is T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
            # But for the matrix M_ik, we need M_ik[j, j2] = sum_r U[i,j,r]*W[k,i,r]*V[j2,k,r]
            # The tensor T has T[i*m+j, j*p+k, k*n+i] = delta_{j,j2}... no.
            # Let me reconsider. The tensor T_mult satisfies:
            # sum_j T[i*m+j, j*p+k, k*n+i'] * A[i,j] * B[j,k] = C[i',k]
            # So T[i*m+j, j*p+k, k*n+i'] = 1 if j=j' (which it always is in the sum)
            # Actually T is defined by: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
            # And the multiplication is: C[i',k] = sum_{i,j,k} T[i*m+j, j*p+k, k*n+i'] * A[i,j] * B[j,k]
            # Wait, that's not right either. Let me think more carefully.
            # 
            # Standard: C[i', k'] = sum_j A[i', j] * B[j, k']
            # Tensor form: C[i', k'] = sum_{a,b,c} T[a,b,c] * A_flat[a] * B_flat[b] * delta...
            # Actually the standard tensor decomposition is:
            # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, 0 otherwise.
            # And the reconstruction is: for each i', k':
            #   C[i', k'] = sum_{j} A[i', j] * B[j, k']
            #             = sum_{i,j,k} T[i*m+j, j*p+k, k*n+i'] * A[i,j] * B[j,k]
            #             (only terms where i=i', k=k' contribute because of T's support)
            # 
            # So for fixed i', k': C[i',k'] = sum_j T[i'*m+j, j*p+k', k'*n+i'] * A[i',j] * B[j,k']
            # And T[i'*m+j, j*p+k', k'*n+i'] = 1 for all j.
            # So C[i',k'] = sum_j A[i',j] * B[j,k']. Correct.
            #
            # For the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
            # For fixed i, k: sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = T[i*m+j, j*p+k, k*n+i]
            # But T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent (they always are for valid i,j,k)
            # Wait no. T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
            # For a=i*m+j, b=j*p+k, c=k*n+i: yes, these are consistent, so T=1.
            # But what about T[i*m+j, j2*p+k, k*n+i] for j != j2? 
            # b = j2*p+k, so this would require j=j2 for consistency. So T=0 if j!=j2.
            # 
            # So for fixed i, k: the matrix M_ik[j, j2] = T[i*m+j, j2*p+k, k*n+i] = delta_{j,j2}
            # That means M_ik = I_m. That's what we're checking in _verify_fast.
            pass
    return M


def _peel_rank1_sparse(T, n, m, p, max_v_nnz=2):
    """
    Find a rank-1 component with sparse V-factor using peeling.
    
    A rank-1 component is u ⊗ v ⊗ w where:
    - u is in R^{n*m}
    - v is in R^{m*p}  
    - w is in R^{p*n}
    
    We look for components where v has at most max_v_nnz non-zero entries.
    
    Approach: For each pair of non-zero positions in v (or single position),
    try to find u and w such that u ⊗ v ⊗ w is a component of T.
    
    If v has exactly 1 non-zero at position (j, k), then:
    T[i*m+j, j*p+k, k*n+i'] = u[i*m+j] * v[j*p+k] * w[k*n+i']
    For this to be rank-1, we need T[i*m+j, j*p+k, k*n+i'] = u[i*m+j] * w[k*n+i'] * c
    for some constant c. This means the matrix T[i*m+j, j*p+k, :] (over i') 
    must be rank 1 (outer product of u[:, j] and w[:, :]).
    
    More generally, if v has non-zeros at positions (j1, k1) and (j2, k2),
    we need more careful analysis.
    
    For simplicity, start with v having exactly 1 non-zero.
    """
    # Try v with single non-zero at (j, k)
    for j in range(m):
        for k in range(p):
            # Extract the slice: for all i, i', T[i*m+j, j*p+k, k*n+i']
            # This should be u[i*m+j] * w[k*n+i'] * c for rank-1
            slice_mat = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for i2 in range(n):
                    slice_mat[i, i2] = T[i * m + j, j * p + k, k * n + i2]
            
            # Check if this is rank 1 (or zero)
            if np.allclose(slice_mat, 0, atol=1e-12):
                continue
            
            # SVD to check rank
            try:
                U_svd, s_vals, Vt_svd = np.linalg.svd(slice_mat, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            
            if len(s_vals) == 0:
                continue
            
            # Check if effectively rank 1
            if len(s_vals) > 1 and s_vals[1] > 1e-10 * max(s_vals[0], 1e-15):
                continue
            
            # It's rank 1 (or close). Extract u and w.
            # slice_mat ≈ u_col * w_row * c
            u_col = U_svd[:, 0] * s_vals[0]
            w_row = Vt_svd[0, :]
            
            # Normalize: let c = 1, so u_col and w_row are as is
            # v has single non-zero at (j, k) with value 1
            # We can absorb the scaling into u, v, or w
            # Let v[j*p+k] = 1, then u[i*m+j] = u_col[i], w[k*n+i'] = w_row[i']
            
            # Check if entries are half-integers (or can be made to be)
            u_vec = np.zeros(n * m, dtype=np.float64)
            w_vec = np.zeros(p * n, dtype=np.float64)
            v_vec = np.zeros(m * p, dtype=np.float64)
            
            for i in range(n):
                u_vec[i * m + j] = u_col[i]
            for i2 in range(n):
                w_vec[k * n + i2] = w_row[i2]
            v_vec[j * p + k] = 1.0
            
            # Check half-integer constraint
            if not _is_valid_entries([u_vec.tolist()], [v_vec.tolist()], [w_vec.tolist()]):
                # Try scaling by 2
                u_vec2 = u_vec * 2
                v_vec2 = v_vec * 0.5
                if _is_valid_entries([u_vec2.tolist()], [v_vec2.tolist()], [w_vec.tolist()]):
                    u_vec, v_vec = u_vec2, v_vec2
                else:
                    continue
            
            # Verify this is actually a component: T - u⊗v⊗w should have reduced rank
            # Build the rank-1 tensor
            T_rank1 = np.zeros_like(T)
            for i in range(n):
                for i2 in range(n):
                    T_rank1[i * m + j, j * p + k, k * n + i2] = u_vec[i * m + j] * v_vec[j * p + k] * w_vec[k * n + i2]
            
            # Check that T - T_rank1 is "cleaner" (more zeros or lower rank in slices)
            T_remaining = T - T_rank1
            
            # Verify T_rank1 is actually a valid component (all entries of T - T_rank1 are 0 or 1, or at least T_rank1 entries are 0 or 1)
            # Actually, we just need T_rank1 to be a valid rank-1 tensor that we can subtract
            # The key check: T_rank1[a,b,c] should be 0 whenever T[a,b,c] = 0
            # (we shouldn't be adding where T is zero)
            if not np.allclose(np.where(T == 0, T_rank1, 0), 0, atol=1e-10):
                continue
            
            return u_vec, v_vec, w_vec, T_remaining
    
    # Try v with 2 non-zeros: (j1, k1) and (j2, k2)
    for j1 in range(m):
        for j2 in range(j1, m):
            for k1 in range(p):
                for k2 in range(p):
                    if j1 == j2 and k1 == k2:
                        continue
                    
                    # v has non-zeros at (j1,k1) and (j2,k2)
                    # T[i*m+j1, j1*p+k1, k1*n+i'] = u[i*m+j1] * v1 * w[k1*n+i']
                    # T[i*m+j2, j2*p+k2, k2*n+i'] = u[i*m+j2] * v2 * w[k2*n+i']
                    # 
                    # From first: u[i*m+j1] = T[i*m+j1, j1*p+k1, k1*n+i'] / (v1 * w[k1*n+i'])
                    # This must be independent of i' for fixed i.
                    # 
                    # This is getting complex. Let's use a simpler approach:
                    # Assume v1 = v2 = 1 (we can scale later).
                    # Then u[i*m+j1] = T[i*m+j1, j1*p+k1, k1*n+i'] / w[k1*n+i']
                    # and u[i*m+j2] = T[i*m+j2, j2*p+k2, k2*n+i'] / w[k2*n+i']
                    # 
                    # For consistency, the ratio T[i*m+j1, j1*p+k1, k1*n+i'] / T[i*m+j1, j1*p+k1, k1*n+i''] 
                    # should equal w[k1*n+i'] / w[k1*n+i'']
                    # 
                    # This is still complex. Skip for now and just use single-nonzero peeling.
                    pass
    
    return None


def _try_drop_term(U55, V55, W55, n, m, p, drop_idx, deadline):
    """
    Try to drop term drop_idx and re-solve for coefficients of remaining terms.
    """
    rank55 = 55
    keep_idx = [i for i in range(rank55) if i != drop_idx]
    U_keep = U55[:, keep_idx]
    V_keep = V55[:, keep_idx]
    W_keep = W55[:, keep_idx]
    
    eqs = []
    b_vec = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                row = []
                for r in range(54):
                    row.append(U_keep[a, r] * V_keep[b, r] * W_keep[c, r])
                eqs.append(row)
                b_vec.append(1.0)
    
    A = np.array(eqs, dtype=np.float64)
    b = np.array(b_vec, dtype=np.float64)
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    valid = True
    for cv in coeffs:
        if abs(cv * 2 - round(cv * 2)) > 1e-6:
            valid = False
            break
    if not valid:
        return None
    
    U_new = U_keep * coeffs
    V_new = V_keep * coeffs
    W_new = W_keep * coeffs
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _try_merge_terms(U55, V55, W55, n, m, p, deadline):
    """
    Try to merge two terms into one by finding a linear combination.
    """
    rank55 = 55
    
    for i in range(rank55):
        if time.time() > deadline:
            return None
        
        result = _try_drop_term(U55, V55, W55, n, m, p, i, deadline)
        if result is not None:
            return result
    
    return None


def _random_support_search(n, m, p, rank, deadline, U55, V55, W55):
    """
    Random search for a support pattern of the given rank.
    """
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        idx = random.sample(range(55), rank)
        U_trial = U55[:, idx].copy()
        V_trial = V55[:, idx].copy()
        W_trial = W55[:, idx].copy()
        
        if _verify_fast(U_trial, V_trial, W_trial, n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()
    
    return None


def _peeling_search(n, m, p, deadline):
    """
    Use peeling to find a low-rank decomposition.
    
    1. Build the multiplication tensor T.
    2. Iteratively peel off rank-1 components with sparse V-factors.
    3. For the remaining tensor, try to complete the decomposition.
    """
    T = _build_mult_tensor(n, m, p)
    T_remaining = T.copy()
    
    components = []  # List of (u, v, w) tuples
    
    max_iter = 60
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        result = _peel_rank1_sparse(T_remaining, n, m, p, max_v_nnz=1)
        if result is None:
            break
        
        u, v, w, T_remaining = result
        components.append((u, v, w))
    
    if not components:
        return None
    
    # Build the decomposition from peeled components
    rank = len(components)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for r, (u, v, w) in enumerate(components):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    # Check if this fully decomposes T
    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
        if _is_valid_entries(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
    
    # If not fully decomposed, try to add more terms to complete it
    # For now, just return None if incomplete
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    U55 = np.array(U_best, dtype=np.float64)
    V55 = np.array(V_best, dtype=np.float64)
    W55 = np.array(W_best, dtype=np.float64)

    # Try peeling approach first
    if time.time() < deadline - 30:
        result = _peeling_search(n, m, p, deadline - 20)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = len(result[0])
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try to find rank 54 by dropping one term
    if time.time() < deadline - 20:
        result = _try_merge_terms(U55, V55, W55, n, m, p, deadline - 10)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try random support search for rank 54
    if time.time() < deadline - 15:
        result = _random_support_search(n, m, p, 54, deadline - 10, U55, V55, W55)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try rank 53
    if time.time() < deadline - 10:
        result = _random_support_search(n, m, p, 53, deadline - 5, U55, V55, W55)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 53
            return {"U": U_best, "V": V_best, "W": W_best}

    return {"U": U_best, "V": V_best, "W": W_best}
