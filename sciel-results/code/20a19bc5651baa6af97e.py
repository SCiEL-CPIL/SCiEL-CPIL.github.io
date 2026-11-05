import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0  # Safety margin under 150s kill

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the tensor product of Strassen's 2x2x2 decomposition to create a
    rank-49 decomposition for 4x4x4, then duplicates it for the two 4x4 blocks
    of the 4x8 output.
    """
    # Step 1: Get 2x2x2 Strassen tensors
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Step 2: Construct 4x4x4 tensors via tensor product (Kronecker product)
    # Rank is 7 * 7 = 49
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    # Vectorized construction
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    # Step 3: Construct 4x4x8 tensors
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        
        U[:, col_start:col_end] = U4
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Step 4: Verification
    n, m, p = 4, 4, 8
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    U_sel = U[u_rows, :]
    V_sel = V[v_rows, :]
    W_sel = W[w_rows, :]
    
    prod = U_sel * V_sel * W_sel
    total = prod.sum(axis=1)
    
    if not np.allclose(total, 1.0, atol=1e-9):
        return None
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_reduce_rank_98():
    """
    Try to find linear dependencies among the 98 tensor terms to reduce rank.
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We look for a subset of columns that still spans the same tensor.
    """
    n, m, p = 4, 4, 8
    rank_98 = 98
    
    # Build the 98-term decomposition
    U2, V2, W2 = _build_strassen_2x2x2()
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.float64)
    V4 = np.zeros((16, rank_444), dtype=np.float64)
    W4 = np.zeros((16, rank_444), dtype=np.float64)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    U = np.zeros((16, rank_98), dtype=np.float64)
    V = np.zeros((32, rank_98), dtype=np.float64)
    W = np.zeros((32, rank_98), dtype=np.float64)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    # Build the target tensor as a vector of length n*m*p = 128
    # T[i,j,k] = 1 for all (i,j,k)
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    num_terms = n * m * p  # 128
    
    # For each column r, compute the contribution vector:
    # contrib_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find the minimal subset of columns such that
    # sum_{r in subset} contrib_r = ones(128)
    
    # Build matrix M where M[t, r] = contrib_r[t] for t in 0..127, r in 0..97
    M = np.zeros((num_terms, rank_98), dtype=np.float64)
    for r in range(rank_98):
        for t in range(num_terms):
            i = i_flat[t]
            j = j_flat[t]
            k = k_flat[t]
            M[t, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We want to find the minimal subset S such that M[:, S] @ x = ones(128) for some x
    # This is a linear system. First, find the rank of M.
    # If rank(M) < 98, there are dependencies.
    
    # Use SVD to find the rank and null space
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # The effective rank is the number of significant singular values
    tol = 1e-10
    effective_rank = np.sum(s_vals > tol)
    
    if effective_rank >= rank_98:
        # No dependencies found via this method
        return None
    
    # The null space of M^T gives linear dependencies among columns
    # null_space dimension = 98 - effective_rank
    # We can remove (98 - effective_rank) columns
    
    # Actually, we need to be more careful. We need sum_r c_r * col_r = 0
    # The null space of M gives vectors c such that M @ c = 0
    # i.e., sum_r c_r * col_r = 0
    # So we can set one column to be a combination of others.
    
    # For now, let's just check if we can reduce by removing columns
    # that are linear combinations of others.
    
    # Find a basis for the column space
    # The first 'effective_rank' columns in the SVD basis span the column space
    # But we need to find which original columns are redundant.
    
    # Use QR decomposition with column pivoting
    try:
        from scipy.linalg import qr
        Q, R, piv = qr(M.T, pivoting=True)  # M.T is 98 x 128
        # R is 128 x 98, piv is permutation
        # The rank is the number of non-zero diagonal entries of R
        r_diag = np.abs(np.diag(R[:98, :98]))
        rank_qr = np.sum(r_diag > tol)
    except Exception:
        return None
    
    if rank_qr >= rank_98:
        return None
    
    # We can potentially reduce rank by (98 - rank_qr)
    # But we need to verify the reduced decomposition still works.
    
    # For simplicity, let's just return the rank-98 decomposition if no easy reduction
    # The key insight: U is shared, so the 98 columns of U have rank at most 49.
    # This means there ARE dependencies. Let's find them.
    
    # The U matrix has 16 rows and 98 columns. Rank(U) <= 16.
    # So there are at least 98-16=82 linear dependencies in U alone.
    # But the tensor decomposition depends on all three factors together.
    
    # Let's try a greedy approach: start with all 98 columns, try to remove
    # one at a time and see if the remaining still sum to the target.
    
    target = np.ones(num_terms, dtype=np.float64)
    active = list(range(rank_98))
    
    # For each column, check if it's a linear combination of the others
    # by solving M[:, active\{r}] @ x = M[:, r]
    
    reduced = list(active)
    for r in range(rank_98):
        if time.time() > DEADLINE - 30:
            break
        if r not in reduced:
            continue
        # Try removing column r
        remaining = [c for c in reduced if c != r]
        if len(remaining) < effective_rank:
            continue
        M_sub = M[:, remaining]
        target_sub = M[:, r]
        # Solve M_sub @ x = target_sub
        try:
            x, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, target_sub, rcond=tol)
            # Check if the solution is exact
            recon = M_sub @ x
            if np.allclose(recon, target_sub, atol=1e-8):
                # Column r is a linear combination of others
                # But we need integer/half-integer coefficients for the final answer
                # For now, just track that it's removable
                reduced.remove(r)
        except Exception:
            pass
    
    if len(reduced) < rank_98:
        # Build the reduced decomposition
        U_red = U[:, reduced]
        V_red = V[:, reduced]
        W_red = W[:, reduced]
        
        # Verify
        prod = U_red[i_flat*m+j_flat, :] * V_red[j_flat*p+k_flat, :] * W_red[k_flat*n+i_flat, :]
        total = prod.sum(axis=1)
        if np.allclose(total, 1.0, atol=1e-8):
            return {
                "U": U_red.tolist(),
                "V": V_red.tolist(),
                "W": W_red.tolist()
            }
    
    return None

def _schoolbook(n=4, m=4, p=8):
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

def construct():
    # First try to find a reduced-rank decomposition
    try:
        if time.time() < DEADLINE - 60:
            res = _try_reduce_rank_98()
            if res is not None:
                return res
    except Exception:
        pass
    
    # Try the rank-98 block-Strassen decomposition
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
