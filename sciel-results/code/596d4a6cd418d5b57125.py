"""Combined program for matrix multiplication 4x5 * 5x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
_START = time.time()
_DEADLINE = _START + 130.0

def _schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_step(n, m, p, get_decomp):
    """
    One Strassen step: split into halves, use get_decomp for sub-problems, combine.
    Returns (U, V, W) as numpy arrays.
    """
    nh, mh, ph = n // 2, m // 2, p // 2

    # Get sub-decompositions
    M1 = get_decomp(nh, mh, ph)
    M2 = get_decomp(nh, mh, ph)
    M3 = get_decomp(nh, mh, ph)
    M4 = get_decomp(nh, mh, ph)
    M5 = get_decomp(nh, mh, ph)
    M6 = get_decomp(nh, mh, ph)
    M7 = get_decomp(nh, mh, ph)

    r = M1[0].shape[1]
    rank = 7 * r

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Helper functions to add sub-matrix contributions
    def add_U(U, sub_U, ii, jj, col_offset):
        for ii_local in range(nh):
            for jj_local in range(mh):
                row = (ii * nh + ii_local) * m + (jj * mh + jj_local)
                sub_row = ii_local * mh + jj_local
                U[row, col_offset:col_offset + r] += sub_U[sub_row, :]

    def add_V(V, sub_V, jj, kk, col_offset):
        for jj_local in range(mh):
            for kk_local in range(ph):
                row = (jj * mh + jj_local) * p + (kk * ph + kk_local)
                sub_row = jj_local * ph + kk_local
                V[row, col_offset:col_offset + r] += sub_V[sub_row, :]

    def add_W(W, sub_W, kk, ii, col_offset):
        for kk_local in range(ph):
            for ii_local in range(nh):
                row = (kk * ph + kk_local) * n + (ii * nh + ii_local)
                sub_row = kk_local * nh + ii_local
                W[row, col_offset:col_offset + r] += sub_W[sub_row, :]

    # M1 = (A00+A11)(B00+B11)
    add_U(U, M1[0], 0, 0, 0)
    add_U(U, M1[0], 1, 1, 0)
    add_V(V, M1[1], 0, 0, 0)
    add_V(V, M1[1], 1, 1, 0)
    add_W(W, M1[2], 0, 0, 0)
    add_W(W, M1[2], 1, 1, 0)

    # M2 = (A10-A00)(B00+B01)
    add_U(U, M2[0], 1, 0, r)
    # Subtract A00 contribution
    for ii_local in range(nh):
        for jj_local in range(mh):
            row_a00 = ii_local * m + jj_local
            sub_row = ii_local * mh + jj_local
            U[row_a00, r:r + r] -= M2[0][sub_row, :]
    add_V(V, M2[1], 0, 0, r)
    add_V(V, M2[1], 0, 1, r)
    add_W(W, M2[2], 0, 1, r)
    add_W(W, M2[2], 1, 1, r)

    # M3 = A00(B10-B11)
    add_U(U, M3[0], 0, 0, 2 * r)
    add_V(V, M3[1], 1, 0, 2 * r)
    # Subtract B11 contribution
    for jj_local in range(mh):
        for kk_local in range(ph):
            row_b11 = (mh + jj_local) * p + (ph + kk_local)
            sub_row = jj_local * ph + kk_local
            V[row_b11, 2 * r:2 * r + r] -= M3[1][sub_row, :]
    add_W(W, M3[2], 0, 0, 2 * r)
    add_W(W, M3[2], 1, 1, 2 * r)

    # M4 = A11(B01-B00)
    add_U(U, M4[0], 1, 1, 3 * r)
    add_V(V, M4[1], 0, 1, 3 * r)
    # Subtract B00 contribution
    for jj_local in range(mh):
        for kk_local in range(ph):
            row_b00 = jj_local * p + kk_local
            sub_row = jj_local * ph + kk_local
            V[row_b00, 3 * r:3 * r + r] -= M4[1][sub_row, :]
    add_W(W, M4[2], 0, 0, 3 * r)
    add_W(W, M4[2], 0, 1, 3 * r)

    # M5 = (A00+A10)B11
    add_U(U, M5[0], 0, 0, 4 * r)
    add_U(U, M5[0], 1, 0, 4 * r)
    add_V(V, M5[1], 1, 1, 4 * r)
    add_W(W, M5[2], 1, 0, 4 * r)
    add_W(W, M5[2], 1, 1, 4 * r)

    # M6 = (A10-A11)(B01+B11)
    add_U(U, M6[0], 1, 0, 5 * r)
    # Subtract A11 contribution
    for ii_local in range(nh):
        for jj_local in range(mh):
            row_a11 = (nh + ii_local) * m + (mh + jj_local)
            sub_row = ii_local * mh + jj_local
            U[row_a11, 5 * r:5 * r + r] -= M6[0][sub_row, :]
    add_V(V, M6[1], 0, 1, 5 * r)
    add_V(V, M6[1], 1, 1, 5 * r)
    add_W(W, M6[2], 1, 1, 5 * r)
    add_W(W, M6[2], 0, 1, 5 * r)

    # M7 = (A01-A10)(B00+B10)
    add_U(U, M7[0], 0, 1, 6 * r)
    # Subtract A10 contribution
    for ii_local in range(nh):
        for jj_local in range(mh):
            row_a10 = (nh + ii_local) * m + jj_local
            sub_row = ii_local * mh + jj_local
            U[row_a10, 6 * r:6 * r + r] -= M7[0][sub_row, :]
    add_V(V, M7[1], 0, 0, 6 * r)
    add_V(V, M7[1], 1, 0, 6 * r)
    add_W(W, M7[2], 0, 0, 6 * r)
    add_W(W, M7[2], 1, 0, 6 * r)

    return U, V, W

def _try_strassen_for_4x5x6():
    """
    Try to construct a Strassen-like decomposition for 4x5*5x6.
    Strategy: Pad to 4x6*6x6 or use a custom decomposition.
    For now, try padding approach: treat as 4x6 * 6x6 with last column/row zeroed.
    """
    n, m, p = 4, 5, 6
    
    # Try to decompose 4x6 * 6x6 using Strassen (both even)
    # Then adapt to 4x5 * 5x6
    # This is complex; instead, let's try a direct recursive approach on 4x6*6x6
    
    # Actually, let's just try the even case 4x6*6x6
    n2, m2, p2 = 4, 6, 6
    try:
        U2, V2, W2 = _strassen_step(n2, m2, p2, lambda a, b, c: _schoolbook(a, b, c))
        # Now we have decomposition for 4x6 * 6x6
        # We need 4x5 * 5x6
        # The tensor for 4x5*5x6 is a sub-tensor of 4x6*6x6
        # Specifically, we can project: A has shape 4x5 (first 5 cols of 4x6), 
        # B has shape 5x6 (first 5 rows of 6x6)
        # This projection reduces rank potentially
        
        # U2 has shape (4*6, rank2) = (24, rank2)
        # For 4x5, we only need rows corresponding to i in [0,4), j in [0,5)
        # Row index in U2: i*6 + j, we need j in [0,5), so rows 0-4, 6-10, 12-16, 18-22
        U_rows = []
        for i in range(4):
            for j in range(5):
                U_rows.append(i * 6 + j)
        U = U2[U_rows, :]
        
        # V2 has shape (6*6, rank2) = (36, rank2)
        # For 5x6, we need rows corresponding to j in [0,5), k in [0,6)
        # Row index in V2: j*6 + k, we need j in [0,5), so rows 0-5, 6-11, 12-17, 18-23, 24-29
        V_rows = []
        for j in range(5):
            for k in range(6):
                V_rows.append(j * 6 + k)
        V = V2[V_rows, :]
        
        # W2 has shape (6*4, rank2) = (24, rank2)
        # For 6x4 output, W[k*n+i, c], k in [0,6), i in [0,4)
        # Row index in W2: k*4 + i, all rows needed
        W = W2[:, :]
        
        return U, V, W
    except Exception as e:
        print(f"Strassen padding failed: {e}")
        return None

def _optimize_rank(U, V, W):
    """
    Try to reduce rank by finding linear dependencies among columns.
    Uses SVD to identify low-rank structure.
    """
    try:
        # The decomposition is valid if for all i,j,k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
        # We can form the "contraction tensor" and check if columns are independent
        
        n, m, p = DIMENSIONS
        
        # For each column c, we have a rank-1 tensor T_c with entries
        # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # The sum over c should give the multiplication tensor
        
        # To find dependencies, we can flatten the 3D tensor for each column
        # and check if the set of flattened vectors is linearly independent
        
        rank = U.shape[1]
        if rank == 0:
            return U, V, W
        
        # Create matrix where each column is a flattened rank-1 tensor
        # Shape: (rank, n*m*p)
        T_flat = np.zeros((rank, n * m * p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_flat[c, k * n * m + i * m + j] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        
        # Find rank of T_flat
        # Use SVD to identify significant singular values
        try:
            U_svd, s, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
            # Count significant singular values
            tol = 1e-10 * max(s) if len(s) > 0 else 0
            effective_rank = np.sum(s > tol)
            
            if effective_rank < rank:
                # We can reduce rank
                # Keep only the top effective_rank components
                # This requires recombining, which is complex
                # For now, just return as is
                pass
        except Exception:
            pass
        
        return U, V, W
    except Exception:
        return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # First, try schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try Strassen padding approach
    result = _try_strassen_for_4x5x6()
    if result is not None:
        U2, V2, W2 = result
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Try to optimize rank
    U, V, W = _optimize_rank(U, V, W)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
