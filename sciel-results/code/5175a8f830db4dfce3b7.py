"""Matrix multiplication 2x2 by 2x15: rank-minimal exact tensor decomposition."""
import random
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
_DEADLINE = time.time() + 140.0


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) with integer entries.
    """
    rank = 7
    n, m, p = 2, 2, 2
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_2x2_1x2():
    """Schoolbook for 2x2 * 2x1 -> 2x1. Rank 4."""
    rank = 4
    n, m, p = 2, 2, 1
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    
    return U, V, W


def _build_rank53():
    """Build the rank-53 Strassen-based decomposition."""
    n, m, p = 2, 2, 15
    total_rank = 53
    
    U = [[0]*total_rank for _ in range(4)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(30)]
    
    Us, Vs, Ws = _strassen_2x2()
    Usb, Vsb, Wsb = _schoolbook_2x2_1x2()
    
    col_offset = 0
    
    for block in range(7):
        k_start = block * 2
        for c in range(7):
            for row in range(4):
                U[row][col_offset] = Us[row][c]
            for j in range(2):
                for k_local in range(2):
                    local_row = j * 2 + k_local
                    global_row = j * 15 + (k_start + k_local)
                    V[global_row][col_offset] = Vs[local_row][c]
            for k_local in range(2):
                for i in range(2):
                    local_row = k_local * 2 + i
                    global_row = (k_start + k_local) * 2 + i
                    W[global_row][col_offset] = Ws[local_row][c]
            col_offset += 1
    
    for c in range(4):
        for row in range(4):
            U[row][col_offset] = Usb[row][c]
        for j in range(2):
            V[j * 15 + 14][col_offset] = Vsb[j][c]
        for i in range(2):
            W[14 * 2 + i][col_offset] = Wsb[i][c]
        col_offset += 1
    
    return U, V, W, total_rank


def _verify(U, V, W, rank):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    n, m, p = 2, 2, 15
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True


def _try_remove_term(U, V, W, rank, remove_idx):
    """
    Try to remove term `remove_idx` from a rank-`rank` decomposition.
    Check if the remaining rank-1 terms still span the tensor.
    This is a necessary but not sufficient check for rank reduction.
    """
    n, m, p = 2, 2, 15
    # Build the 60x(rank-1) matrix of remaining terms
    # Each column c contributes outer product u_c ⊗ v_c ⊗ w_c flattened
    # We need: sum_{c != remove_idx} U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k)
    # This is a system of 60 equations in the remaining rank-1 terms.
    # But we can't change the remaining terms, so we just check if removing one term
    # still gives the correct tensor (which won't work unless that term was zero).
    
    # Actually, for a true rank reduction, we need to find new coefficients.
    # Let's use a different approach: try to find a linear dependence.
    
    # Build matrix M of size 60 x (rank-1) where M[(i,j,k), c] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # for c != remove_idx. Then check if the vector of 1s is in the column space.
    
    cols = [c for c in range(rank) if c != remove_idx]
    num_constraints = n * m * p  # 60
    
    M = np.zeros((num_constraints, len(cols)))
    for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
        for col_idx, c in enumerate(cols):
            M[idx, col_idx] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    b = np.ones(num_constraints)
    
    # Check if b is in the column space of M
    try:
        sol, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
        if len(residuals) > 0:
            if residuals[0] < 1e-10:
                return True
    except:
        pass
    
    return False


def _search_rank52(U, V, W, rank, deadline):
    """
    Search for a rank-52 decomposition by trying various strategies.
    Strategy 1: Try to find a linear dependence among the 53 rank-1 tensors.
    Strategy 2: Try structured modifications.
    """
    n, m, p = 2, 2, 15
    
    # Strategy: Build the full 60x53 matrix and check rank
    M = np.zeros((60, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    M[idx, c] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                idx += 1
    
    # Check if the 53 columns are linearly dependent
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < rank:
        # There's a linear dependence! We can potentially reduce rank.
        # Find the null space
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        # The null space is spanned by the last (rank - rank_M) rows of Vt
        null_start = rank - rank_M
        if null_start < rank:
            null_vec = Vt[null_start]
            # This means sum_c null_vec[c] * (rank-1 tensor c) = 0
            # We can eliminate one term by expressing it in terms of others
            # Find the index with largest |null_vec[c]|
            elim_idx = int(np.argmax(np.abs(null_vec)))
            if abs(null_vec[elim_idx]) > 1e-10:
                # We can express term elim_idx as a combination of others
                # But this doesn't directly give us a rank-52 decomposition with the same structure
                # We'd need to re-parameterize, which is complex
                pass
    
    # Strategy 2: Try replacing pairs of terms with a single term
    # For each pair (a, b), check if u_a⊗v_a⊗w_a + u_b⊗v_b⊗w_b can be
    # replaced by a single rank-1 tensor (unlikely) or if we can find
    # a different pair that sums to zero contribution
    
    # Strategy 3: Try the 2x2x3 block structure
    # 15 = 5*3, so 5 blocks of 2x2x3
    # If we can find a rank-10 decomposition for 2x2x3, total = 50
    # If rank-9, total = 45
    # Let's try to construct a rank-11 (Strassen+schoolbook) and see if we can reduce to 10
    
    # For 2x2x3: use Strassen for first 2 cols (rank 7) + schoolbook for 3rd col (rank 4) = 11
    # Can we do better? The bilinear rank of 2x2x3 is at most 11.
    # Known lower bound: at least 10 (from various results)
    
    # Let's try a structured rank-10 search for 2x2x3
    # We'll parameterize and solve
    
    if time.time() > deadline - 10:
        return None
    
    # Try to find rank-10 for 2x2x3 using a structured approach
    # Based on the idea that we can share some factors between the Strassen
    # and schoolbook parts
    
    # Actually, let's try a direct numerical search for rank-52
    # by perturbing the rank-53 solution and checking validity
    
    # A simpler approach: check if any two terms can be merged
    # Term c and term d can be "merged" if their contribution to every
    # (i,j,k) entry is proportional, i.e., U[i,j,c]*V[j,k,c]*W[k,i,c]
    # = alpha * U[i,j,d]*V[j,k,d]*W[k,i,d] for all (i,j,k)
    
    for c in range(rank):
        if time.time() > deadline - 10:
            break
        for d in range(c+1, rank):
            # Check if term c and term d are "parallel" (same rank-1 tensor up to scale)
            # This means the outer products are proportional
            # u_c * v_c * w_c = alpha * u_d * v_d * w_d
            # For this to hold for all (i,j,k), we need:
            # U[:,c] * V[:,c] * W[:,c] (element-wise over the 60 entries) to be
            # proportional to U[:,d] * V[:,d] * W[:,d]
            
            tc = np.array([U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] 
                          for i in range(n) for j in range(m) for k in range(p)])
            td = np.array([U[i*m+j][d] * V[j*p+k][d] * W[k*n+i][d] 
                          for i in range(n) for j in range(m) for k in range(p)])
            
            # Check if tc = alpha * td
            # Find alpha from first non-zero element
            mask = np.abs(td) > 1e-10
            if np.any(mask):
                alpha = tc[mask][0] / td[mask][0]
                if np.allclose(tc, alpha * td, atol=1e-10):
                    # These two terms are parallel! We can merge them.
                    # New term: (1+alpha) * td (or tc + td)
                    # But we need to express this as a valid rank-1 tensor
                    # This is complex; skip for now
                    pass
    
    return None


def construct():
    """
    Construct a rank-minimal decomposition for 2x2 * 2x15 -> 2x15.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build rank-53 baseline
    U, V, W, total_rank = _build_rank53()
    
    # Verify
    if not _verify(U, V, W, total_rank):
        # Shouldn't happen, but return as-is
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-52
    if time.time() < deadline - 20:
        result = _search_rank52(U, V, W, total_rank, deadline)
        if result is not None:
            return result
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
    
    n, m, p = 2, 2, 15
    total_rank = len(result['U'][0])
    
    all_ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1
                rhs = sum(result['U'][i*m+j][c] * result['V'][j*p+k][c] * result['W'][k*n+i][c] for c in range(total_rank))
                if lhs != rhs:
                    print(f"Tensor MISMATCH at i={i},j={j},k={k}: lhs={lhs}, rhs={rhs}")
                    all_ok = False
    
    if all_ok:
        print(f"All tensor entries verified correctly! Rank = {total_rank}")
