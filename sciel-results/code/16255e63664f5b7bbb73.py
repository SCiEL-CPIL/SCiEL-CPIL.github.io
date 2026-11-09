import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
_DEADLINE = 120.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i==i, j==j, k==k (diagonal) and 0 otherwise
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if we're computing the (i,k) entry from (i,j) and (j,k)
    # The tensor is: T[i,j,k] = delta(i,j,k) meaning it's 1 when the indices form a valid triple
    # For matmul 3x3*3x4: T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,4)
    # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor has 1s at all (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False
    return True

def _verify_float(U, V, W, tol=1e-9):
    """Verify using floating point for speed."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Uf[i*m+j], Vf[j*p+k] * Wf[k*n+i])
                if abs(val - 1) > tol:
                    return False
    return True

def _construct_rank34():
    """Fallback rank-34 decomposition using Strassen for 2x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    # Two 2x2 Strassen blocks for columns 0-1 and 2-3
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    # Row 2 of A with all columns of B
    for j in range(m):
        for k in range(p):
            add_term(u_idx(2,j), 1, v_idx(j,k), 1, c_idx(2,k), 1)

    # Rows 0,1 of A with row 2 of B
    for i in range(2):
        for k in range(p):
            add_term(u_idx(i,2), 1, v_idx(2,k), 1, c_idx(i,k), 1)

    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, target_rank, max_iters=2000):
    """Try to reduce rank by finding linear dependencies among terms."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if rank <= target_rank:
        return U, V, W
    
    # Convert to numpy for numerical linear algebra
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    
    # For each term r, the contribution to the tensor is the outer product
    # of U[:,r], V[:,r], W[:,r]. We want to find a linear combination
    # of terms that gives zero tensor contribution.
    
    # Flatten each term into a vector of length n*m*p
    # term_vector[r][i*m+j, j*p+k, k*n+i] ... 
    # Actually, the tensor entry T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # So for a fixed r, the "contribution vector" has entries:
    # contrib[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # We want to find coefficients alpha_r such that sum_r alpha_r * contrib_r = 0
    # This is a linear system in alpha.
    
    # Build matrix: rows = (i,j,k) indices, cols = r
    # M[i,j,k, r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    total_entries = n * m * p
    M = np.zeros((total_entries, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    M[idx, r] = Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i, r]
                idx += 1
    
    # Find null space of M
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
    except np.linalg.LinAlgError:
        return U, V, W
    
    # Null space vectors are rows of Vt where corresponding singular value is ~0
    tol = 1e-10
    null_dim = 0
    for i in range(rank):
        if s[i] < tol:
            null_dim += 1
    
    if null_dim == 0:
        return U, V, W
    
    # Get null space basis
    null_space = Vt[rank-null_dim:rank, :].T  # shape (rank, null_dim)
    
    # Try to eliminate terms using null space vectors
    # For each null vector, we can add it to the terms
    # We want to find a set of terms to eliminate
    
    # Strategy: greedily find a null vector that has a large component
    # on some term, and use it to eliminate that term
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for iter in range(max_iters):
        if _remaining() < 5:
            break
        if best_rank <= target_rank:
            break
        
        # Find a term that can be eliminated
        # Look for a null vector with a nonzero component
        eliminated = False
        for null_vec_idx in range(null_dim):
            nv = null_space[:, null_vec_idx]
            # Find the term with largest absolute value in this null vector
            r_max = np.argmax(np.abs(nv))
            if abs(nv[r_max]) < 1e-8:
                continue
            
            # Check if this term is still "active" (not already eliminated)
            # We'll track which terms are active
            # For simplicity, try to eliminate term r_max
            
            # New terms: for each r != r_max, add (nv[r]/nv[r_max]) * term_r to term_r_max
            # Actually, we want to express term_r_max as a combination of others
            # alpha[r_max] * term_r_max + sum_{r != r_max} alpha[r] * term_r = 0
            # So term_r_max = -sum_{r != r_max} (alpha[r]/alpha[r_max]) * term_r
            
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            
            # Set term r_max to zero by subtracting its contribution
            # Actually, we replace term r_max with a combination
            # term_new[r_max] = sum_{r != r_max} (-nv[r]/nv[r_max]) * term_r
            
            coeff = -1.0 / nv[r_max]
            for r in range(rank):
                if r == r_max:
                    continue
                if abs(nv[r]) < 1e-15:
                    continue
                c = coeff * nv[r]
                for i in range(n*m):
                    new_U[i][r_max] += c * U[i][r]
                for i in range(m*p):
                    new_V[i][r_max] += c * V[i][r]
                for i in range(p*n):
                    new_W[i][r_max] += c * W[i][r]
            
            # Now term r_max should be (nearly) zero
            # Check if it's actually zero
            max_val = 0
            for i in range(n*m):
                max_val = max(max_val, abs(new_U[i][r_max]))
            for i in range(m*p):
                max_val = max(max_val, abs(new_V[i][r_max]))
            for i in range(p*n):
                max_val = max(max_val, abs(new_W[i][r_max]))
            
            if max_val < 1e-8:
                # Successfully eliminated term r_max
                # Remove it from the decomposition
                new_U = [row[:r_max] + row[r_max+1:] for row in new_U]
                new_V = [row[:r_max] + row[r_max+1:] for row in new_V]
                new_W = [row[:r_max] + row[r_max+1:] for row in new_W]
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank -= 1
                eliminated = True
                
                # Update null space for the new decomposition
                # Rebuild M for the new decomposition
                rank_new = best_rank
                M_new = np.zeros((total_entries, rank_new))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(rank_new):
                                M_new[idx, r] = new_U[i*m+j][r] * new_V[j*p+k][r] * new_W[k*n+i][r]
                            idx += 1
                try:
                    U_svd, s, Vt = np.linalg.svd(M_new, full_matrices=True)
                except np.linalg.LinAlgError:
                    break
                null_dim_new = 0
                for i in range(rank_new):
                    if s[i] < tol:
                        null_dim_new += 1
                if null_dim_new == 0:
                    break
                null_space = Vt[rank_new-null_dim_new:rank_new, :].T
                null_dim = null_dim_new
                break
            else:
                # Couldn't eliminate cleanly, try to clean up
                # Set small values to zero
                for i in range(n*m):
                    if abs(new_U[i][r_max]) < 1e-10:
                        new_U[i][r_max] = 0
                for i in range(m*p):
                    if abs(new_V[i][r_max]) < 1e-10:
                        new_V[i][r_max] = 0
                for i in range(p*n):
                    if abs(new_W[i][r_max]) < 1e-10:
                        new_W[i][r_max] = 0
                
                # Check if it's now zero
                max_val = 0
                for i in range(n*m):
                    max_val = max(max_val, abs(new_U[i][r_max]))
                for i in range(m*p):
                    max_val = max(max_val, abs(new_V[i][r_max]))
                for i in range(p*n):
                    max_val = max(max_val, abs(new_W[i][r_max]))
                
                if max_val < 1e-8:
                    new_U = [row[:r_max] + row[r_max+1:] for row in new_U]
                    new_V = [row[:r_max] + row[r_max+1:] for row in new_V]
                    new_W = [row[:r_max] + row[r_max+1:] for row in new_W]
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank -= 1
                    eliminated = True
                    
                    rank_new = best_rank
                    M_new = np.zeros((total_entries, rank_new))
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for r in range(rank_new):
                                    M_new[idx, r] = new_U[i*m+j][r] * new_V[j*p+k][r] * new_W[k*n+i][r]
                                idx += 1
                    try:
                        U_svd, s, Vt = np.linalg.svd(M_new, full_matrices=True)
                    except np.linalg.LinAlgError:
                        break
                    null_dim_new = 0
                    for i in range(rank_new):
                        if s[i] < tol:
                            null_dim_new += 1
                    if null_dim_new == 0:
                        break
                    null_space = Vt[rank_new-null_dim_new:rank_new, :].T
                    null_dim = null_dim_new
                    break
        
        if not eliminated:
            # No more progress
            break
    
    # Verify the result
    if best_rank < rank:
        if _verify_float(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return U, V, W

def _try_gf2_search(U, V, W, target_rank):
    """Try to find a rank reduction using GF(2) linear algebra."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if rank <= target_rank:
        return U, V, W
    
    # Convert to GF(2) - only works if entries are 0 or 1
    # Check if all entries are 0 or 1
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if val not in [0, 1]:
                    return U, V, W
    
    # Build the tensor matrix over GF(2)
    total_entries = n * m * p
    # M[i,j,k, r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] mod 2
    M = [[0]*rank for _ in range(total_entries)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    M[idx][r] = (U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]) % 2
                idx += 1
    
    # Gaussian elimination over GF(2) to find null space
    # We want to find vectors alpha in GF(2)^rank such that M * alpha = 0
    
    # Perform row reduction
    M_copy = [row[:] for row in M]
    pivot_cols = []
    row_idx = 0
    for col in range(rank):
        # Find pivot
        pivot = -1
        for r in range(row_idx, total_entries):
            if M_copy[r][col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap
        M_copy[row_idx], M_copy[pivot] = M_copy[pivot], M_copy[row_idx]
        # Eliminate
        for r in range(total_entries):
            if r != row_idx and M_copy[r][col] == 1:
                for c in range(col, rank):
                    M_copy[r][c] = (M_copy[r][c] + M_copy[row_idx][c]) % 2
        pivot_cols.append(col)
        row_idx += 1
        if row_idx >= total_entries:
            break
    
    # Free columns are those not in pivot_cols
    free_cols = [c for c in range(rank) if c not in pivot_cols]
    null_dim = len(free_cols)
    
    if null_dim == 0:
        return U, V, W
    
    # Find null space basis over GF(2)
    null_basis = []
    for fc in free_cols:
        vec = [0]*rank
        vec[fc] = 1
        for pc in pivot_cols:
            # Find which row has pivot in column pc
            for r in range(len(pivot_cols)):
                if pivot_cols[r] == pc:
                    vec[pc] = M_copy[r][fc]
                    break
        null_basis.append(vec)
    
    # Try to eliminate terms using null space vectors
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for nv in null_basis:
        if _remaining() < 5:
            break
        if best_rank <= target_rank:
            break
        
        # Find the term with value 1 in this null vector
        ones = [r for r in range(best_rank) if nv[r] == 1]
        if len(ones) < 2:
            continue
        
        # Pick the first one to eliminate
        r_max = ones[0]
        others = ones[1:]
        
        # term_r_max = sum_{r in others} term_r (mod 2)
        new_U = [row[:] for row in best_U]
        new_V = [row[:] for row in best_V]
        new_W = [row[:] for row in best_W]
        
        for r in others:
            for i in range(n*m):
                new_U[i][r_max] = (new_U[i][r_max] + best_U[i][r]) % 2
            for i in range(m*p):
                new_V[i][r_max] = (new_V[i][r_max] + best_V[i][r]) % 2
            for i in range(p*n):
                new_W[i][r_max] = (new_W[i][r_max] + best_W[i][r]) % 2
        
        # Check if term r_max is now zero
        is_zero = True
        for i in range(n*m):
            if new_U[i][r_max] != 0:
                is_zero = False
                break
        if is_zero:
            for i in range(m*p):
                if new_V[i][r_max] != 0:
                    is_zero = False
                    break
        if is_zero:
            for i in range(p*n):
                if new_W[i][r_max] != 0:
                    is_zero = False
                    break
        
        if is_zero:
            new_U = [row[:r_max] + row[r_max+1:] for row in new_U]
            new_V = [row[:r_max] + row[r_max+1:] for row in new_V]
            new_W = [row[:r_max] + row[r_max+1:] for row in new_W]
            best_U, best_V, best_W = new_U, new_V, new_W
            best_rank -= 1
    
    if best_rank < rank:
        if _verify_float(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return U, V, W

def _structured_search(max_rank=30, time_budget=60):
    """Structured search for low-rank decompositions."""
    n, m, p = DIMENSIONS
    
    # Start with the rank-34 decomposition
    result = _construct_rank34()
    U, V, W = result["U"], result["V"], result["W"]
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try to reduce rank
    U2, V2, W2 = _try_reduce_rank(U, V, W, max_rank)
    if len(U2[0]) < best_rank:
        best_U, best_V, best_W = U2, V2, W2
        best_rank = len(U2[0])
    
    U3, V3, W3 = _try_gf2_search(best_U, best_V, best_W, max_rank)
    if len(U3[0]) < best_rank:
        best_U, best_V, best_W = U3, V3, W3
        best_rank = len(U3[0])
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    global _START
    _START = time.time()
    
    try:
        # First, get the rank-34 baseline
        baseline = _construct_rank34()
        
        # Verify it
        if not _verify_float(baseline["U"], baseline["V"], baseline["W"]):
            # Fallback to schoolbook
            return _schoolbook()
        
        # Try to reduce rank
        if _remaining() > 10:
            result = _structured_search(max_rank=29, time_budget=min(_remaining()-5, 60))
            if result is not None:
                if _verify_float(result["U"], result["V"], result["W"]):
                    return result
        
        return baseline
    except Exception as e:
        # Fallback to schoolbook
        return _schoolbook()
