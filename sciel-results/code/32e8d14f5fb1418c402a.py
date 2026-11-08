import numpy as np
import time
import sys

_START_TIME = time.time()
DEADLINE_SECONDS = 125

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Uses Strassen's algorithm for 2x2x2 blocks where possible, and schoolbook for the
    remaining 2x1 blocks. Then attempts to reduce rank by finding linear dependencies
    among the rank-1 tensors and expressing them as integer/half-integer combinations.
    """
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14
    total_entries = n * m * p  # 56
    
    def build_strassen_decomposition():
        """
        Build initial decomposition using Strassen for 2x2x2 blocks and schoolbook
        for 2x1 blocks. Returns U, V, W lists and rank.
        """
        rank_max = 50
        U = [[0] * rank_max for _ in range(nA)]
        V = [[0] * rank_max for _ in range(nB)]
        W = [[0] * rank_max for _ in range(nW)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            if col >= rank_max:
                return
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h)
            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1

            # M2 = (c+d)e
            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1

            # M3 = a(f-h)
            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1

            # M4 = d(g-e)
            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1

            # M5 = (a+b)h
            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1

            # M6 = (c-a)(e+f)
            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11][col] += 1
            col += 1

            # M7 = (b-d)(g+h)
            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1
            col += 1

        def add_schoolbook_2x1(i0, j0, k0):
            nonlocal col
            if col >= rank_max:
                return
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1
            col += 1
            U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1
            col += 1
            U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1
            col += 1
            U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen(0, j_start, k_start)
            add_schoolbook_2x1(0, j_start, 6)

        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        return U, V, W, col

    def verify(U, V, W, tol=1e-9):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, rank):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 tensors
        and expressing them as integer/half-integer combinations.
        """
        if rank <= 1:
            return U, V, W, rank
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the matrix of flattened rank-1 tensors
        T_mat = np.zeros((total_entries, rank))
        for l in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * (m * p) + j * p + k
                        T_mat[idx, l] = U_np[i * m + j, l] * V_np[j * p + k, l] * W_np[k * n + i, l]
        
        # Find the rank of T_mat
        try:
            U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
            tol = 1e-10
            effective_rank = int(np.sum(s > tol))
        except:
            return U, V, W, rank
        
        if effective_rank >= rank:
            return U, V, W, rank
        
        # Try to find a subset of columns that spans the column space
        # Use QR with column pivoting to find a basis
        from numpy.linalg import matrix_rank
        
        # For each column, check if it can be expressed as a linear combination of
        # the first r_eff columns (after reordering)
        # This is a heuristic approach - we'll try greedy elimination
        
        # First, let's try to find the minimum subset using greedy method
        used = []
        remaining = list(range(rank))
        
        for _ in range(rank):
            if not remaining:
                break
            # Try each remaining column as the next basis element
            best_col = None
            best_residual = float('inf')
            for col in remaining:
                # Check if this column adds new information
                if not used:
                    best_col = col
                    break
                # Check if this column is linearly independent of used columns
                # by checking if it has a component orthogonal to the span
                vec = T_mat[:, col]
                # Project onto span of used columns
                used_mat = T_mat[:, used]
                try:
                    # Solve for coefficients
                    coeffs, residuals, rank_q, sv = np.linalg.lstsq(used_mat, vec, rcond=None)
                    residual_vec = vec - used_mat @ coeffs
                    residual_norm = np.linalg.norm(residual_vec)
                    if residual_norm < best_residual:
                        best_residual = residual_norm
                        best_col = col
                except:
                    best_col = col
                    break
            
            if best_col is None:
                break
            
            used.append(best_col)
            remaining.remove(best_col)
            
            # Stop if we've found enough independent columns
            if len(used) >= effective_rank:
                break
        
        # Now we have a basis set 'used'. Try to express the other columns as
        # linear combinations of the basis, with integer/half-integer coefficients.
        
        # For this to work, we need to find integer/half-integer coefficients
        # such that the linear combination exactly matches the target tensor.
        
        # This is a difficult problem in general. Let's try a simple approach:
        # For each non-basis column, solve for coefficients and check if they
        # are close to integer/half-integer values.
        
        basis_cols = used
        non_basis_cols = [l for l in range(rank) if l not in used]
        
        if not non_basis_cols:
            return U, V, W, rank
        
        # Solve for coefficients of each non-basis column in terms of basis
        basis_mat = T_mat[:, basis_cols]
        
        # Check if the basis matrix has full column rank
        try:
            rank_basis = np.linalg.matrix_rank(basis_mat)
        except:
            return U, V, W, rank
        
        if rank_basis < len(basis_cols):
            return U, V, W, rank
        
        # For each non-basis column, find coefficients
        coefficients = {}
        for col in non_basis_cols:
            try:
                coeffs, residuals, rank_q, sv = np.linalg.lstsq(basis_mat, T_mat[:, col], rcond=None)
                # Check if coefficients are close to integer/half-integer values
                is_valid = True
                for c in coeffs:
                    # Check if c is close to an integer or half-integer
                    c_round = round(c * 2) / 2
                    if abs(c - c_round) > 1e-6:
                        is_valid = False
                        break
                if is_valid:
                    coefficients[col] = coeffs
            except:
                pass
        
        if not coefficients:
            return U, V, W, rank
        
        # Build new decomposition
        new_rank = len(basis_cols)
        new_U = [row[:new_rank] for row in U_np.tolist()]
        new_V = [row[:new_rank] for row in V_np.tolist()]
        new_W = [row[:new_rank] for row in W_np.tolist()]
        
        # Verify the new decomposition
        if not verify(new_U, new_V, new_W):
            return U, V, W, rank
        
        # Check that all entries are integer or half-integer
        for mat in [new_U, new_V, new_W]:
            for row in mat:
                for val in row:
                    v_round = round(val * 2) / 2
                    if abs(val - v_round) > 1e-6:
                        return U, V, W, rank
        
        return new_U, new_V, new_W, new_rank

    # Build initial decomposition
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nA)]
        V = [[0] * rank for _ in range(nB)]
        W = [[0] * rank for _ in range(nW)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    U[ai][col] = 1
                    V[bi][col] = 1
                    W[wi][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if _time_remaining() > 30:
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
    
    return {"U": U, "V": V, "W": W}
