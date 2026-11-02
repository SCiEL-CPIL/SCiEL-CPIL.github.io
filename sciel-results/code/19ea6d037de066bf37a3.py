import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        """Returns list of 7 (U_dict, V_dict, W_dict) for Strassen 2x2x2 block."""
        cols = []
        # M1 = (A11+A22)(B11+B22)
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        # M2 = (A21+A22)B11
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        # M3 = A11(B12-B22)
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        # M4 = A22(B21-B11)
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        # M5 = (A11+A12)B22
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        # M6 = (A21-A11)(B11+B12)
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        # M7 = (A12-A22)(B21+B22)
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        return cols

    def direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        for di in range(nc):
            for dj in range(mc):
                for dk in range(pc):
                    u_idx = (ii + di) * m_dim + (jj + dj)
                    v_idx = (jj + dj) * p_dim + (kk + dk)
                    w_idx = (kk + dk) * n_dim + (ii + di)
                    cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

    def decompose_recursive(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        if time.time() > deadline:
            return
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        if nc >= 2 and mc >= 2 and pc >= 2:
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            if nc > 2:
                decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            if mc > 2:
                decompose_recursive(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            if pc > 2:
                decompose_recursive(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            return
        # Fallback split
        if nc > 1:
            decompose_recursive(1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc - 1, mc, pc, ii + 1, jj, kk, n_dim, m_dim, p_dim, cols)
        elif mc > 1:
            decompose_recursive(nc, 1, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc - 1, pc, ii, jj + 1, kk, n_dim, m_dim, p_dim, cols)
        elif pc > 1:
            decompose_recursive(nc, mc, 1, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc, pc - 1, ii, jj, kk + 1, n_dim, m_dim, p_dim, cols)

    def cols_to_matrices(cols):
        """Convert list of (u_dict, v_dict, w_dict) to U, V, W matrices."""
        rank = len(cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (u_dict, v_dict, w_dict) in enumerate(cols):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        rank = U.shape[1]
        # Build tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T[i, j, k] = s
        return np.allclose(T, 1.0, atol=tol)

    def verify_exact(U_frac, V_frac, W_frac):
        """Verify exact tensor decomposition with Fraction arithmetic."""
        rank = len(U_frac[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                    if s != Fraction(1):
                        return False
        return True

    def check_removable_float(U, V, remove_indices):
        """
        Check if a float solution for W exists for the given U, V and removed columns.
        Returns True if a solution exists, False otherwise.
        """
        rank = U.shape[1]
        remaining_indices = [r for r in range(rank) if r not in remove_indices]
        if len(remaining_indices) < m:
            return False
            
        U_rem = U[:, remaining_indices]
        V_rem = V[:, remaining_indices]
        
        # For each (i, k), we need to solve A * w = 1
        # A is m x len(remaining_indices)
        # A[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, len(remaining_indices)))
                for j in range(m):
                    for r_idx, r_orig in enumerate(remaining_indices):
                        A[j, r_idx] = U[i*m+j, r_orig] * V[j*p+k, r_orig]
                
                b = np.ones(m)
                
                # Check if system is consistent
                # Use least squares and check residual
                try:
                    sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                    if residuals is None or len(residuals) == 0:
                        # Underdetermined or exactly determined
                        # Check if A @ sol == b
                        if np.allclose(A @ sol, b, atol=1e-8):
                            continue
                        else:
                            return False
                    else:
                        if residuals[0] > 1e-8:
                            return False
                except np.linalg.LinAlgError:
                    return False
        
        return True

    def solve_for_W_exact(U_frac, V_frac, remaining_indices):
        """
        Exact rational solve for W given U and V as Fraction matrices,
        using only the columns in remaining_indices.
        Returns W as list of lists of Fraction (shape p*n x len(remaining_indices)), or None.
        """
        num_vars = len(remaining_indices)
        W = [[Fraction(0) for _ in range(num_vars)] for _ in range(p*n)]
        
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_{r in rem} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r_idx] = 1
                # A is m x num_vars
                A = [[Fraction(0) for _ in range(num_vars)] for _ in range(m)]
                b = [Fraction(1) for _ in range(m)]
                
                for j in range(m):
                    for r_idx, r_orig in enumerate(remaining_indices):
                        A[j][r_idx] = U_frac[i*m+j][r_orig] * V_frac[j*p+k][r_orig]
                
                # Solve A * w = b
                # Gaussian elimination
                augmented = [A[i][:] + [b[i]] for i in range(m)]
                
                # We have m equations and num_vars unknowns.
                # If num_vars >= m, we can set free variables to 0 and solve for the first m pivots.
                # If num_vars < m, it might be overdetermined.
                
                # Perform row reduction
                rank_A = 0
                for col in range(num_vars):
                    if rank_A >= m:
                        break
                    # Find pivot
                    pivot = None
                    for row in range(rank_A, m):
                        if augmented[row][col] != 0:
                            pivot = row
                            break
                    if pivot is None:
                        continue
                    # Swap
                    augmented[rank_A], augmented[pivot] = augmented[pivot], augmented[rank_A]
                    # Normalize
                    pivot_val = augmented[rank_A][col]
                    augmented[rank_A] = [x / pivot_val for x in augmented[rank_A]]
                    # Eliminate other rows
                    for row in range(m):
                        if row != rank_A and augmented[row][col] != 0:
                            factor = augmented[row][col]
                            augmented[row] = [augmented[row][t] - factor * augmented[rank_A][t] for t in range(num_vars + 1)]
                    rank_A += 1
                
                # Check consistency
                for row in range(rank_A, m):
                    if any(augmented[row][t] != 0 for t in range(num_vars)):
                        return None
                    if augmented[row][num_vars] != 0:
                        return None
                
                # Extract solution (set free variables to 0)
                w_sol = [Fraction(0) for _ in range(num_vars)]
                for r_idx in range(rank_A):
                    w_sol[r_idx] = augmented[r_idx][num_vars]
                
                W[k*n+i] = w_sol
        
        return W

    def generate_strassen_decomposition():
        """Generate the Strassen-based decomposition."""
        cols = []
        decompose_recursive(n, m, p, 0, 0, 0, n, m, p, cols)
        return cols

    def try_optimized_elimination():
        """
        Try to reduce rank by finding a set of columns to remove.
        Uses float check first, then exact solve.
        """
        if time.time() > deadline:
            return None
        
        cols = generate_strassen_decomposition()
        U, V, W = cols_to_matrices(cols)
        rank = len(cols)
        
        # Convert to Fraction for exact arithmetic
        U_frac = [[Fraction(U[i, r]).limit_denominator(10**6) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, r]).limit_denominator(10**6) for r in range(rank)] for i in range(m*p)]
        
        # Verify base
        W_base_frac = [[Fraction(W[i, r]).limit_denominator(10**6) for r in range(rank)] for i in range(p*n)]
        if not verify_exact(U_frac, V_frac, W_base_frac):
            return None

        best_rank = rank
        best_remaining = list(range(rank))
        
        # Try removing 1 column
        for r in range(rank):
            if time.time() > deadline:
                break
            remove_set = {r}
            if check_removable_float(U, V, remove_set):
                remaining = [x for x in range(rank) if x != r]
                W_new = solve_for_W_exact(U_frac, V_frac, remaining)
                if W_new is not None:
                    # Verify exact
                    if verify_exact(U_frac, V_frac, W_new):
                        best_rank = rank - 1
                        best_remaining = remaining
                        # Don't break, might find more? No, removing 1 is max 1 reduction in this loop.
                        # But we can try removing 2 from the new base?
                        # Let's update U, V, W to the new best and continue searching?
                        # It's complex to restart. Let's just collect candidates.
                        pass
        
        # Try removing 2 columns (from original)
        # This is expensive, so limit combinations
        # Sample random pairs if too many
        num_pairs = len(list(combinations(range(rank), 2)))
        if num_pairs > 1000:
            # Random sample
            import random
            random.seed(42)
            pairs = random.sample(list(combinations(range(rank), 2)), 1000)
        else:
            pairs = list(combinations(range(rank), 2))
            
        for r1, r2 in pairs:
            if time.time() > deadline:
                break
            remove_set = {r1, r2}
            if check_removable_float(U, V, remove_set):
                remaining = [x for x in range(rank) if x not in remove_set]
                W_new = solve_for_W_exact(U_frac, V_frac, remaining)
                if W_new is not None:
                    if verify_exact(U_frac, V_frac, W_new):
                        if rank - 2 < best_rank:
                            best_rank = rank - 2
                            best_remaining = remaining
        
        # If we found a better rank, construct the output
        if best_rank < rank:
            new_cols = []
            for r_idx, r_orig in enumerate(best_remaining):
                u_dict = {i: U_frac[i][r_orig] for i in range(n*m) if U_frac[i][r_orig] != 0}
                v_dict = {i: V_frac[i][r_orig] for i in range(m*p) if V_frac[i][r_orig] != 0}
                # Need W_new for this configuration
                # We didn't store W_new for the best one, so we need to re-solve or store it.
                # Let's re-solve for the best one to be safe.
                pass
            
            # Re-solve for the best remaining set
            W_best = solve_for_W_exact(U_frac, V_frac, best_remaining)
            if W_best is not None:
                 for r_idx, r_orig in enumerate(best_remaining):
                    u_dict = {i: U_frac[i][r_orig] for i in range(n*m) if U_frac[i][r_orig] != 0}
                    v_dict = {i: V_frac[i][r_orig] for i in range(m*p) if V_frac[i][r_orig] != 0}
                    w_dict = {i: W_best[i][r_idx] for i in range(p*n) if W_best[i][r_idx] != 0}
                    new_cols.append((u_dict, v_dict, w_dict))
            return new_cols
        
        return None

    # Generate the base Strassen decomposition
    cols = generate_strassen_decomposition()
    U, V, W = cols_to_matrices(cols)
    rank = len(cols)
    
    # Verify base decomposition
    if not verify(U, V, W):
        # Fallback to schoolbook
        cols = []
        direct_decompose(n, m, p, 0, 0, 0, n, m, p, cols)
        U, V, W = cols_to_matrices(cols)
        rank = len(cols)
    
    # Try optimized elimination to reduce rank
    optimized_cols = try_optimized_elimination()
    if optimized_cols is not None and len(optimized_cols) < rank:
        cols = optimized_cols
        rank = len(cols)
        U, V, W = cols_to_matrices(cols)
        # Final verification
        if not verify(U, V, W):
            # Revert
            cols = generate_strassen_decomposition()
            U, V, W = cols_to_matrices(cols)
            rank = len(cols)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
