"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build the rank-28 Strassen-based decomposition (4 blocks x 7 Strassen terms).
    2. Try to reduce rank by finding linear dependencies among the 28 terms.
    3. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def construct_strassen_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard 2x2x2 Strassen terms (7 terms)
        strassen_terms = [
            {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
            {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
            {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
            {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, 1)]},
            {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
            {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
            {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
        ]
        
        # Four 2x2 blocks in the 4x4 matrix
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        col = 0
        for j_off, k_off in blocks:
            for term in strassen_terms:
                for (li, lj, coeff) in term['U']:
                    U[li * m + (lj + j_off)][col] += coeff
                for (lj, lk, coeff) in term['V']:
                    V[(lj + j_off) * p + (lk + k_off)][col] += coeff
                for (lk, li, coeff) in term['W']:
                    W[(lk + k_off) * n + li][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build the rank-28 decomposition
    best = construct_strassen_rank28()
    
    if not verify(best['U'], best['V'], best['W']):
        # Should not happen, but return anyway
        return best
    
    best_rank = 28
    
    # Try to find linear dependencies to reduce rank
    # Convert to numpy arrays
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    best_rank = U_arr.shape[1]
    
    def try_remove_columns(arrays, max_removals=5):
        """Try to remove up to max_removals columns one at a time."""
        U_a, V_a, W_a = arrays
        rank = U_a.shape[1]
        removed = 0
        for _ in range(max_removals):
            if time_left() < 5:
                break
            found = False
            for c_remove in range(U_a.shape[1]):
                U_test = np.delete(U_a, c_remove, axis=1)
                V_test = np.delete(V_a, c_remove, axis=1)
                W_test = np.delete(W_a, c_remove, axis=1)
                if verify(U_test, V_test, W_test):
                    U_a, V_a, W_a = U_test, V_test, W_test
                    rank -= 1
                    removed += 1
                    found = True
                    break
            if not found:
                break
        return U_a, V_a, W_a, rank, removed
    
    # Phase 1: Try removing single columns iteratively
    U_arr, V_arr, W_arr, best_rank, _ = try_remove_columns((U_arr, V_arr, W_arr), max_removals=8)
    
    # Phase 2: Try column swap optimization - for each pair of columns,
    # check if we can combine them into a single column with smaller entries
    if time_left() > 20:
        rank = U_arr.shape[1]
        improved = True
        while improved and time_left() > 10:
            improved = False
            for i in range(rank):
                if time_left() < 10:
                    break
                for j in range(i + 1, rank):
                    if time_left() < 10:
                        break
                    # Try to see if column i can be expressed as a linear combination
                    # of other columns, allowing us to remove it
                    # Or if we can find a better representation
                    # Check if removing column i still works (already done in phase 1)
                    # Try: can we replace columns i and j with a single combined column?
                    # This is more complex; skip for now and rely on phase 1
                    
                    # Alternative: try permuting columns to see if a different order
                    # allows more removals
                    pass
                    
                    # Try: for column i, see if it's a linear combination of others
                    # Build the tensor contribution of column i
                    U3 = U_arr.reshape(n, m, rank)
                    V3 = V_arr.reshape(m, p, rank)
                    W3 = W_arr.reshape(p, n, rank)
                    
                    # Contribution of column i
                    T_i = np.einsum('ij,jk,ki->ijk', U3[:, :, i], V3[:, :, i], W3[:, :, i])
                    
                    # Check if T_i can be written as sum of other columns' contributions
                    # This is a linear algebra problem on the tensor space
                    # Flatten the tensor space
                    dim = n * m * p
                    T_i_flat = T_i.flatten()
                    
                    # Build matrix of other columns' contributions
                    other_cols = [c for c in range(rank) if c != i]
                    if len(other_cols) < dim:
                        # Underdetermined - might still work but hard to check
                        continue
                    
                    M = np.zeros((dim, len(other_cols)))
                    for idx, c in enumerate(other_cols):
                        T_c = np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
                        M[:, idx] = T_c.flatten()
                    
                    # Check if T_i_flat is in the column space of M
                    # Use least squares with residual check
                    try:
                        coeffs, residuals, rank_M, _ = np.linalg.lstsq(M, T_i_flat, rcond=1e-10)
                        if len(residuals) == 0:
                            residual = np.linalg.norm(M @ coeffs - T_i_flat)
                        else:
                            residual = np.linalg.norm(residuals[0])
                        
                        if residual < 1e-8:
                            # Column i is redundant! Remove it
                            U_arr = np.delete(U_arr, i, axis=1)
                            V_arr = np.delete(V_arr, i, axis=1)
                            W_arr = np.delete(W_arr, i, axis=1)
                            rank -= 1
                            best_rank = rank
                            improved = True
                            break
                    except Exception:
                        continue
                    
                    if improved:
                        break
                
                if improved:
                    break
    
    # Phase 3: Final verification and cleanup
    if not verify(U_arr, V_arr, W_arr):
        # Fallback to original rank-28
        best = construct_strassen_rank28()
        return best
    
    # Convert back to lists
    best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    return best

# EVOLVE-BLOCK-END
