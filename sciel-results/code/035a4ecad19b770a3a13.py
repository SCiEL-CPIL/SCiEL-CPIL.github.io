"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build a rank-28 Strassen-based decomposition (4 independent 2x2 Strassen blocks).
    2. Verify correctness.
    3. Attempt column removal (single-column greedy) to reduce rank.
    4. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        if rank == 0:
            return False
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def construct_strassen_rank28():
        """
        Construct rank-28 decomposition using 4 independent Strassen (2x2x2) algorithms.
        Split A into two 2x2 blocks (j=0,1 and j=2,3), B into four 2x2 blocks (j,k offsets).
        Each block uses the rank-7 Strassen algorithm.
        """
        rank = 28
        U = [[0] * rank for _ in range(n * m)]   # 8 x 28
        V = [[0] * rank for _ in range(m * p)]   # 16 x 28
        W = [[0] * rank for _ in range(p * n)]   # 8 x 28
        
        # Strassen's 7 terms for 2x2 matrix multiplication
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h), M2 = (c+d)*e, M3 = a*(f-h), M4 = d*(g-e)
        # M5 = (a+b)*h, M6 = (c-a)*(e+f), M7 = (b-d)(g+h)
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Each term: U has (local_i, local_j, coeff), V has (local_j, local_k, coeff), W has (local_k, local_i, coeff)
        strassen_terms = [
            # M1 = (a+d)(e+h)
            {'U': [(0, 0, 1), (1, 1, 1)],
             'V': [(0, 0, 1), (1, 1, 1)],
             'W': [(0, 0, 1), (1, 1, 1)]},
            # M2 = (c+d)*e
            {'U': [(1, 0, 1), (1, 1, 1)],
             'V': [(0, 0, 1)],
             'W': [(0, 1, 1), (1, 1, -1)]},
            # M3 = a*(f-h)
            {'U': [(0, 0, 1)],
             'V': [(0, 1, 1), (1, 1, -1)],
             'W': [(1, 0, 1), (1, 1, 1)]},
            # M4 = d*(g-e)
            {'U': [(1, 1, 1)],
             'V': [(1, 0, 1), (0, 0, -1)],
             'W': [(0, 0, 1), (0, 1, 1)]},
            # M5 = (a+b)*h
            {'U': [(0, 0, 1), (0, 1, 1)],
             'V': [(1, 1, 1)],
             'W': [(0, 0, -1), (1, 0, 1)]},
            # M6 = (c-a)*(e+f)
            {'U': [(1, 0, 1), (0, 0, -1)],
             'V': [(0, 0, 1), (0, 1, 1)],
             'W': [(1, 1, 1)]},
            # M7 = (b-d)(g+h)
            {'U': [(0, 1, 1), (1, 1, -1)],
             'V': [(1, 0, 1), (1, 1, 1)],
             'W': [(0, 0, 1)]},
        ]
        
        # 4 blocks: (j_offset, k_offset)
        # j_offset: which 2-column block of A (0 or 2)
        # k_offset: which 2-column block of B (0 or 2)
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        col = 0
        for j_off, k_off in blocks:
            for term in strassen_terms:
                for (li, lj, coeff) in term['U']:
                    global_i = li
                    global_j = lj + j_off
                    U[global_i * m + global_j][col] += coeff
                for (lj, lk, coeff) in term['V']:
                    global_j = lj + j_off
                    global_k = lk + k_off
                    V[global_j * p + global_k][col] += coeff
                for (lk, li, coeff) in term['W']:
                    global_k = lk + k_off
                    global_i = li
                    W[global_k * n + global_i][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build the rank-28 Strassen decomposition
    best = construct_strassen_rank28()
    
    # Verify the base decomposition
    if not verify(best['U'], best['V'], best['W']):
        # Should not happen, but return as-is
        return best
    
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    # Phase 1: Greedy single-column removal
    # Try removing each column one at a time; if the decomposition still holds, keep it removed.
    removed_count = 0
    max_removals = 15  # safety cap
    for _ in range(max_removals):
        if time_left() < 15:
            break
        current_rank = U_arr.shape[1]
        if current_rank <= 1:
            break
        found_removal = False
        for c_remove in range(current_rank):
            if time_left() < 10:
                break
            U_test = np.delete(U_arr, c_remove, axis=1)
            V_test = np.delete(V_arr, c_remove, axis=1)
            W_test = np.delete(W_arr, c_remove, axis=1)
            if verify(U_test, V_test, W_test):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                removed_count += 1
                found_removal = True
                break
        if not found_removal:
            break
    
    # Phase 2: Try removing pairs of columns (more aggressive reduction)
    if time_left() > 40:
        current_rank = U_arr.shape[1]
        for _ in range(5):  # up to 5 pair removals
            if time_left() < 20:
                break
            found_pair = False
            for c1 in range(current_rank):
                if time_left() < 15:
                    break
                for c2 in range(c1 + 1, current_rank):
                    if time_left() < 12:
                        break
                    U_test = np.delete(U_arr, [c1, c2], axis=1)
                    V_test = np.delete(V_arr, [c1, c2], axis=1)
                    W_test = np.delete(W_arr, [c1, c2], axis=1)
                    if verify(U_test, V_test, W_test):
                        U_arr = U_test
                        V_arr = V_test
                        W_arr = W_test
                        current_rank = U_arr.shape[1]
                        found_pair = True
                        break
                if found_pair:
                    break
            if not found_pair:
                break
    
    # Final verification
    if not verify(U_arr, V_arr, W_arr):
        # Fallback to the original rank-28
        return best
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END
