"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build the rank-28 Strassen-based decomposition (4 blocks x 7 Strassen terms).
    2. Use systematic linear algebra to find and remove redundant columns.
    3. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
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
        return best
    
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    def compute_tensor_contributions(U_a, V_a, W_a):
        """Compute the tensor contribution of each column."""
        rank = U_a.shape[1]
        U3 = U_a.reshape(n, m, rank)
        V3 = V_a.reshape(m, p, rank)
        W3 = W_a.reshape(p, n, rank)
        contribs = []
        for c in range(rank):
            T_c = np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
            contribs.append(T_c.flatten())
        return np.array(contribs).T  # shape (n*m*p, rank)
    
    def find_redundant_columns(U_a, V_a, W_a, tol=1e-8):
        """Find columns that are linear combinations of others using SVD."""
        rank = U_a.shape[1]
        if rank <= 1:
            return []
        
        contribs = compute_tensor_contributions(U_a, V_a, W_a)
        dim = n * m * p
        
        # For each column, check if it's in the span of the others
        redundant = []
        for i in range(rank):
            if time_left() < 5:
                break
            others = [j for j in range(rank) if j != i]
            M = contribs[:, others]  # (dim, rank-1)
            target = contribs[:, i]
            
            # Use SVD to check if target is in column space of M
            try:
                U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
                # Project target onto column space
                proj = U_svd @ (U_svd.T @ target)
                residual = np.linalg.norm(target - proj)
                
                if residual < tol * (1 + np.linalg.norm(target)):
                    redundant.append(i)
            except Exception:
                continue
        
        return redundant
    
    # Iteratively remove redundant columns
    improved = True
    while improved and time_left() > 10:
        improved = False
        redundant = find_redundant_columns(U_arr, V_arr, W_arr)
        
        for i in redundant:
            if time_left() < 10:
                break
            U_arr = np.delete(U_arr, i, axis=1)
            V_arr = np.delete(V_arr, i, axis=1)
            W_arr = np.delete(W_arr, i, axis=1)
            improved = True
    
    # Final verification
    if not verify(U_arr, V_arr, W_arr):
        best = construct_strassen_rank28()
        return best
    
    best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    return best

# EVOLVE-BLOCK-END
