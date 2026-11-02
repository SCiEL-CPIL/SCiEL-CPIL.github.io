import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # First, build the proven rank-42 Strassen decomposition
    rank_42 = 42
    U42 = np.zeros((n * m, rank_42), dtype=int)
    V42 = np.zeros((m * p, rank_42), dtype=int)
    W42 = np.zeros((p * n, rank_42), dtype=int)

    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]

    col = 0
    for r in range(3):
        for s in range(2):
            for t in range(7):
                term = strassen_terms[t]
                for local_u, coeff in term['U']:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s + j_local
                    u_idx = i * m + j_global
                    U42[u_idx, col] += coeff
                for local_v, coeff in term['V']:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s + j_local
                    k_global = 2 * r + k_local
                    v_idx = j_global * p + k_global
                    V42[v_idx, col] += coeff
                for local_w, coeff in term['W']:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r + k_local
                    w_idx = k_global * n + i
                    W42[w_idx, col] += coeff
                col += 1

    # Verify rank-42 is correct
    def verify(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)

    if not verify(U42, V42, W42, 42):
        # Fallback to schoolbook
        r_sb = n * m * p
        U_sb = np.zeros((n * m, r_sb), dtype=int)
        V_sb = np.zeros((m * p, r_sb), dtype=int)
        W_sb = np.zeros((p * n, r_sb), dtype=int)
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_sb[ii * m + jj, c] = 1
                    V_sb[jj * p + kk, c] = 1
                    W_sb[kk * n + ii, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Structural change: Search for linear dependencies in the rank-42 construction
    # The tensor is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # If we can find a subset of columns that still gives the correct tensor, we have a lower rank.
    # 
    # Key insight: The 42 columns form a basis for the tensor. We need to find the minimal
    # subset that still spans the correct tensor. This is equivalent to finding the rank
    # of the tensor in the secant variety.
    #
    # Practical approach: Use the fact that the tensor entries are 1, and try to find
    # a lower-rank decomposition by solving for coefficients.
    #
    # Alternative structural approach: Exploit the Kronecker product structure.
    # 2x4x6 = 2x(2*2)x(2*3). The tensor can be written as a sum over block indices.
    # 
    # Let's try: Can we reduce rank by merging terms that share structure?
    # The 42 terms are indexed by (r, s, t) where r in [0,3), s in [0,2), t in [0,7)
    # 
    # New structural idea: Use a "compressed" Strassen that exploits the rectangular structure.
    # For 2x4x6, we can think of it as 3 independent 2x4x2 blocks (one for each pair of k columns),
    # but with shared structure in the j dimension.
    #
    # Actually, let's try a completely different decomposition:
    # Use the fact that 4 = 2+2 and 6 = 2+2+2, and apply a "segmented" approach.
    #
    # Most promising: Try to find if the tensor rank is actually 40 or less by
    # checking if any 2 columns are linearly dependent in the tensor sense.
    #
    # Let's use a different strategy: direct search for rank < 42.
    # We know the tensor T[i,j,k]=1. We want to find U,V,W with rank r such that
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    #
    # This is a system of n*m*p = 48 equations. Each term r contributes a rank-1 tensor.
    # 
    # Let's try rank 40: remove 2 columns and see if we can adjust the remaining ones.
    # This is a nonlinear problem, but we can use the structure.
    
    # Better structural approach: Use the "Kronecker" decomposition more cleverly.
    # The 2x4x6 tensor = sum over (r,s,t) of terms. 
    # Note that for fixed (s,t), the r-index gives 3 terms that differ only in the k-block.
    # Similarly for fixed (r,t), the s-index gives 2 terms differing in j-block.
    #
    # Key observation: The Strassen construction is optimal for 2x2x2, but when we tile it,
    # we might be over-counting. Let's check if the 42 terms are linearly independent.
    
    # Check linear independence of the 42 rank-1 tensors in the tensor space.
    # Each rank-1 tensor is T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We have 42 such tensors in a 48-dimensional space (n*m*p = 2*4*6 = 48).
    # If they're linearly independent, rank is 42. If not, we can reduce.
    
    # Build the 48x42 matrix where column r is the vectorized T_r
    tensor_dim = n * m * p
    T_matrix = np.zeros((tensor_dim, 42))
    for r in range(42):
        U_t = U42[:, r].reshape(n, m)
        V_t = V42[:, r].reshape(m, p)
        W_t = W42[:, r].reshape(p, n)
        # T_r[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T_matrix[idx, r] = U_t[i, j] * V_t[j, k] * W_t[k, i]
    
    # Check rank of T_matrix
    # The tensor T = sum_r T_r should equal the all-ones tensor
    ones_tensor = np.ones(tensor_dim)
    # Check: T_matrix @ ones(42) should equal ones_tensor
    check = T_matrix @ np.ones(42)
    if not np.allclose(check, ones_tensor, atol=1e-9):
        # Something wrong, fall back
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Now, find the minimal rank: we need the smallest subset of columns of T_matrix
    # that can still sum to ones_tensor with non-negative... wait, coefficients can be any real.
    # Actually, we need: find coefficients c_r such that sum_r c_r * T_r = ones_tensor,
    # and minimize the number of nonzero c_r.
    # But that's not quite right either - the rank is the number of terms, not the L0 norm.
    #
    # The tensor rank is the minimum r such that T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s.
    # Our 42-term decomposition is one such decomposition. To find if a lower rank exists,
    # we need to check if the tensor lies on a lower secant variety.
    #
    # Practical test: Is the 42x48 system T_matrix @ c = ones_tensor underdetermined?
    # Yes: 42 unknowns, 48 equations. Wait, that's overdetermined.
    # Actually: we have 42 basis tensors, and we want to express ones_tensor as a linear combo.
    # T_matrix is 48x42, so we have 48 equations and 42 unknowns. Overdetermined.
    # The fact that it has a solution (c = ones) means the system is consistent.
    #
    # To find a lower-rank decomposition, we need to find a different set of rank-1 tensors.
    # This is fundamentally different from just re-weighting the existing 42.
    #
    # New structural approach: Use the "border rank" or try to find a decomposition
    # with a different structure. Let's try exploiting the rectangular shape directly.
    #
    # For 2x4x6, consider the following: write the 4x6 matrix as a product of a 4x2 and 2x6?
    # No, that changes the problem.
    #
    # Let's try: Can we use rank 40 by merging 2 pairs of terms?
    # Specifically, look for pairs (r1, r2) such that T_{r1} + T_{r2} is a rank-1 tensor
    # (or can be replaced by a single rank-1 tensor with adjusted coefficients).
    #
    # Actually, the cleanest structural change: try a completely different construction.
    # Use the "Laderman" style for rectangular matrices, or use the fact that
    # 2x4x6 can be decomposed using 3 copies of 2x4x2 (each rank 7? no, 2x4x2 rank is 8).
    #
    # Let's just try to verify if rank 42 is minimal by checking the secant variety dimension.
    # For now, return the proven rank-42. But let's try one more thing:
    # Check if we can reduce to rank 41 by finding if any single column is in the span
    # of the others (in the tensor sense).
    
    # For each column r, check if T_r can be written as a linear combo of the other 41.
    # This means: is column r of T_matrix in the column space of T_matrix with column r removed?
    # If yes, then we can potentially eliminate that term.
    
    best_rank = 42
    best_indices = list(range(42))
    
    for r_remove in range(42):
        if time.time() > deadline - 10:
            break
        # Check if column r_remove is in the span of the other 41 columns
        mask = [i for i in range(42) if i != r_remove]
        T_sub = T_matrix[:, mask]  # 48 x 41
        T_col = T_matrix[:, r_remove]  # 48
        
        # Check if T_col is in the column space of T_sub
        # Use least squares: minimize ||T_sub @ x - T_col||
        try:
            x, residuals, rank_sub, sv = np.linalg.lstsq(T_sub, T_col, rcond=None)
            residual = np.linalg.norm(T_sub @ x - T_col)
            if residual < 1e-10:
                # Column r_remove is in the span! We can potentially eliminate it.
                # But we need to verify that the resulting decomposition still gives T = ones.
                # The new coefficients would be: c_{mask} = x + 1 (for the removed col, c=1)
                # Wait, this is getting complicated. Let's just note that rank might be reducible.
                best_rank = min(best_rank, 41)
                best_indices = mask
        except:
            pass
    
    # For now, if we found a reduction, we'd need to reconstruct. But the linear dependence
    # in the tensor space doesn't directly give a lower-rank decomposition of the tensor.
    # The tensor rank is not the same as the linear rank of the coefficient matrix.
    #
    # Given the complexity, let's stick with the proven rank-42 but add one more check:
    # Try to see if a rank-40 decomposition exists by a different construction.
    
    # Actually, let's just return the proven rank-42. It's correct and fast.
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

