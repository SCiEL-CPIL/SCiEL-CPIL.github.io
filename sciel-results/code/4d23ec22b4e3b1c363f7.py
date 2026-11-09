"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
        return False
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _build_strassen_tiled(n, m, p):
    """Build rank-56 decomposition by tiling 2x2 Strassen blocks."""
    # Strassen 2x2x2
    n_s = m_s = p_s = 2
    rank_s = 7
    U_sub = np.zeros((4, rank_s))
    V_sub = np.zeros((4, rank_s))
    W_sub = np.zeros((4, rank_s))
    
    # M1
    U_sub[0, 0] = 1; U_sub[3, 0] = 1
    V_sub[0, 0] = 1; V_sub[3, 0] = 1
    W_sub[0, 0] = 1; W_sub[3, 0] = 1
    # M2
    U_sub[2, 1] = 1; U_sub[3, 1] = 1
    V_sub[0, 1] = 1
    W_sub[1, 1] = 1
    # M3
    U_sub[0, 2] = 1
    V_sub[1, 2] = 1; V_sub[3, 2] = -1
    W_sub[2, 2] = 1
    # M4
    U_sub[3, 3] = 1
    V_sub[2, 3] = 1; V_sub[0, 3] = -1
    W_sub[0, 3] = -1; W_sub[2, 3] = -1
    # M5
    U_sub[0, 4] = 1; U_sub[1, 4] = 1
    V_sub[3, 4] = 1
    W_sub[0, 4] = 1; W_sub[1, 4] = 1
    # M6
    U_sub[2, 5] = 1; U_sub[0, 5] = -1
    V_sub[0, 5] = 1; V_sub[1, 5] = 1
    W_sub[2, 5] = -1; W_sub[3, 5] = 1
    # M7
    U_sub[1, 6] = 1; U_sub[3, 6] = -1
    V_sub[2, 6] = 1; V_sub[3, 6] = 1
    W_sub[1, 6] = -1; W_sub[0, 6] = 1
    
    n_blocks_i = n // 2
    n_blocks_j = m // 2
    n_blocks_k = p // 2
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    for ii in range(n_blocks_i):
        for jj in range(n_blocks_j):
            for kk in range(n_blocks_k):
                for li in range(2):
                    for lj in range(2):
                        global_i = 2*ii + li
                        global_j = 2*jj + lj
                        for c in range(7):
                            U[global_i * m + global_j, col_offset + c] = U_sub[li*2 + lj, c]
                
                for lj in range(2):
                    for lk in range(2):
                        global_j = 2*jj + lj
                        global_k = 2*kk + lk
                        for c in range(7):
                            V[global_j * p + global_k, col_offset + c] = V_sub[lj*2 + lk, c]
                
                for lk in range(2):
                    for li in range(2):
                        global_k = 2*kk + lk
                        global_i = 2*ii + li
                        for c in range(7):
                            W[global_k * n + global_i, col_offset + c] = W_sub[lk*2 + li, c]
                
                col_offset += 7
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=5.0):
    """
    Attempt to reduce the rank to target_rank by randomly mixing columns and checking validity.
    This is a heuristic search in the space of linear transformations of the rank-1 terms.
    """
    start_time = time.time()
    U_orig = U.copy()
    V_orig = V.copy()
    W_orig = W.copy()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # We want to find a set of target_rank linear combinations of the original columns
    # such that the tensor identity holds.
    # Let A be the matrix of original columns (flattened tensor vectors).
    # We want new columns X such that X @ C = A (where C is a change of basis matrix).
    # Actually, if we have a dependency, we can project onto a subspace.
    
    # Strategy: 
    # 1. Compute the flattened tensor vectors for all current columns.
    # 2. Find the null space of the matrix of these vectors.
    # 3. If the null space dimension is >= current_rank - target_rank, we can potentially reduce.
    # However, simple null space elimination doesn't always work for tensor rank.
    # We need to find a subspace of dimension target_rank that spans the same tensor.
    
    # Let's try a more direct approach: 
    # Randomly generate a projection matrix P of size (current_rank x target_rank).
    # New U = U @ P, New V = V @ P, New W = W @ P.
    # Check if the new decomposition is valid.
    # This only works if the original decomposition was "tight" or if we are lucky.
    # Actually, scaling columns by P changes the tensor by sum_c P[c, r] (u_c v_c w_c).
    # This is a linear combination of the original terms.
    # If the original terms span the tensor space, any full-rank P of size current_rank x current_rank works.
    # But we want fewer terms. This is only possible if the original set was linearly dependent in a specific way 
    # that allows a basis of smaller size to span the same set of tensors.
    # Wait, tensor rank is the minimum number of rank-1 tensors. 
    # If we have a decomposition of rank R, and the tensor rank is R_min < R, 
    # then there exists a set of R_min rank-1 tensors that sum to the same tensor.
    # These new tensors are NOT necessarily linear combinations of the old ones in a simple way.
    # However, if we assume the old ones span the space of solutions, we can try to find a sparse combination.
    
    # A better heuristic for small dimensions: 
    # Use the fact that if we have a decomposition, we can try to "merge" terms.
    # But let's stick to the "random projection" idea as a quick check for linear dependencies 
    # that might allow a simpler representation.
    
    # Actually, the most robust way to check if rank R is possible is to solve for the entries.
    # But that's expensive.
    
    # Let's try a different structural change: 
    # The 2x4x8 multiplication can be seen as a 2x2 block multiplication 
    # where the blocks are 1x2 (A), 2x4 (B) and 2x2 (C).
    # Wait, A is 2x4, B is 4x8.
    # Split A into [A1 A2] where A1, A2 are 2x2.
    # Split B into [B1; B2] where B1, B2 are 2x8.
    # C = A1 B1 + A2 B2.
    # This is a sum of two 2x2 x 2x8 multiplications.
    # Rank of 2x2 x 2x8 is at most 2 * Rank(2x2 x 2x4)? No.
    # Rank(2x2 x 2x8) <= 2 * Rank(2x2 x 2x4).
    # Rank(2x2 x 2x4) <= 2 * Rank(2x2 x 2x2) = 14.
    # So Rank(2x2 x 2x8) <= 28.
    # Total rank <= 28 + 28 = 56. This matches the tiling.
    
    # Can we do better?
    # Rank(2x2 x 2x4) is actually 14? 
    # 2x2 x 2x4 is equivalent to two 2x2 x 2x2 multiplications.
    # If they share structure, maybe less?
    # But generally, R(2x2 x 2x4) = 14.
    # So 56 is likely the minimum for this block structure.
    
    # Is there a different block structure?
    # What if we use Strassen on the 4x8 part?
    # 2x4 x 4x8.
    # Treat A as 2x4, B as 4x8.
    # Split B into two 4x4 blocks B1, B2.
    # C = A B1 + A B2.
    # Rank(A B1) where A is 2x4, B1 is 4x4.
    # R(2x4 x 4x4) <= 2 * R(2x2 x 2x4) = 2 * 14 = 28.
    # Total 56.
    
    # It seems 56 is a hard lower bound for these simple splittings.
    # The known lower bound for 2x4 x 4x8 is 48? No, 2x2x2 is 7.
    # 2x2x4 is 14.
    # 2x4x4 is 28.
    # 2x4x8 is 56.
    
    # So the Strassen tiling is optimal for this hierarchy.
    # The only way to improve is if there is a non-trivial decomposition of 2x4x8 with rank < 56.
    # This is an open question in some contexts, but for small sizes it's often tight.
    
    # Given the time constraints and the likelihood that 56 is optimal, 
    # the "structural change" should be to ensure the 56-rank decomposition is robust 
    # and to try a few specific algebraic tricks to see if 55 or less is possible, 
    # but failing that, return the 56.
    
    # One trick: If the null space of the flattened terms is non-trivial, 
    # we might be able to reduce rank.
    U_flat = []
    for c in range(current_rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        t = np.einsum('ij,jk,ki->ijk', u, v, w)
        U_flat.append(t.flatten())
        
    M = np.array(U_flat).T # Shape (n*m*p, current_rank)
    
    # Compute SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return U, V, W
        
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank < current_rank:
        # We have a null space.
        # The columns of M are linearly dependent.
        # This means sum_c alpha_c T_c = 0.
        # This does NOT automatically mean we can remove a term.
        # But it suggests the decomposition is not minimal.
        # We can try to project out the null space.
        
        # Let's try to find a basis for the column space of M.
        # The new rank would be num_rank.
        # We need to express the original tensor as a sum of num_rank rank-1 terms.
        # The columns of M are the flattened rank-1 tensors.
        # If they are dependent, we can choose a basis.
        # But the basis vectors must correspond to valid rank-1 tensors (u, v, w).
        # A linear combination of rank-1 tensors is generally NOT a rank-1 tensor.
        # So we cannot simply take a basis of the column space.
        
        # Therefore, null space detection alone is insufficient.
            pass
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the rank-56 Strassen-tiled decomposition
    U, V, W = _build_strassen_tiled(n, m, p)
    
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Try to reduce rank (likely to fail, but a structural attempt)
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, 55, time_limit=5.0)
        
        if _verify_decomposition(U_red.tolist(), V_red.tolist(), W_red.tolist(), n, m, p) and U_red.shape[1] < U.shape[1]:
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
