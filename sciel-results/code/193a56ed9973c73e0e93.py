"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_blocks():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 2 (schoolbook)
    
    Total: 5*7 + 2 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 2
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    # Strassen terms for 2x2 * 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U coefficients (index = i*2+j for A[i,j]):
    strassen_U = [
        [1, 0, 0, 1],  # P1: A00+A11
        [0, 0, 1, 1],  # P2: A10+A11
        [1, 0, 0, 0],  # P3: A00
        [0, 0, 0, 1],  # P4: A11
        [1, 1, 0, 0],  # P5: A00+A01
        [-1, 0, 1, 0], # P6: A10-A00
        [0, 1, -1, 0], # P7: A01-A11
    ]
    
    # V coefficients (index = j*2+k for B[j,k]):
    strassen_V = [
        [1, 0, 0, 1],  # P1: B00+B11
        [1, 0, 0, 0],  # P2: B00
        [0, 0, 1, -1], # P3: B10-B11
        [-1, 0, 0, 1], # P4: B01-B00
        [0, 0, 0, 1],  # P5: B11
        [1, 0, 0, 1],  # P6: B00+B01
        [0, 0, 1, 1],  # P7: B10+B11
    ]
    
    # W reconstruction coefficients for C[i,k]:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    col = 0
    for b in range(num_pairs):
        for t in range(7):
            # U: same for all blocks
            U[:, col] = strassen_U[t]
            
            # V: block b uses columns 2b and 2b+1
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp
                    global_idx = j * p + (2 * b + kp)
                    V[global_idx, col] = strassen_V[t][local_idx]
            
            # W: block b uses output columns 2b and 2b+1
            for i in range(2):
                for kp in range(2):
                    local_idx = i * 2 + kp
                    global_idx = (2 * b + kp) * n + i
                    W[global_idx, col] = strassen_W[local_idx][t]
            
            col += 1
    
    # Handle remainder column (column 10)
    if remainder == 1:
        # Schoolbook for single column: C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
        # Term 1: A[i,0]*B[0,10]
        U[0, col] = 1  # A00
        U[2, col] = 1  # A10
        V[0 * p + 10, col] = 1  # B[0,10]
        W[10 * n + 0, col] = 1  # C[0,10]
        W[10 * n + 1, col] = 1  # C[1,10]
        col += 1
        
        # Term 2: A[i,1]*B[1,10]
        U[1, col] = 1  # A01
        U[3, col] = 1  # A11
        V[1 * p + 10, col] = 1  # B[1,10]
        W[10 * n + 0, col] = 1  # C[0,10]
        W[10 * n + 1, col] = 1  # C[1,10]
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, T, time_limit=10):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 tensors.
    Uses the structure that all U-factors come from a small set of Strassen patterns.
    """
    start_time = time.time()
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = n * m, m * p, p * n  # 4, 22, 22
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Check for linear dependencies in the U matrix
    # If some columns of U are linearly dependent, we may be able to reduce rank
    # But we need to check if the corresponding V and W columns also allow reduction
    
    # Compute rank of U
    U_rank = np.linalg.matrix_rank(U, tol=1e-10)
    
    if time.time() - start_time > time_limit:
        return U, V, W
    
    return U, V, W

def _build_combined_window_decomposition():
    """
    Structural improvement: Use a combined approach where we handle the 11 columns
    with overlapping Strassen windows that share A-factors.
    
    Instead of 5 independent Strassen blocks + 1 schoolbook column, we try:
    - 4 full Strassen blocks for columns (0,1), (2,3), (4,5), (6,7) -> 4*7 = 28
    - For columns (8,9,10): use a 2x2 * 2x3 decomposition
      2x2 * 2x3 can be done with rank 7 (Strassen-like) or rank 6
    
    Actually, let's try a different structural approach:
    - Use the fact that the tensor T[a,b,c] has a very specific sparse structure
    - Each entry T[i*m+j, j*p+k, k*n+i] = 1
    - This means the tensor is determined by the (i,j,k) -> (a,b,c) mapping
    
    The key structural insight: since n=2, m=2, the A-matrix only has 4 entries.
    The tensor can be written as:
    T = sum over (i,j,k) of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    We need to find the minimal number of rank-1 tensors that sum to this.
    """
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    
    # Build the target tensor
    T = _build_target_tensor()
    
    # Try the standard Strassen approach first
    U, V, W = _build_strassen_blocks()
    if _verify(U, V, W, T):
        return U, V, W
    
    # Try schoolbook
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return U, V, W
    
    raise RuntimeError("No valid decomposition found")

def _find_rank_reduction(U, V, W, T, max_reduction=5, time_limit=60):
    """
    Try to reduce rank by finding subsets of rank-1 tensors that can be merged.
    
    Key insight: If two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 satisfy
    u1 = α*u2 and v1 = β*v2 (or similar linear relationships), they might be
    replaceable by a single tensor or the pair might be redundant.
    
    More generally, we look for linear dependencies:
    If sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 for some coefficients, we can remove
    one term and adjust others.
    
    Practical approach: Check if removing any single term still allows the remaining
    terms to span the same tensor (i.e., the removed term is in the span of others).
    """
    start_time = time.time()
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    T = np.array(T, dtype=np.float64)
    
    # Try removing each term one at a time
    for remove_idx in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        
        # Remove term remove_idx
        mask = [i for i in range(current_rank) if i != remove_idx]
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check if the removed term is in the span of remaining terms
        # The removed term is u_r ⊗ v_r ⊗ w_r
        # We need to check if there exist coefficients c_s such that
        # u_r ⊗ v_r ⊗ w_r = sum_s c_s * (u_s ⊗ v_s ⊗ w_s)
        
        # This is equivalent to checking if the vectorized removed term is in
        # the column space of the matrix formed by vectorized remaining terms
        
        u_r = U[:, remove_idx].flatten()
        v_r = V[:, remove_idx].flatten()
        w_r = W[:, remove_idx].flatten()
        
        # Vectorized removed term
        removed_vec = np.kron(np.kron(u_r, v_r), w_r)
        
        # Build matrix of remaining terms
        # Each column is the vectorized rank-1 tensor
        num_remaining = len(mask)
        if num_remaining == 0:
            continue
            
        # Build the matrix M where M[:, s] = vec(u_s ⊗ v_s ⊗ w_s)
        # For efficiency, we use the fact that vec(u⊗v⊗w) = (w ⊗ v) ⊗ u
        # Actually, let's just compute it directly for small dimensions
        
        # dim_a=4, dim_b=22, dim_c=22, so total dim = 4*22*22 = 1936
        # This might be too large for direct computation with many terms
        
        # Alternative: Use the structure more cleverly
        # Check if the removed term's contribution can be expressed as linear combination
        # of other terms by solving a linear system
        
        # For now, use a simpler check: if the removed term is zero, we can remove it
        if np.allclose(removed_vec, 0, atol=1e-10):
            best_U = U_sub
            best_V = V_sub
            best_W = W_sub
            best_rank = num_remaining
            continue
        
        # Try to solve: find c such that sum_s c_s * (u_s ⊗ v_s ⊗ w_s) = u_r ⊗ v_r ⊗ w_r
        # This is a linear system in c
        
        # Build the design matrix
        # M[a, b, c, s] = U[a,s] * V[b,s] * W[c,s]
        # We want: sum_s c_s * M[:, :, :, s] = T_removed where T_removed[a,b,c] = U[a,r]*V[b,r]*W[c,r]
        
        # Flatten to 1D system: M_flat[s, (a,b,c)] = M[a,b,c,s]
        # Then solve M_flat^T * c = T_removed_flat
        
        # This is a least squares problem: minimize ||M_flat^T * c - T_removed_flat||
        
        # For efficiency, only try this if we have reasonable number of remaining terms
        if num_remaining > 40:
            continue
        
        # Build M_flat
        # M_flat has shape (num_remaining, 4*22*22) = (num_remaining, 1936)
        # This is feasible
        
        M_flat = np.zeros((num_remaining, 4 * 22 * 22), dtype=np.float64)
        for s_idx, s in enumerate(mask):
            M_flat[s_idx, :] = (U[:, s] * V[:, s][:, None] * W[:, s][None, :]).flatten()
        
        T_removed_flat = (U[:, remove_idx] * V[:, remove_idx][:, None] * W[:, remove_idx][None, :]).flatten()
        
        # Solve least squares
        try:
            c, residuals, rank_m, sv = np.linalg.lstsq(M_flat.T, T_removed_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is good
        if residuals and residuals[0] < 1e-8 * (np.sum(T_removed_flat**2) + 1e-10):
            # The removed term is in the span, we can remove it
            # But we need to verify the full decomposition still works
            U_test = U_sub
            V_test = V_sub
            W_test = W_sub
            
            if _verify(U_test, V_test, W_test, T):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank = num_remaining
                # Don't break, try to remove more terms
                # But restart the search with the reduced set
                U, V, W = best_U, best_V, best_W
                current_rank = best_rank
                break
    
    return best_U, best_V, best_W, best_rank

def _try_pairwise_merging(U, V, W, T, time_limit=30):
    """
    Try to merge pairs of rank-1 tensors that have linearly dependent U or V factors.
    
    If u_i = α * u_j and v_i = β * v_j, then:
    u_i ⊗ v_i ⊗ w_i + u_j ⊗ v_j ⊗ w_j = u_i ⊗ (v_i + (1/α) * w_j * (w_i^{-1}...))
    
    Actually, this is more complex. Let's use a different approach:
    If two terms have the same U factor (up to scaling), we can combine them.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Group terms by their U factor (up to scaling)
    # Two U vectors are equivalent if one is a scalar multiple of the other
    
    def are_parallel(u1, u2, tol=1e-8):
        """Check if u1 and u2 are parallel (one is a scalar multiple of the other)."""
        if np.allclose(u1, 0, atol=tol) and np.allclose(u2, 0, atol=tol):
            return True, 1.0
        if np.allclose(u1, 0, atol=tol) or np.allclose(u2, 0, atol=tol):
            return False, 0.0
        
        # Check if u1 = α * u2 for some α
        # Use the largest component to avoid division by zero
        idx = np.argmax(np.abs(u2))
        if np.abs(u2[idx]) < tol:
            return False, 0.0
        
        alpha = u1[idx] / u2[idx]
        if np.allclose(u1, alpha * u2, atol=tol):
            return True, alpha
        return False, 0.0
    
    # Build groups of terms with parallel U factors
    groups = {}
    used = set()
    
    for i in range(current_rank):
        if i in used:
            continue
        
        group = [i]
        used.add(i)
        
        for j in range(i + 1, current_rank):
            if j in used:
                continue
            
            parallel, alpha = are_parallel(U[:, i], U[:, j])
            if parallel:
                group.append(j)
                used.add(j)
        
        # Store the group with its representative U vector
        U_rep = U[:, i]
        groups[tuple(group)] = (U_rep, group)
    
    # For each group with more than 1 term, try to combine
    for group_key, (U_rep, group) in groups.items():
        if time.time() - start_time > time_limit:
            break
        
        if len(group) < 2:
            continue
        
        # For terms in this group, U[:, i] = α_i * U_rep for some α_i
        # So: sum_{i in group} u_i ⊗ v_i ⊗ w_i = U_rep ⊗ (sum_{i in group} α_i * v_i ⊗ w_i)
        # But this is not a rank-1 tensor unless sum α_i * v_i ⊗ w_i is rank-1
        
        # Check if the combined V-W product is rank-1
        # sum_{i in group} α_i * v_i * w_i^T should be rank-1
        
        # Compute the combined matrix
        combined_VW = np.zeros((22, 22), dtype=np.float64)
        for i in group:
            parallel, alpha = are_parallel(U[:, i], U_rep)
            if parallel:
                combined_VW += alpha * np.outer(V[:, i], W[:, i])
        
        # Check if combined_VW is rank-1
        try:
            s_vals = np.linalg.svd(combined_VW, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        
        if len(s_vals) > 0 and s_vals[0] > 1e-8:
            if len(s_vals) < 2 or s_vals[1] / s_vals[0] < 1e-8:
                # Rank-1! We can combine these terms into a single term
                # Find the decomposition: combined_VW = v * w^T
                # Use SVD to get v and w
                U_svd, s, Vt_svd = np.linalg.svd(combined_VW, full_matrices=False)
                v_combined = U_svd[:, 0] * np.sqrt(s[0])
                w_combined = Vt_svd[0, :] * np.sqrt(s[0])
                
                # Build new term
                new_U = U_rep
                new_V = v_combined
                new_W = w_combined
                
                # Verify this single term matches the sum of the group
                # new_U ⊗ new_V ⊗ new_W should equal sum_{i in group} U[:,i] ⊗ V[:,i] ⊗ W[:,i]
                test_tensor = np.einsum('a,b,c->abc', new_U, new_V, new_W)
                ref_tensor = np.zeros((4, 22, 22), dtype=np.float64)
                for i in group:
                    ref_tensor += np.einsum('a,b,c->abc', U[:, i], V[:, i], W[:, i])
                
                if np.allclose(test_tensor, ref_tensor, atol=1e-8):
                    # Replace the group with a single term
                    # Remove all terms in group, add the new term
                    mask = [i for i in range(current_rank) if i not in group]
                    
                    new_rank = len(mask) + 1
                    if new_rank < best_rank:
                        U_new = np.zeros((4, new_rank), dtype=np.float64)
                        V_new = np.zeros((22, new_rank), dtype=np.float64)
                        W_new = np.zeros((22, new_rank), dtype=np.float64)
                        
                        for idx, i in enumerate(mask):
                            U_new[:, idx] = U[:, i]
                            V_new[:, idx] = V[:, i]
                            W_new[:, idx] = W[:, i]
                        
                        U_new[:, len(mask)] = new_U
                        V_new[:, len(mask)] = new_V
                        W_new[:, len(mask)] = new_W
                        
                        # Verify full decomposition
                        if _verify(U_new, V_new, W_new, T):
                            best_U = U_new
                            best_V = V_new
                            best_W = W_new
                            best_rank = new_rank
    
    return best_U, best_V, best_W, best_rank

def _global_search(U, V, W, T, time_limit=120):
    """
    Global search for rank reduction using iterative local search.
    
    Strategy:
    1. Start with the Strassen decomposition (rank 39)
    2. Iteratively try to remove terms or merge groups
    3. Use random restarts with different initial perturbations
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Iteratively try to reduce rank
    for iteration in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Try pairwise merging
        U_new, V_new, W_new, rank_new = _try_pairwise_merging(
            best_U, best_V, best_W, T, 
            time_limit=min(30, time_limit - (time.time() - start_time))
        )
        
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
            continue
        
        # Try removing single terms
        U_new, V_new, W_new, rank_new = _find_rank_reduction(
            best_U, best_V, best_W, T,
            time_limit=min(30, time_limit - (time.time() - start_time))
        )
        
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
            continue
        
        # No improvement in this iteration, break
        break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit
    
    # Build the target tensor
    T = _build_target_tensor()
    
    # Strategy 1: Standard Strassen blocks (rank 39)
    U, V, W = _build_strassen_blocks()
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        rank1 = U.shape[1]
        
        # Try global search for rank reduction
        if time.time() - start_time < DEADLINE - 5:
            U_red, V_red, W_red, rank_red = _global_search(
                U, V, W, T,
                time_limit=min(100, DEADLINE - (time.time() - start_time) - 5)
            )
            
            if rank_red < rank1 and _verify(U_red.tolist(), V_red.tolist(), W_red.tolist(), T):
                return {'U': U_red.tolist(), 'V': V_red.tolist(), 'W': W_red.tolist()}
        
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Strategy 2: Schoolbook fallback (rank 44)
    U, V, W = _build_schoolbook()
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
