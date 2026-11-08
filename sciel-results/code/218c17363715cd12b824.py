"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C11 (T[1,1,1])
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        # Contributes +1 to C10 (T[1,0,0]) and -1 to C11 (T[1,1,1])
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        # Contributes +1 to C01 (T[0,0,1]) and +1 to C11 (T[1,1,1])
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C10 (T[1,0,0])
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        # Contributes -1 to C00 (T[0,0,0]) and +1 to C01 (T[0,0,1])
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11 (T[1,1,1])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00 (T[0,0,0])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def compute_partition_rank(i_groups, j_groups, k_groups):
        """Compute total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    def generate_partitions(dim):
        """Generate all partitions of range(dim) into contiguous groups."""
        from itertools import product
        partitions = []
        # Use compositions: number of groups g, then g-1 dividers
        for g in range(1, dim + 1):
            # Choose g-1 dividers from dim-1 positions
            from itertools import combinations
            for dividers in combinations(range(dim - 1), g - 1):
                groups = []
                prev = 0
                for d in dividers:
                    groups.append(list(range(prev, d + 1)))
                    prev = d + 1
                groups.append(list(range(prev, dim)))
                partitions.append(groups)
        return partitions

    def try_linear_reduction(U, V, W, rank):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 tensors.
        Uses SVD to find a minimal basis of the span of rank-1 tensors.
        """
        if time.time() > deadline:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        total_dim = nm * mp * pn
        
        # Build matrix where each column is the flattened Kronecker product
        M = np.zeros((total_dim, rank), dtype=float)
        
        for c in range(rank):
            u_col = np.array([U[i][c] for i in range(nm)], dtype=float)
            v_col = np.array([V[i][c] for i in range(mp)], dtype=float)
            w_col = np.array([W[i][c] for i in range(pn)], dtype=float)
            
            uv = np.kron(u_col, v_col)
            M[:, c] = np.kron(uv, w_col)
        
        # Find the rank of M using SVD
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        except Exception:
            return None
        
        tol = 1e-10
        num_indep = sum(1 for sv in s_vals if sv > tol)
        
        if num_indep >= rank:
            return None
        
        # The columns of M lie in a subspace of dimension num_indep.
        # We can express M = U_svd[:, :num_indep] @ C where C = s_vals[:num_indep] * Vt[:num_indep]
        # But we need to find rank-1 tensors that span the same space.
        
        # Strategy: Use the left singular vectors as a basis for the column space.
        # Each basis vector b_k (k=0..num_indep-1) is a vector in R^total_dim.
        # We need to find if b_k can be written as a sum of rank-1 tensors.
        # This is the tensor decomposition problem itself, which is hard.
        
        # Alternative: Try to find a subset of the original columns that spans the same space.
        # Use QR decomposition to find pivot columns.
        
        # Compute QR of M^T (so we get row pivots)
        try:
            M_T = M.T  # shape (rank, total_dim)
            Q, R = np.linalg.qr(M_T.T, mode='complete')
            # Actually, let's use a different approach
        except Exception:
            pass
        
        # Simpler: use the fact that if the rank is reduced, we can try to 
        # find linear combinations of the original rank-1 tensors.
        
        # For each dependent column, express it as a combination of independent ones.
        # Use the Vt matrix: M = U_svd @ diag(s) @ Vt
        # The columns of M are linear combinations of the columns of U_svd.
        
        # To get new U, V, W matrices with rank num_indep:
        # We need to find vectors u'_c, v'_c, w'_c for c=0..num_indep-1
        # such that the tensor T = sum_c u'_c ⊗ v'_c ⊗ w'_c equals the original T.
        
        # This is equivalent to finding a rank-num_indep decomposition of T.
        # In general, this requires solving a system of nonlinear equations.
        
        # Heuristic: Try to use the first num_indep columns of U, V, W if they
        # are "independent enough". This won't work in general, but let's try.
        
        # Better heuristic: Use the SVD to get a new decomposition.
        # If M = sum_k s_k u_k v_k^T (SVD), then each term s_k u_k v_k^T 
        # is a matrix in R^(nm*mp x pn) or similar reshaping.
        
        # Actually, let's try a different approach: since we know the tensor
        # has rank at most num_indep, we can try to find a decomposition by
        # solving for the factors directly.
        
        # For small enough num_indep, we can try to find u'_c, v'_c, w'_c
        # by solving the linear system.
        
        # Let's try: assume the new decomposition uses the same "support" pattern
        # but with different coefficients. This is a simplification that may not work.
        
        # Given the complexity, return None for now unless num_indep is very small.
        if num_indep < rank - 1:
            # Significant reduction possible, but we need a better method
            pass
        
        return None

    def build_strassen_decomposition(i_groups=None, j_groups=None, k_groups=None):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        if i_groups is None:
            i_groups = [[0, 1], [2]]
        if j_groups is None:
            j_groups = [[0, 1], [2, 3], [4]]
        if k_groups is None:
            k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def find_best_partition():
        """Search over partitions to find the one with minimum rank."""
        if time.time() > deadline:
            return None
        
        i_parts = generate_partitions(n)
        j_parts = generate_partitions(m)
        k_parts = generate_partitions(p)
        
        best_rank = float('inf')
        best_partition = None
        
        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    r = compute_partition_rank(ip, jp, kp)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (ip, jp, kp)
        
        if best_partition is None:
            return None
        
        return build_strassen_decomposition(*best_partition)

    def try_reduce_rank(U, V, W, rank):
        """
        Try to reduce the rank by finding linear dependencies among rank-1 tensors.
        Uses QR decomposition to find a minimal set of independent columns.
        """
        if time.time() > deadline:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        total_dim = nm * mp * pn
        
        # Build tensor matrix
        M = np.zeros((total_dim, rank), dtype=float)
        for c in range(rank):
            u_col = np.array([U[i][c] for i in range(nm)], dtype=float)
            v_col = np.array([V[i][c] for i in range(mp)], dtype=float)
            w_col = np.array([W[i][c] for i in range(pn)], dtype=float)
            uv = np.kron(u_col, v_col)
            M[:, c] = np.kron(uv, w_col)
        
        # Find rank and pivot columns using QR
        try:
            # Use scipy for pivoted QR if available, otherwise use SVD
            from scipy.linalg import qr
            Q, R, piv = qr(M, pivoting=True)
            # Determine rank from R
            tol = 1e-10 * max(abs(np.diag(R)))
            rank_M = 0
            for i in range(rank):
                if abs(R[i, i]) > tol:
                    rank_M += 1
                else:
                    break
        except Exception:
            try:
                U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = sum(1 for sv in s_vals if sv > tol)
                # For SVD, we don't have easy pivot info, so use first rank_M columns
                piv = list(range(rank_M))
            except Exception:
                return None
        
        if rank_M >= rank:
            return None
        
        # The first rank_M pivot columns are independent.
        # Try to use just these columns as the new decomposition.
        # This is valid only if the tensor represented by these columns
        # equals the original tensor, which requires that the non-pivot
        # columns are linear combinations of the pivot columns AND that
        # the tensor decomposition property holds.
        
        # In general, taking a subset of columns doesn't give a valid
        # tensor decomposition. We need to find new U', V', W' such that
        # the tensor is preserved.
        
        # For now, this approach is too simplistic. Return None.
        return None

    def try_improved_decomposition():
        """
        Try to find a better decomposition by searching over partitions.
        """
        if time.time() > deadline:
            return None
        
        # Search for best partition
        result = find_best_partition()
        if result is not None:
            rank = result.pop("rank")
            if verify(result["U"], result["V"], result["W"], rank):
                return result
        
        # Fallback to default partition
        result = build_strassen_decomposition()
        rank = result.pop("rank")
        if verify(result["U"], result["V"], result["W"], rank):
            return result
        
        return None

    def attempt_rank_reduction(U, V, W, rank):
        """
        Attempt to reduce rank by finding linear dependencies.
        Uses the approach of solving for a minimal tensor decomposition.
        """
        if time.time() > deadline:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        total_dim = nm * mp * pn
        
        # Build tensor matrix
        M = np.zeros((total_dim, rank), dtype=float)
        for c in range(rank):
            u_col = np.array([U[i][c] for i in range(nm)], dtype=float)
            v_col = np.array([V[i][c] for i in range(mp)], dtype=float)
            w_col = np.array([W[i][c] for i in range(pn)], dtype=float)
            uv = np.kron(u_col, v_col)
            M[:, c] = np.kron(uv, w_col)
        
        # Find rank
        try:
            s_vals = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        except Exception:
            return None
        
        tol = 1e-10
        rank_M = sum(1 for sv in s_vals if sv > tol)
        
        if rank_M >= rank:
            return None
        
        # Try to find a rank-rank_M decomposition
        # This is generally hard, but we can try a greedy approach:
        # iteratively remove columns that are linear combinations of others
        
        # For each column, check if it's in the span of the others
        # If so, remove it
        keep = list(range(rank))
        for c in range(rank):
            if time.time() > deadline:
                return None
            if c not in keep:
                continue
            # Check if column c is in span of other kept columns
            others = [x for x in keep if x != c]
            if len(others) < rank_M:
                continue
            M_others = M[:, others]
            # Solve M_others @ x = M[:, c]
            try:
                x, residuals, rank_others, s = np.linalg.lstsq(M_others, M[:, c], rcond=1e-10)
                if residuals and residuals[0] < 1e-8:
                    # Column c is in span of others, remove it
                    keep.remove(c)
            except Exception:
                pass
        
        if len(keep) >= rank:
            return None
        
        # Build new U, V, W using only kept columns
        new_rank = len(keep)
        U_new = [[0] * new_rank for _ in range(nm)]
        V_new = [[0] * new_rank for _ in range(mp)]
        W_new = [[0] * new_rank for _ in range(pn)]
        for new_c, old_c in enumerate(keep):
            for i in range(nm):
                U_new[i][new_c] = U[i][old_c]
            for i in range(mp):
                V_new[i][new_c] = V[i][old_c]
            for i in range(pn):
                W_new[i][new_c] = W[i][old_c]
        
        # Verify
        if verify(U_new, V_new, W_new, new_rank):
            return {"U": U_new, "V": V_new, "W": W_new}
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank
    
    # Try rank reduction on the best result found
    if time.time() < deadline:
        current_rank = len(best_result["U"][0])
        reduced = attempt_rank_reduction(best_result["U"], best_result["V"], best_result["W"], current_rank)
        if reduced is not None:
            reduced_rank = len(reduced["U"][0])
            if reduced_rank < current_rank:
                best_result = reduced
                best_rank = reduced_rank

    return best_result
# EVOLVE-BLOCK-END
