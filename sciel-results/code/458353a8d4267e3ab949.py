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
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
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
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, w7))
        
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
        from itertools import combinations
        partitions = []
        for g in range(1, dim + 1):
            for dividers in combinations(range(dim - 1), g - 1):
                groups = []
                prev = 0
                for d in dividers:
                    groups.append(list(range(prev, d + 1)))
                    prev = d + 1
                groups.append(list(range(prev, dim)))
                partitions.append(groups)
        return partitions

    def build_strassen_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
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
        Uses SVD on the flattened tensor matrix to find a minimal basis.
        """
        if time.time() > deadline:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        M = np.zeros((nm * mp * pn, rank))
        
        for c in range(rank):
            u_col = np.array(U[:, c]) if isinstance(U[0], list) else U[:, c]
            v_col = np.array(V[:, c]) if isinstance(V[0], list) else V[:, c]
            w_col = np.array(W[:, c]) if isinstance(W[0], list) else W[:, c]
            
            uv = np.kron(u_col, v_col)
            uvw = np.kron(uv, w_col)
            M[:, c] = uvw
        
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except Exception:
            return None
        
        # Find the effective rank
        tol = max(M.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-10
        effective_rank = sum(1 for sv in s if sv > tol)
        
        if effective_rank >= rank:
            return None
        
        # Reconstruct using only the top effective_rank singular vectors
        # The new factor matrices are derived from the SVD
        # U_new[:, c] = U_svd[:, c] reshaped appropriately
        
        # For each singular vector, we need to decompose it back into U, V, W components
        # This is the hard part - we need to find U, V, W such that kron(U,V,W) = U_svd[:, c]
        
        # Alternative approach: use the SVD to get a new basis for the column space
        # Then re-express the original U, V, W in terms of this basis
        
        # Simpler approach: if we have linear dependencies, we can eliminate redundant columns
        # Find which columns are linearly dependent
        if effective_rank < rank:
            # Use QR decomposition to find independent columns
            Q, R = np.linalg.qr(M, mode='complete')
            # The first effective_rank columns of Q form a basis
            # We need to find coefficients that express each original column in terms of this basis
            
            # Actually, let's just try to find a smaller set of columns that spans the same space
            # Use greedy algorithm: start with empty set, add columns that increase rank
            
            selected = []
            current_rank = 0
            for c in range(rank):
                if current_rank >= effective_rank:
                    break
                # Check if adding column c increases rank
                if len(selected) == 0:
                    selected.append(c)
                    current_rank = 1
                else:
                    # Check if M[:, selected + [c]] has rank current_rank + 1
                    subset = selected + [c]
                    sub_matrix = M[:, subset]
                    # Use SVD to check rank
                    try:
                        s_sub = np.linalg.svd(sub_matrix, full_matrices=False, compute_uv=False)
                        new_rank = sum(1 for sv in s_sub if sv > tol)
                    except Exception:
                        new_rank = current_rank
                    
                    if new_rank > current_rank:
                        selected.append(c)
                        current_rank = new_rank
            
            if len(selected) < rank:
                # Build new decomposition using only selected columns
                new_rank = len(selected)
                new_U = [[U[i][c] for c in selected] for i in range(nm)]
                new_V = [[V[i][c] for c in selected] for i in range(mp)]
                new_W = [[W[i][c] for c in selected] for i in range(pn)]
                
                # Verify
                if verify(new_U, new_V, new_W, new_rank):
                    return {"U": new_U, "V": new_V, "W": new_W, "rank": new_rank}
        
        return None

    def try_improved_decomposition():
        """
        Try to find a better decomposition by searching over partitions
        and then attempting rank reduction via linear algebra.
        """
        if time.time() > deadline:
            return None
        
        # Search for best partition
        result = find_best_partition()
        if result is not None:
            rank = result.pop("rank")
            if verify(result["U"], result["V"], result["W"], rank):
                # Try to reduce rank further
                reduced = try_reduce_rank(result["U"], result["V"], result["W"], rank)
                if reduced is not None:
                    return reduced
                return result
        
        # Fallback to default partition
        result = build_strassen_decomposition([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]])
        rank = result.pop("rank")
        if verify(result["U"], result["V"], result["W"], rank):
            reduced = try_reduce_rank(result["U"], result["V"], result["W"], rank)
            if reduced is not None:
                return reduced
            return result
        
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

    return best_result
# EVOLVE-BLOCK-END
