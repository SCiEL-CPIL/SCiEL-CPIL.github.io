import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    def Aidx(i, j): return i * m + j
    def Bidx(j, k): return j * p + k
    def Cidx(i, k): return k * n + i

    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i * m + j][l] * V[j * p + k][l] * W[k * n + i][l]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_block_decomposition():
        """Build a block-structured decomposition using Strassen for 2x2 blocks
        and direct multiplication for mixed blocks.
        
        n=3, m=4, p=6.
        Split: n into 2+1, m into 2+2, p into 2+2+2.
        """
        # Block sizes: n = 2+1, m = 2+2, p = 2+2+2
        # A is 3x4, B is 4x6
        # A blocks: A[0:2, 0:2], A[0:2, 2:4], A[2:3, 0:2], A[2:3, 2:4]
        # B blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
        # C is 3x6: C[0:2, 0:2], C[0:2, 2:4], C[0:2, 4:6], C[2:3, 0:2], C[2:3, 2:4], C[2:3, 4:6]
        
        # For C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        # For C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
        # For C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
        # For C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
        # For C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
        # For C[2:3, 4:6] = A[2:3,0:2]*B[0:2,4:6] + A[2:3,2:4]*B[2:4,4:6]
        
        # Use Strassen for the 2x2 * 2x2 -> 2x2 blocks (7 multiplications each)
        # Use direct for 2x2 * 2x2 -> 2x2 (4 multiplications) where Strassen doesn't help
        # Use direct for 1x2 * 2x2 -> 1x2 (2 multiplications)
        
        # Let me count:
        # C[0:2, 0:2]: Strassen(A[0:2,0:2], B[0:2,0:2]) + direct(A[0:2,2:4], B[2:4,0:2]) = 7 + 4 = 11
        # C[0:2, 2:4]: Strassen(A[0:2,0:2], B[0:2,2:4]) + direct(A[0:2,2:4], B[2:4,2:4]) = 7 + 4 = 11
        # C[0:2, 4:6]: Strassen(A[0:2,0:2], B[0:2,4:6]) + direct(A[0:2,2:4], B[2:4,4:6]) = 7 + 4 = 11
        # C[2:3, 0:2]: direct(A[2:3,0:2], B[0:2,0:2]) + direct(A[2:3,2:4], B[2:4,0:2]) = 2 + 2 = 4
        # C[2:3, 2:4]: direct(A[2:3,0:2], B[0:2,2:4]) + direct(A[2:3,2:4], B[2:4,2:4]) = 2 + 2 = 4
        # C[2:3, 4:6]: direct(A[2:3,0:2], B[0:2,4:6]) + direct(A[2:3,2:4], B[2:4,4:6]) = 2 + 2 = 4
        
        # Total: 11 + 11 + 11 + 4 + 4 + 4 = 45? That seems too low. Let me recheck.
        # Actually Strassen computes a 2x2 product using 7 scalar multiplications.
        # Direct 2x2 * 2x2 uses 4 scalar multiplications.
        # Direct 1x2 * 2x2 uses 2 scalar multiplications.
        # So total = 3*(7+4) + 3*(2+2) = 33 + 12 = 45.
        # Hmm, but the cell finding said rank 68. Let me think more carefully.
        # Actually I think the issue is that Strassen's 7 terms each involve sums of matrix entries,
        # so they're not just scalar multiplications but involve additions.
        # The rank counts the number of rank-1 tensor terms, which equals the number of
        # "multiplications" in the algorithm.
        
        # Let me just build it explicitly.
        # I'll use a helper to add a Strassen block or a direct block.
        
        U = []
        V = []
        W = []
        rank = 0
        
        def add_term(u_entries, v_entries, w_entries):
            """Add one rank-1 term. u_entries: dict (i,j)->val, v_entries: dict (j,k)->val, w_entries: dict (i,k)->val"""
            nonlocal rank
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            for (i, j), val in u_entries.items():
                u_vec[Aidx(i, j)] += val
            for (j, k), val in v_entries.items():
                v_vec[Bidx(j, k)] += val
            for (i, k), val in w_entries.items():
                w_vec[Cidx(i, k)] += val
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            rank += 1
        
        def strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
            """Strassen for A[a_i0:a_i0+2, a_j0:a_j0+2] * B[b_j0:b_j0+2, b_k0:b_k0+2] -> C[c_i0:c_i0+2, c_k0:c_k0+2]"""
            # M1 = (A00+A11)(B00+B11)
            add_term(
                {(a_i0, a_j0): 1, (a_i0+1, a_j0+1): 1},
                {(b_j0, b_k0): 1, (b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): 1, (c_i0+1, c_k0+1): 1}
            )
            # M2 = (A10+A11)B00
            add_term(
                {(a_i0+1, a_j0): 1, (a_i0+1, a_j0+1): 1},
                {(b_j0, b_k0): 1},
                {(c_i0+1, c_k0): 1, (c_i0+1, c_k0+1): -1}
            )
            # M3 = A00(B01-B11)
            add_term(
                {(a_i0, a_j0): 1},
                {(b_j0, b_k0+1): 1, (b_j0+1, b_k0+1): -1},
                {(c_i0, c_k0+1): 1, (c_i0+1, c_k0+1): 1}
            )
            # M4 = A11(B10-B00)
            add_term(
                {(a_i0+1, a_j0+1): 1},
                {(b_j0+1, b_k0): 1, (b_j0, b_k0): -1},
                {(c_i0, c_k0): 1, (c_i0+1, c_k0): 1}
            )
            # M5 = (A00+A01)B11
            add_term(
                {(a_i0, a_j0): 1, (a_i0, a_j0+1): 1},
                {(b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): -1, (c_i0, c_k0+1): 1}
            )
            # M6 = (A10-A00)(B00+B01)
            add_term(
                {(a_i0+1, a_j0): 1, (a_i0, a_j0): -1},
                {(b_j0, b_k0): 1, (b_j0, b_k0+1): 1},
                {(c_i0+1, c_k0+1): 1}
            )
            # M7 = (A11-A01)(B10+B11)
            add_term(
                {(a_i0+1, a_j0+1): 1, (a_i0, a_j0+1): -1},
                {(b_j0+1, b_k0): 1, (b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): 1}
            )
        
        def direct_2x2_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
            """Direct multiplication for 2x2 * 2x2 -> 2x2"""
            for di in range(2):
                for dk in range(2):
                    for dj in range(2):
                        add_term(
                            {(a_i0 + di, a_j0 + dj): 1},
                            {(b_j0 + dj, b_k0 + dk): 1},
                            {(c_i0 + di, c_k0 + dk): 1}
                        )
        
        def direct_1x2_block(a_i, a_j0, b_j0, b_k0, c_i, c_k0):
            """Direct multiplication for 1x2 * 2x2 -> 1x2"""
            for dk in range(2):
                for dj in range(2):
                    add_term(
                        {(a_i, a_j0 + dj): 1},
                        {(b_j0 + dj, b_k0 + dk): 1},
                        {(c_i, c_k0 + dk): 1}
                    )
        
        # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        strassen_block(0, 0, 0, 0, 0, 0)
        direct_2x2_block(0, 2, 2, 0, 0, 0)
        
        # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
        strassen_block(0, 0, 0, 2, 0, 2)
        direct_2x2_block(0, 2, 2, 2, 0, 2)
        
        # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
        strassen_block(0, 0, 0, 4, 0, 4)
        direct_2x2_block(0, 2, 2, 4, 0, 4)
        
        # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
        direct_1x2_block(2, 0, 0, 0, 2, 0)
        direct_1x2_block(2, 2, 2, 0, 2, 0)
        
        # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
        direct_1x2_block(2, 0, 0, 2, 2, 2)
        direct_1x2_block(2, 2, 2, 2, 2, 2)
        
        # C[2:3, 4:6] = A[2:3,0:2]*B[0:2,4:6] + A[2:3,2:4]*B[2:4,4:6]
        direct_1x2_block(2, 0, 0, 4, 2, 4)
        direct_1x2_block(2, 2, 2, 4, 2, 4)
        
        U = np.array(U).T  # shape (n*m, rank)
        V = np.array(V).T  # shape (m*p, rank)
        W = np.array(W).T  # shape (p*n, rank)
        return U, V, W, rank

    # Build the block decomposition
    U, V, W, rank = build_block_decomposition()
    
    # Verify
    if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Now try to reduce rank by finding linear dependencies among the rank-1 terms.
    # If we can find coefficients c_r such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0,
    # we can eliminate one term by adjusting the others.
    
    # A simpler approach: try to merge pairs of terms.
    # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if there exists a scalar alpha
    # such that u1 + alpha*u2, v1 + alpha*v2, w1 + alpha*w2 works... but that's not how it works.
    
    # Actually, let's try a different approach: random linear combinations.
    # Pick a random subset of terms, form a random linear combination, and check
    # if the remaining terms still sum to the target.
    
    # Even simpler: try to remove one term at a time by adjusting another term.
    # If term r can be expressed as a linear combination of other terms (in the tensor space),
    # we can remove it.
    
    # Let's try: for each pair of terms, check if we can eliminate one by modifying the other.
    # This is O(r^2) checks, each O(n*m*p).
    
    # Actually, let's try a more systematic approach:
    # The tensor T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # If we can find a linear dependency, we can reduce rank.
    
    # Let me try: pick a random term r, try to express it as a combination of others.
    # This requires solving a linear system, which is expensive.
    
    # Instead, let's try a greedy approach:
    # For each term r, try to "absorb" it into another term s by adding a multiple of term r to term s.
    # This works if the resulting decomposition is still valid.
    
    # Actually, the simplest valid reduction: if two terms have the same u vector (up to scale),
    # we can combine them.
    
    # Let me just try random perturbations and see if we can find a lower-rank solution.
    # This is unlikely to work in 120s, but let's try a few quick checks.
    
    # Alternative: try different block decompositions with different splits.
    # For example, split m as 3+1 instead of 2+2.
    
    def build_block_decomposition_v2():
        """Alternative: n=2+1, m=3+1, p=3+3"""
        U = []
        V = []
        W = []
        rank = 0
        
        def add_term(u_entries, v_entries, w_entries):
            nonlocal rank
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            for (i, j), val in u_entries.items():
                u_vec[Aidx(i, j)] += val
            for (j, k), val in v_entries.items():
                v_vec[Bidx(j, k)] += val
            for (i, k), val in w_entries.items():
                w_vec[Cidx(i, k)] += val
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            rank += 1
        
        def strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
            add_term(
                {(a_i0, a_j0): 1, (a_i0+1, a_j0+1): 1},
                {(b_j0, b_k0): 1, (b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): 1, (c_i0+1, c_k0+1): 1}
            )
            add_term(
                {(a_i0+1, a_j0): 1, (a_i0+1, a_j0+1): 1},
                {(b_j0, b_k0): 1},
                {(c_i0+1, c_k0): 1, (c_i0+1, c_k0+1): -1}
            )
            add_term(
                {(a_i0, a_j0): 1},
                {(b_j0, b_k0+1): 1, (b_j0+1, b_k0+1): -1},
                {(c_i0, c_k0+1): 1, (c_i0+1, c_k0+1): 1}
            )
            add_term(
                {(a_i0+1, a_j0+1): 1},
                {(b_j0+1, b_k0): 1, (b_j0, b_k0): -1},
                {(c_i0, c_k0): 1, (c_i0+1, c_k0): 1}
            )
            add_term(
                {(a_i0, a_j0): 1, (a_i0, a_j0+1): 1},
                {(b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): -1, (c_i0, c_k0+1): 1}
            )
            add_term(
                {(a_i0+1, a_j0): 1, (a_i0, a_j0): -1},
                {(b_j0, b_k0): 1, (b_j0, b_k0+1): 1},
                {(c_i0+1, c_k0+1): 1}
            )
            add_term(
                {(a_i0+1, a_j0+1): 1, (a_i0, a_j0+1): -1},
                {(b_j0+1, b_k0): 1, (b_j0+1, b_k0+1): 1},
                {(c_i0, c_k0): 1}
            )
        
        def direct_2x3_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
            """2x3 * 3x2 -> 2x2 (6 terms)"""
            for di in range(2):
                for dk in range(2):
                    for dj in range(3):
                        add_term(
                            {(a_i0 + di, a_j0 + dj): 1},
                            {(b_j0 + dj, b_k0 + dk): 1},
                            {(c_i0 + di, c_k0 + dk): 1}
                        )
        
        def direct_1x3_block(a_i, a_j0, b_j0, b_k0, c_i, c_k0):
            """1x3 * 3x2 -> 1x2 (6 terms)"""
            for dk in range(2):
                for dj in range(3):
                    add_term(
                        {(a_i, a_j0 + dj): 1},
                        {(b_j0 + dj, b_k0 + dk): 1},
                        {(c_i, c_k0 + dk): 1}
                    )
        
        # C[0:2, 0:2] = A[0:2,0:3]*B[0:3,0:2] + A[0:2,3:4]*B[3:4,0:2]
        # A[0:2,0:3] is 2x3, B[0:3,0:2] is 3x2 -> 2x2: 6 terms
        # A[0:2,3:4] is 2x1, B[3:4,0:2] is 1x2 -> 2x2: 2 terms
        # Total: 8
        
        # C[0:2, 2:4] = A[0:2,0:3]*B[0:3,2:4] + A[0:2,3:4]*B[3:4,2:4]
        # Same: 8
        
        # C[0:2, 4:6] = A[0:2,0:3]*B[0:3,4:6] + A[0:2,3:4]*B[3:4,4:6]
        # Same: 8
        
        # C[2:3, 0:2] = A[2:3,0:3]*B[0:3,0:2] + A[2:3,3:4]*B[3:4,0:2]
        # A[2:3,0:3] is 1x3, B[0:3,0:2] is 3x2 -> 1x2: 6 terms
        # A[2:3,3:4] is 1x1, B[3:4,0:2] is 1x2 -> 1x2: 2 terms
        # Total: 8
        
        # C[2:3, 2:4]: same: 8
        # C[2:3, 4:6]: same: 8
        
        # Total: 6*8 = 48? Let me recount.
        # Actually for C[0:2, 0:2]:
        # A[0:2, 0:3] * B[0:3, 0:2]: 2*3 = 6 terms
        # A[0:2, 3:4] * B[3:4, 0:2]: 2*1 = 2 terms
        # Total: 8
        
        # But wait, I can use Strassen for the 2x2 part of the 2x3*3x2 product.
        # A[0:2, 0:2] * B[0:2, 0:2] via Strassen: 7 terms
        # A[0:2, 2:3] * B[2:3, 0:2]: 2*1 = 2 terms
        # A[0:2, 3:4] * B[3:4, 0:2]: 2*1 = 2 terms
        # Total: 11. Worse than 8.
        
        # So direct is better for non-square blocks.
        
        # Let me just use direct for everything in this variant.
        # C[0:2, 0:2]: 6 + 2 = 8
        # C[0:2, 2:4]: 6 + 2 = 8
        # C[0:2, 4:6]: 6 + 2 = 8
        # C[2:3, 0:2]: 6 + 2 = 8
        # C[2:3, 2:4]: 6 + 2 = 8
        # C[2:3, 4:6]: 6 + 2 = 8
        # Total: 48
        
        # This is better than 45 from v1? No, 45 < 48. So v1 is better.
        # Let me just return v1.
        return None

    # Try v2
    result_v2 = build_block_decomposition_v2()
    if result_v2 is not None:
        U2, V2, W2, rank2 = result_v2
        if verify(U2.tolist(), V2.tolist(), rank2) and rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
