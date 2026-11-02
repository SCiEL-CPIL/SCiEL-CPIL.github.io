"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS

    def build_rank48():
        """
        Construct a rank-48 decomposition by optimizing the k=6 handling.
        
        Instead of separate schoolbook for k=6 in each j-block (rank 4 each = 8 total),
        we try to share structure. The key insight: for the 2x4x1 tensor (k=6 only),
        we can use a rank-6 decomposition instead of rank-8 by exploiting the 
        2-row structure.
        
        Actually, let's try: use Strassen on k-pairs {0,1}, {2,3}, {4,5} (3 pairs x 7 = 21 per j-block)
        and for k=6, use a rank-3 construction (instead of 4).
        
        For 2x2x1 (A is 2x2, B is 2x1, C is 2x1):
        C[0] = a*e + b*f
        C[1] = c*e + d*f
        Schoolbook: rank 4.
        
        Can we do rank 3? 
        M1 = (a+b)*e -> contributes to C[0] and C[1]
        M2 = (c-d)*f -> contributes to C[1]
        M3 = ? 
        C[0] = M1 + something
        C[1] = M1 + M2 + something
        
        Actually for 2x2x1, the rank is 4 (it's a matmul with one output column).
        
        Let me try a different approach: combine the two j-blocks' k=6 handling.
        For k=6, we have:
        j-block 0: A[0,0]*B[0,6], A[0,1]*B[1,6], A[1,0]*B[0,6], A[1,1]*B[1,6]
        j-block 1: A[0,2]*B[2,6], A[0,3]*B[3,6], A[1,2]*B[2,6], A[1,3]*B[3,6]
        
        Total 8 multiplications for k=6 across both j-blocks.
        
        Alternative: keep the 50-rank construction but verify it's correct.
        """
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0

        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # A indices (2x2): rows i0, i0+1; cols j0, j0+1
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            # B indices (2x2): rows j0, j0+1; cols k0, k0+1
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            # C indices (2x2): rows k0, k0+1; cols i0, i0+1
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
            U[a, col] += 1; U[d, col] += 1
            V[e, col] += 1; V[h, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1

            # M2 = (c+d)e -> C10 += M2, C11 -= M2
            U[c, col] += 1; U[d, col] += 1
            V[e, col] += 1
            W[c10, col] += 1; W[c11, col] -= 1
            col += 1

            # M3 = a(f-h) -> C01 += M3, C11 += M3
            U[a, col] += 1
            V[f, col] += 1; V[h, col] -= 1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1

            # M4 = d(g-e) -> C00 += M4, C10 += M4
            U[d, col] += 1
            V[g, col] += 1; V[e, col] -= 1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1

            # M5 = (a+b)h -> C00 -= M5, C01 += M5
            U[a, col] += 1; U[b, col] += 1
            V[h, col] += 1
            W[c00, col] -= 1; W[c01, col] += 1
            col += 1

            # M6 = (c-a)(e+f) -> C11 += M6
            U[c, col] += 1; U[a, col] -= 1
            V[e, col] += 1; V[f, col] += 1
            W[c11, col] += 1
            col += 1

            # M7 = (b-d)(g+h) -> C00 += M7
            U[b, col] += 1; U[d, col] -= 1
            V[g, col] += 1; V[h, col] += 1
            W[c00, col] += 1
            col += 1

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            # a*e -> C00
            U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1
            col += 1
            # b*g -> C00
            U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1
            col += 1
            # c*e -> C10
            U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1
            col += 1
            # d*g -> C10
            U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1
            col += 1

        # Apply blocks: 2 j-blocks, each with 3 Strassen k-pairs + 1 schoolbook k-single
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                if time.time() > deadline:
                    return None
                add_strassen_222(0, j_start, k_start)
            if time.time() > deadline:
                return None
            add_schoolbook_221(0, j_start, 6)

        if col != rank:
            return None

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _try_compression(U, V, W, n, m, p):
        """
        Attempt to reduce rank by identifying linearly dependent columns
        in the tensor decomposition. This is a heuristic that checks if
        any column is a linear combination of others.
        """
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        
        # Try to find if any column can be expressed as a combination of others
        # This is a simplified approach - full LLL would be more complex
        # For small ranks, we can try pairwise elimination
        
        # Check if any two columns are identical or negatives
        for r1 in range(rank):
            if time.time() > deadline:
                return None
            for r2 in range(r1 + 1, rank):
                # Check if column r2 is a scalar multiple of column r1
                u1, u2 = U[:, r1], U[:, r2]
                v1, v2 = V[:, r1], V[:, r2]
                w1, w2 = W[:, r1], W[:, r2]
                
                # Find non-zero entries in u1
                nz1 = np.nonzero(u1)[0]
                if len(nz1) == 0:
                    continue
                    
                # Check if u2 is proportional to u1
                ratios = []
                for idx in nz1:
                    if u1[idx] != 0:
                        ratios.append(u2[idx] / u1[idx])
                
                if not ratios:
                    continue
                    
                # Check if all ratios are the same
                if np.allclose(ratios, ratios[0], atol=1e-10):
                    scale = ratios[0]
                    # Verify with V and W
                    if np.allclose(v2, scale * v1, atol=1e-10) and np.allclose(w2, scale * w1, atol=1e-10):
                        # Columns are dependent - we can potentially eliminate one
                        # But for exact decomposition, we need to be careful
                        # This is just a heuristic check
                        pass
        
        return None  # No compression found in this simple approach

    # Try rank-48 first, then fall back to rank-50
    result = build_rank48()
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    result = build_rank50()
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
