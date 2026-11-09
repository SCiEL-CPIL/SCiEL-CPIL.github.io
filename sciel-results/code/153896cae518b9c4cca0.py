"""
Matrix multiplication 4x5 x 5x5: find a rank-minimal exact tensor decomposition.

Strategy:
1. Build the tensor T for 4x5 * 5x5 multiplication.
2. Use a constructive block-based approach combining Strassen and schoolbook.
3. Use ALS (Alternating Least Squares) to find lower-rank exact decompositions.
4. Fallback to schoolbook if no better decomposition is found.

The key insight is to exploit the block structure:
- Split 4x5 into two 2x5 blocks
- Use Strassen-like ideas for 2x2 sub-blocks
- Combine with schoolbook for remaining parts
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit

    # Build the tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) with matching indices
    # Using the convention W[k*n+i, r] for output
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W[k*n+i, r] convention
                T[a, b, c] = 1.0

    def schoolbook():
        """Standard rank-100 decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((n * p, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def to_output(U, V, W):
        """Convert numpy arrays to the required dict format."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 * 2x2.
        Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))

        # m1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # m2 = (a10+a11)*b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1

        # m3 = a00*(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = -1

        # m4 = a11*(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1; W[2, 3] = -1

        # m5 = (a00+a01)*b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1

        # m6 = (a10-a00)*(b00+b10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[2, 5] = 1

        # m7 = (a01-a11)*(b01+b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1

        return U, V, W

    def block_decomposition():
        """
        Construct a decomposition using block structure.
        Split 4x5 into two 2x5 blocks, each multiplied by 5x5.
        Further split 5x5 into 2x2, 2x3, 3x2, 3x3 blocks.
        
        For 2x5 * 5x5:
        - Split B into columns: first 2 cols (2x2), middle 1 col (2x1), last 2 cols (2x2)
        - Or: first 2 cols, last 3 cols
        - Use Strassen for 2x2*2x2 parts, schoolbook for rest
        
        Actually, let's use a simpler approach:
        For 2x5 * 5x5, split the 5 columns of B into groups.
        
        Let's try: split B's 5 columns into 2+3.
        A (2x5) * B (5x5) = A_left (2x2) * B_left (2x5) + A_right (2x3) * B_right (3x5)
        
        For A_left * B_left: 2x2 * 2x5
          Split B_left into two 2x2 blocks: 2x2*2x2 + 2x2*2x2 (but second is 2x2*2x2 with padding)
          Actually 2x2 * 2x5: treat as 2x2 * [2x2 | 2x2|2x1]
          Use Strassen for first 2x2*2x2, schoolbook for 2x2*2x2 and 2x2*2x1
        
        This is getting complex. Let me use a different approach:
        Just use Strassen on 2x2 sub-blocks where possible and schoolbook elsewhere.
        """
        # For 4x5 * 5x5:
        # Split A into A1 (2x5) and A2 (2x5)
        # Split B into B1 (5x2), B2 (5x2), B3 (5x1)
        # C1 = A1*B1 (2x2), C2 = A1*B2 (2x2), C3 = A1*B3 (2x1)
        # C4 = A2*B1 (2x2), C5 = A2*B2 (2x2), C6 = A2*B3 (2x1)
        
        # For each 2x2 * 2x2: use Strassen (rank 7)
        # For each 2x2 * 2x1: use schoolbook (rank 4)
        # Total: 4 * 7 + 2 * 4 = 28 + 8 = 36
        
        # But we can do better by sharing some computations.
        # Let's just implement this and see.
        
        rank = 36
        U = np.zeros((20, rank))
        V = np.zeros((25, rank))
        W = np.zeros((20, rank))
        
        col = 0
        
        # Helper: add a 2x2*2x2 Strassen block
        # A block at rows (i0, i1), cols (j0, j1)
        # B block at rows (j0, j1), cols (k0, k1)
        # C block at rows (i0, i1), cols (k0, k1)
        def add_strassen_2x2x2(i0, i1, j0, j1, k0, k1):
            nonlocal col
            su, sv, sw = strassen_2x2x2()
            for s in range(7):
                # U: A indices
                U[i0 * 5 + j0, col] = su[0, s]
                U[i0 * 5 + j1, col] = su[1, s]
                U[i1 * 5 + j0, col] = su[2, s]
                U[i1 * 5 + j1, col] = su[3, s]
                # V: B indices
                V[j0 * 5 + k0, col] = sv[0, s]
                V[j0 * 5 + k1, col] = sv[1, s]
                V[j1 * 5 + k0, col] = sv[2, s]
                V[j1 * 5 + k1, col] = sv[3, s]
                # W: C indices (using k*n+i convention)
                W[k0 * 4 + i0, col] = sw[0, s]
                W[k0 * 4 + i1, col] = sw[1, s]
                W[k1 * 4 + i0, col] = sw[2, s]
                W[k1 * 4 + i1, col] = sw[3, s]
                col += 1
        
        # Helper: add a 2x2 * 2x1 schoolbook block
        def add_schoolbook_2x2x1(i0, i1, j0, j1, k0):
            nonlocal col
            for i in range(2):
                for j in range(2):
                    U[(i0 + i) * 5 + (j0 + j), col] = 1.0
                    V[(j0 + j) * 5 + k0, col] = 1.0
                    W[k0 * 4 + (i0 + i), col] = 1.0
                    col += 1
        
        # A1: rows 0,1; A2: rows 2,3
        # B columns: 0,1 | 2,3 | 4
        # C columns: 0,1 | 2,3 | 4
        
        # A1 * B1: 2x2 * 2x2 -> Strassen
        add_strassen_2x2x2(0, 1, 0, 1, 0, 1)
        # A1 * B2: 2x2 * 2x2 -> Strassen
        add_strassen_2x2x2(0, 1, 2, 3, 2, 3)
        # A1 * B3: 2x2 * 2x1 -> schoolbook
        add_schoolbook_2x2x1(0, 1, 0, 1, 4)
        # A2 * B1: 2x2 * 2x2 -> Strassen
        add_strassen_2x2x2(2, 3, 0, 1, 0, 1)
        # A2 * B2: 2x2 * 2x2 -> Strassen
        add_strassen_2x2x2(2, 3, 2, 3, 2, 3)
        # A2 * B3: 2x2 * 2x1 -> schoolbook
        add_schoolbook_2x2x1(2, 3, 0, 1, 4)
        
        return U[:, :col], V[:, :col], W[:, :col]

    def try_als_decomposition(rank, max_iters=50, seed=42):
        """
        Try to find a rank-r decomposition using ALS.
        Initialize with schoolbook and iteratively refine.
        """
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with a random mixture of schoolbook columns
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((n * p, rank))
        
        # Start with schoolbook and randomly select/compress
        U_sb, V_sb, W_sb = schoolbook()
        
        # Random linear combinations
        for s in range(rank):
            # Pick 2-3 schoolbook columns and combine
            idxs = rng.choice(n * m * p, size=rng.randint(2, 4), replace=False)
            coeffs = rng.randn(len(idxs))
            coeffs = coeffs / (np.linalg.norm(coeffs) + 1e-10)
            for t, idx in enumerate(idxs):
                U[:, s] += coeffs[t] * U_sb[:, idx]
                V[:, s] += coeffs[t] * V_sb[:, idx]
                W[:, s] += coeffs[t] * W_sb[:, idx]
        
        # Normalize
        for s in range(rank):
            norm = np.linalg.norm(U[:, s]) + np.linalg.norm(V[:, s]) + np.linalg.norm(W[:, s])
            if norm > 1e-10:
                U[:, s] /= norm
                V[:, s] /= norm
                W[:, s] /= norm
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each column s, U[:,s] = T_1 @ (V[:,s] * W[:,s]) / ||V[:,s] * W[:,s]||^2
            for s in range(rank):
                vw = V[:, s] * W[:, s]
                vw_norm_sq = np.dot(vw, vw)
                if vw_norm_sq < 1e-15:
                    continue
                # T_1 is mode-1 unfolding: T_1[a, (b,c)] = T[a,b,c]
                # We need T_1 @ vw
                # T_1[a, b, c] reshaped: T_1[a, b*p*n + c]... let's use einsum
                # sum_{b,c} T[a,b,c] * V[b,s] * W[c,s]
                target = np.einsum('abc,bs,cs->a', T, V[:, s], W[:, s])
                U[:, s] = target / vw_norm_sq
            
            # Update V
            for s in range(rank):
                uw = U[:, s] * W[:, s]
                uw_norm_sq = np.dot(uw, uw)
                if uw_norm_sq < 1e-15:
                    continue
                target = np.einsum('abc,as,cs->b', T, U[:, s], W[:, s])
                V[:, s] = target / uw_norm_sq
            
            # Update W
            for s in range(rank):
                uv = U[:, s] * V[:, s]
                uv_norm_sq = np.dot(uv, uv)
                if uv_norm_sq < 1e-15:
                    continue
                target = np.einsum('abc,as,bs->c', T, U[:, s], V[:, s])
                W[:, s] = target / uv_norm_sq
            
            # Check convergence
            T_recon = np.einsum('as,bs,cs->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-10:
                return U, V, W
        
        # Final check
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-8:
            return U, V, W
        return None

    # Try various approaches
    best_result = None
    best_rank = n * m * p

    # 1. Try block decomposition first
    if time.time() < deadline - 30:
        U_b, V_b, W_b = block_decomposition()
        if verify(U_b, V_b, W_b):
            best_rank = U_b.shape[1]
            best_result = to_output(U_b, V_b, W_b)
            print(f"Block decomposition works with rank {best_rank}")

    # 2. Try ALS with various ranks
    if time.time() < deadline - 15:
        # Try ranks from lower bound up to best_rank - 1
        lower_bound = max(9, best_rank - 20)  # Start from reasonable lower bound
        for rank in range(lower_bound, best_rank):
            if time.time() > deadline - 10:
                break
            
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                
                result = try_als_decomposition(rank, max_iters=30, seed=seed + rank * 7)
                if result is not None:
                    U, V, W = result
                    if verify(U, V, W):
                        best_rank = rank
                        best_result = to_output(U, V, W)
                        break
            if best_result is not None and best_rank <= rank:
                break

    if best_result is not None:
        return best_result

    # Fallback to schoolbook
    U, V, W = schoolbook()
    return to_output(U, V, W)

