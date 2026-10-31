import time
import numpy as np

def construct():
    """
    Structural change: instead of recursive Strassen (rank 49), search for a lower-rank
    exact tensor decomposition of the 4x4x4 matrix-multiplication tensor.

    Approach:
      1. Build the 64x64x64 multiplication tensor T (flattened) with the official
         trace convention W[k*n+i, r].
      2. Use a randomized greedy rank-reduction / iterative compression heuristic
         (a structural alternative to fixed recursive Strassen) that tries to find
         a rank-r decomposition with r < 49.
      3. If a valid lower-rank decomposition is found within the time budget, return it.
         Otherwise fall back to the known rank-49 recursive Strassen decomposition.

    The key structural difference: we model the problem as a direct tensor decomposition
    optimization (searching the space of rank-r factorizations) rather than composing
    a fixed algebraic construction. This opens the possibility of finding non-Strassen
    decompositions (e.g., leveraging the specific 4x4x4 structure) that have rank < 49.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    N = n * m  # 16 (A dimension)
    P = m * p  # 16 (B dimension)
    Q = p * n  # 16 (C dimension, trace convention)

    # Build the multiplication tensor T as a 3D array of shape (16, 16, 16)
    # T[a, b, c] = 1 iff the (a,b,c) triple corresponds to a valid matmul term
    # a = i*m + j  (i=row of A, j=col of A = row of B)
    # b = j*p + k  (j=row of B, k=col of B = col of C)
    # c = k*n + i  (k=col of C, i=row of C)  -- TRACE CONVENTION
    # T[a,b,c] = 1 iff j_a == j_b (the shared index matches)
    T = np.zeros((N, P, Q), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Flatten T to a vector for operations
    T_flat = T.flatten()  # length 4096

    def verify_decomp(U, V, W):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        # Efficient: use einsum or matrix operations
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, outer(U[:,r], V[:,r]) * W[:,r]
        # But for verification, let's do it in a vectorized way
        # Reshape: U is (16, r), V is (16, r), W is (16, r)
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_greedy_search(max_rank, time_limit):
        """
        Attempt to find a rank-`max_rank` decomposition using a greedy/iterative approach.
        Returns (U, V, W) if found, else None.
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return None

        # Strategy: Start from a random initialization and use a simple iterative
        # method to try to fit T. This is a structural alternative to Strassen.
        # We use a variant of the "tensor decomposition via alternating least squares"
        # adapted for exact (not approximate) fitting.
        
        # Actually, for exact decomposition, let's try a different structural approach:
        # Use the fact that the matmul tensor has a specific structure and try to
        # find a decomposition by solving linear systems.
        
        # Approach: Assume rank r. Pick r random "directions" in the tensor space
        # and try to find U, V, W that satisfy the constraints.
        
        # A more principled approach: use the "support vector" method.
        # The matmul tensor T has support on the set S = {(a,b,c) : j_a == j_b}.
        # |S| = 4 * 16 = 64 nonzero entries.
        # A rank-r decomposition means T = sum_{r=1}^R u_r v_r w_r (outer products).
        
        # Let's try a direct approach: for small ranks, enumerate or search.
        # For rank < 49, we need a smarter method.
        
        # Structural idea: Use the "border rank" or "tensor train" like decomposition
        # but for the specific matmul structure.
        
        # Actually, let's try a different structural family: 
        # Use a decomposition based on the Fourier transform / DFT structure.
        # The 4x4 matmul tensor can potentially be decomposed using circulant-like
        # structures that might give lower rank.
        
        # For now, let's implement a randomized search with verification.
        # We'll try ranks from 40 down to 30 (known lower bound for 4x4 is 26 or so).
        
        np.random.seed(42)
        
        # Try a specific structured decomposition: use the "Laderman" style
        # for 4x4 if possible, or a hybrid.
        
        # Actually, let's implement a proper iterative solver:
        # Given a target rank r, initialize U, V, W randomly, then iteratively
        # update each factor by solving a least-squares problem.
        
        def als_decompose(r, max_iter=200):
            """Alternating Least Squares for tensor decomposition."""
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                return None
            
            # Initialize
            U = np.random.randn(N, r) * 0.1
            V = np.random.randn(P, r) * 0.1
            W = np.random.randn(Q, r) * 0.1
            
            for it in range(max_iter):
                elapsed = time.time() - start_time
                if elapsed > time_limit:
                    return None
                
                # Update U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
                # This is a linear system in U[a,:]:
                # Let M_b_c = V[b,:] * W[c,:] (element-wise), shape (r,)
                # Then sum_r U[a,r] * M_b_c[r] = T[a,b,c]
                # Stack all (b,c) pairs: we get a system A_u * U[a,:] = t_u
                # where A_u has rows M_b_c for each (b,c), and t_u = T[a,b,c]
                
                # Build the system matrix for U update
                # A_u shape: (P*Q, r) = (256, r)
                # Row for (b,c): V[b,:] * W[c,:]
                A_u = np.zeros((P * Q, r))
                for bi in range(P):
                    for ci in range(Q):
                        A_u[bi * Q + ci, :] = V[bi, :] * W[ci, :]
                t_u = T.flatten().reshape(N, P * Q).T  # (P*Q, N)
                # Actually t_u[a, bi*Q+ci] = T[a, bi, ci]
                t_u = T.reshape(N, P * Q).T  # (P*Q, N)
                
                # Solve for each a: A_u @ U[a,:] = t_u[:, a]
                # Use lstsq
                for a in range(N):
                    # A_u @ x = t_u[:, a]
                    x, residuals, rank_s, sv = np.linalg.lstsq(A_u, t_u[:, a], rcond=None)
                    U[a, :] = x
                
                # Update V: for each b, V[b,:] should satisfy
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,c
                A_v = np.zeros((N * Q, r))
                for ai in range(N):
                    for ci in range(Q):
                        A_v[ai * Q + ci, :] = U[ai, :] * W[ci, :]
                t_v = T.reshape(P, N * Q).T  # (N*Q, P)
                # t_v[ai*Q+ci, b] = T[ai, b, ci]
                t_v = T.transpose(1, 0, 2).reshape(P, N * Q).T
                
                for b in range(P):
                    x, residuals, rank_s, sv = np.linalg.lstsq(A_v, t_v[:, b], rcond=None)
                    V[b, :] = x
                
                # Update W: for each c, W[c,:] should satisfy
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b
                A_w = np.zeros((N * P, r))
                for ai in range(N):
                    for bi in range(P):
                        A_w[ai * P + bi, :] = U[ai, :] * V[bi, :]
                t_w = T.reshape(Q, N * P).T  # (N*P, Q)
                # t_w[ai*P+bi, c] = T[ai, bi, c]
                t_w = T.transpose(2, 0, 1).reshape(Q, N * P).T
                
                for c in range(Q):
                    x, residuals, rank_s, sv = np.linalg.lstsq(A_w, t_w[:, c], rcond=None)
                    W[c, :] = x
                
                # Check convergence
                if it % 10 == 0:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T_recon - T))
                    if err < 1e-9:
                        return (U, V, W)
            
            # Final check
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_recon, T, atol=1e-9):
                return (U, V, W)
            return None
        
        # Try ranks from 40 down to 30
        for r in range(40, 29, -1):
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                return None
            
            result = als_decompose(r, max_iter=100)
            if result is not None:
                U, V, W = result
                if verify_decomp(U, V, W):
                    return (U, V, W)
        
        return None

    # Try the structural search
    result = try_greedy_search(max_rank=40, time_limit=deadline - 30)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank-49 recursive Strassen
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)

    inner_terms = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,1),1)], [((0,1),1), ((0,0),-1)], [((1,0),1), ((1,1),-1)]),
        ([((0,0),1), ((1,1),-1)], [((1,1),1)], [((0,0),1), ((1,0),1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),-1)]),
        ([((0,1),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((0,0),-1)]),
        ([((0,0),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((1,1),-1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((1,0),-1)], [((0,0),1)]),
    ]

    outer_terms = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((1,0),1), ((1,1),-1)]),
        ([((0,0),1), ((1,1),-1)], [((1,1),1)], [((0,0),1), ((1,0),1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),-1)]),
        ([((0,1),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((0,0),-1)]),
        ([((0,0),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((1,1),-1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((1,0),-1)], [((0,0),1)]),
    ]

    col = 0
    for o_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for i_idx, (a_elem, b_elem, c_elem) in enumerate(inner_terms):
            for (br, bc), ca in a_blocks:
                for (er, ec), cv in a_elem:
                    a_idx = (br * 2 + er) * 4 + (bc * 2 + ec)
                    U[a_idx, col] += ca * cv
            for (br, bc), cb in b_blocks:
                for (er, ec), cv in b_elem:
                    b_idx = (br * 2 + er) * 4 + (bc * 2 + ec)
                    V[b_idx, col] += cb * cv
            for (br, bc), cc in c_blocks:
                for (er, ec), cv in c_elem:
                    k = bc * 2 + ec
                    i = br * 2 + er
                    c_idx = k * 4 + i
                    W[c_idx, col] += cc * cv
            col += 1

    assert col == 49, f"Expected 49, got {col}"

    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Schoolbook fallback
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col_fb = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U_fb[ii * 4 + jj, col_fb] = 1
                    V_fb[jj * 4 + kk, col_fb] = 1
                    W_fb[kk * 4 + ii, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
