"""
Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.

Strategy:
1. Build a rank-46 decomposition using 4 Strassen 2x2 blocks + schoolbook for the
   remaining cross-terms (the 5th row/column of the 5x5 matrix).
2. Optionally attempt to reduce rank further via random initialization + gradient
   descent, snapping to half-integers, and validating.
3. Return the best valid decomposition found (lowest rank).

The tensor T has entries T[u, v, w] = 1 iff the middle indices match, where:
  u = i*m + j  (i in [0,n), j in [0,m))
  v = j*p + k  (j in [0,m), k in [0,p))
  w = k*n + i  (k in [0,p), i in [0,n))
  n=2, m=5, p=5
"""

import time
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import solve
import itertools

DIMENSIONS = [2, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4

    def try_snap_validate(U, V, W):
        # Try half-integers first (most likely for Strassen-based)
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        """
        Build a rank-46 decomposition:
        - 4 Strassen 2x2 blocks covering the 4x4 sub-matrix (indices 0-3 in j and k)
        - Schoolbook for the remaining terms involving index 4 in j or k
        """
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            """Add a Strassen 2x2*2x2 block at offset (j_off, k_off) starting at col_start."""
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        # Add 4 Strassen blocks for the 4x4 sub-matrix
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28

        # Schoolbook for terms involving j=4 or k=4
        # Terms where j=4, k in [0,4): A[i,4] * B[4,k]
        for i in range(n):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        # Terms where j in [0,4), k=4: A[i,j] * B[j,4]
        for i in range(n):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        # Term where j=4, k=4: A[i,4] * B[4,4]
        for i in range(n):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10:
                return U, V, W

        return U, V, W

    def newton_polish(U, V, W, max_iter=100, tol=1e-14):
        """High-precision Newton-style polish for near-exact solutions."""
        U = U.copy()
        V = V.copy()
        W = W.copy()

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break

            # Compute gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            # Adaptive step size
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.005 * min(1.0, 1e-6 / grad_norm)

            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW

        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=300, lr=0.05,
                      beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.9995):
        """Adam optimizer with adaptive restarts."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()

        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)

        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0
        t = 0

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-10:
                break

            # Stagnation detection with restart
            if loss > prev_loss * 1.005 and no_improve > 20:
                restart_count += 1
                if restart_count > 8:
                    break
                U += np.random.randn(*U.shape) * 0.02
                V += np.random.randn(*V.shape) * 0.02
                W += np.random.randn(*W.shape) * 0.02
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                t = 0
                continue
            else:
                no_improve += 1
                prev_loss = loss

            # Adam update
            t += 1
            current_lr = lr * (1 - (it / max_iter) ** 2)  # Quadratic decay

            # Gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            # Adam updates
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * gradU ** 2
            mhatU = mU / (1 - beta1 ** t)
            vhatU = vU / (1 - beta2 ** t)
            U -= current_lr * mhatU / (np.sqrt(vhatU) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * gradV ** 2
            mhatV = mV / (1 - beta1 ** t)
            vhatV = vV / (1 - beta2 ** t)
            V -= current_lr * mhatV / (np.sqrt(vhatV) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * gradW ** 2
            mhatW = mW / (1 - beta1 ** t)
            vhatW = vW / (1 - beta2 ** t)
            W -= current_lr * mhatW / (np.sqrt(vhatW) + eps)

        return best_U, best_V, best_W

    def build_schoolbook():
        """Build the schoolbook rank-50 decomposition as fallback."""
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        return U, V, W

    def try_milp_approach(target_rank, time_budget=30):
        """
        Try to find a rank-target_rank decomposition by fixing V to a canonical
        sparse basis and solving the resulting linear system.
        
        Strategy: Fix V to have exactly one 1 per column (sparse basis vectors).
        Then for each v, M_v[u,w] = T[u,v,w] must equal sum_{r in group(v)} U[u,r]*W[w,r].
        We factor each M_v as a sum of rank-1 outer products with small integer entries.
        """
        start_time = time.time()
        
        # Compute all M_v matrices: M[v, u, w] = T[u, v, w]
        M = np.zeros((V_dim, U_dim, W_dim))
        for v in range(V_dim):
            for u in range(U_dim):
                for w in range(W_dim):
                    M[v, u, w] = T[u, v, w]
        
        # For each v, compute the rank of M[v]
        ranks = []
        for v in range(V_dim):
            r = np.linalg.matrix_rank(M[v], tol=1e-10)
            ranks.append(r)
        
        total_min_rank = sum(ranks)
        if total_min_rank > target_rank:
            return None
        
        def factor_matrix_small(M_mat, max_terms):
            """
            Try to express M_mat (10x10) as sum of rank-1 outer products
            with integer or half-integer entries.
            Returns (U_vecs, W_vecs, num_terms) or None.
            """
            # First try: if matrix is already rank-1, check if it's a simple outer product
            # with integer/half-integer entries
            
            # Try SVD-based approach
            try:
                U_svd, s, Vt = np.linalg.svd(M_mat, full_matrices=False)
            except:
                return None
            
            # Determine numerical rank
            num_rank = 0
            for i in range(len(s)):
                if s[i] > 1e-10:
                    num_rank += 1
                else:
                    break
            
            if num_rank > max_terms:
                return None
            
            # Try to find integer/half-integer factorization
            # For rank 1: M = u * w^T
            # For rank 2: M = u1*w1^T + u2*w2^T
            # etc.
            
            # Strategy: try all combinations of small integer vectors
            # For a 10-dim vector with entries in {-1, 0, 1}, there are 3^10 = 59049 possibilities
            # This is too many for brute force, so use structure of the problem
            
            # Better strategy: exploit that M[v] has a specific structure
            # M[v, u, w] = 1 iff (u % m) == (v // p) AND (w % n) == (u // m)
            
            # Let's analyze the structure:
            # v = j*p + k where j = v//p, k = v%p
            # u = i*m + j' where i = u//m, j' = u%m
            # w = k'*n + i' where k' = w//n, i' = w%n
            # T[u,v,w] = 1 iff j' == j AND i' == i
            
            # So M[v] has 1s at positions (u,w) where:
            #   u % m == v // p  AND  w % n == u // m
            
            j_val = v // p
            k_val = v % p
            
            # Find all (u, w) where M[v, u, w] = 1
            ones = []
            for u in range(U_dim):
                if u % m != j_val:
                    continue
                i_val = u // m
                for w in range(W_dim):
                    if w % n == i_val:
                        ones.append((u, w))
            
            if not ones:
                return np.zeros((U_dim, 0)), np.zeros((W_dim, 0)), 0
            
            # Now we need to express the indicator matrix of these positions
            # as a sum of rank-1 matrices with small integer entries
            
            # For this specific structure, each "row" u with u%m == j_val
            # has 1s at specific w positions (those with w%n == u//m)
            
            # Let's group by i_val = u // m
            # For fixed i_val, u = i_val * m + j_val
            # w positions: w = k' * n + i_val for k' in range(p)
            
            # So M[v] is a block diagonal-like structure
            # For each i in range(n):
            #   u = i*m + j_val
            #   w = k*n + i for k in range(p)
            
            # This matrix has rank = n (since each i gives a distinct row pattern)
            # Actually, let's check: for i=0, u=j_val, w=i=0 => w=0
            # for i=1, u=m+j_val, w=n+1 => w=n+1
            # These are different rows, so rank should be n=2
            
            # Let's try to factor as sum of 2 rank-1 terms
            
            # Term 1: u_vec has 1 at u=j_val (i=0), w_vec has 1 at w=0
            # Term 2: u_vec has 1 at u=m+j_val (i=1), w_vec has 1 at w=n+1
            
            # But wait, we also need w positions for each i:
            # For i=0: w = k*2 + 0 for k=0,1,2,3,4 => w = 0,2,4,6,8
            # For i=1: w = k*2 + 1 for k=0,1,2,3,4 => w = 1,3,5,7,9
            
            # So M[v] has:
            # Row u=j_val: 1s at w=0,2,4,6,8
            # Row u=m+j_val: 1s at w=1,3,5,7,9
            # All other rows: 0
            
            # This is rank 2!
            # u1 = e_{j_val} (1 at position j_val, 0 elsewhere)
            # w1 = [1,0,1,0,1,0,1,0,1,0] (1s at even positions)
            # u2 = e_{m+j_val} (1 at position m+j_val, 0 elsewhere)
            # w2 = [0,1,0,1,0,1,0,1,0,1] (1s at odd positions)
            
            # Check: M[v] = u1*w1^T + u2*w2^T
            # (u1*w1^T)[j_val, w] = w1[w] = 1 if w even, 0 if w odd
            # (u2*w2^T)[m+j_val, w] = w2[w] = 1 if w odd, 0 if w even
            
            # This matches! So we can factor each M[v] as sum of 2 rank-1 terms.
            
            U_vecs = np.zeros((U_dim, 2))
            W_vecs = np.zeros((W_dim, 2))
            
            U_vecs[j_val, 0] = 1
            W_vecs[0, 0] = 1  # w=0
            W_vecs[2, 0] = 1  # w=2
            W_vecs[4, 0] = 1  # w=4
            W_vecs[6, 0] = 1  # w=6
            W_vecs[8, 0] = 1  # w=8
            
            U_vecs[m + j_val, 1] = 1
            W_vecs[1, 1] = 1  # w=1
            W_vecs[3, 1] = 1  # w=3
            W_vecs[5, 1] = 1  # w=5
            W_vecs[7, 1] = 1  # w=7
            W_vecs[9, 1] = 1  # w=9
            
            # Verify
            recon = U_vecs @ W_vecs.T
            if not np.allclose(recon, M_mat, atol=1e-10):
                # Fallback: try SVD with snapping
                U_trunc = U_svd[:, :num_rank]
                s_trunc = s[:num_rank]
                Vt_trunc = Vt[:num_rank, :]
                
                U_scaled = U_trunc * np.sqrt(s_trunc)
                W_scaled = Vt_trunc * np.sqrt(s_trunc)
                
                U_snap = np.round(U_scaled * 2) / 2
                W_snap = np.round(W_scaled * 2) / 2
                
                recon2 = U_snap @ W_snap.T
                if np.allclose(recon2, M_mat, atol=1e-6):
                    return U_snap, W_snap, num_rank
                
                return None
            
            return U_vecs, W_vecs, 2
        
        # Try to factor each M[v]
        U_parts = {}
        W_parts = {}
        total_terms = 0
        
        for v in range(V_dim):
            if time.time() - start_time > time_budget - 5:
                return None
            
            result = factor_matrix_small(M[v], target_rank)
            if result is None:
                return None
            
            U_v, W_v, num_terms = result
            U_parts[v] = U_v
            W_parts[v] = W_v
            total_terms += num_terms
        
        if total_terms > target_rank:
            return None
        
        # Now assemble the full decomposition
        U_full = np.zeros((U_dim, total_terms))
        V_full = np.zeros((V_dim, total_terms))
        W_full = np.zeros((W_dim, total_terms))
        
        col = 0
        for v in range(V_dim):
            if v not in U_parts:
                continue
            U_v, W_v, num_terms = U_parts[v], W_parts[v], U_parts[v].shape[1]
            for r in range(num_terms):
                if col >= total_terms:
                    break
                U_full[:, col] = U_v[:, r]
                V_full[v, col] = 1
                W_full[:, col] = W_v[:, r]
                col += 1
        
        # Validate
        if validate(U_full, V_full, W_full, atol=1e-6):
            return U_full, V_full, W_full
        
        return None

    # Step 1: Build rank-46 Strassen-based decomposition
    U46, V46, W46 = build_rank46()

    # Validate rank-46
    ok46, U46s, V46s, W46s = try_snap_validate(U46, V46, W46)
    if not ok46:
        # Try Newton polish then snap
        U46p, V46p, W46p = newton_polish(U46, V46, W46, max_iter=50)
        ok46, U46s, V46s, W46s = try_snap_validate(U46p, V46p, W46p)

    if ok46:
        best_U, best_V, best_W = U46s, V46s, W46s
        best_rank = 46
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = 50

    # Step 2: Try to find lower rank decompositions
    if time.time() < deadline - 30:
        # First try the MILP-based approach for a few ranks
        for target_rank in [45, 44, 43]:
            if time.time() > deadline - 40:
                break
            
            result = try_milp_approach(target_rank, time_budget=15)
            if result is not None:
                U_s, V_s, W_s = result
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = target_rank
                break
        
        # Then try gradient descent for remaining ranks
        if best_rank > 43:
            for target_rank in range(min(best_rank - 1, 45), 35, -1):
                if time.time() > deadline - 20:
                    break

                # Initialize from best known decomposition (pad with zeros or truncate)
                if best_rank >= target_rank:
                    U_init = np.zeros((U_dim, target_rank))
                    V_init = np.zeros((V_dim, target_rank))
                    W_init = np.zeros((W_dim, target_rank))
                    # Copy from best if possible
                    min_r = min(best_rank, target_rank)
                    U_init[:, :min_r] = best_U[:, :min_r]
                    V_init[:, :min_r] = best_V[:, :min_r]
                    W_init[:, :min_r] = best_W[:, :min_r]
                else:
                    # Random initialization
                    U_init = np.random.randn(U_dim, target_rank) * 0.5
                    V_init = np.random.randn(V_dim, target_rank) * 0.5
                    W_init = np.random.randn(W_dim, target_rank) * 0.5

                # Run optimization
                U_opt, V_opt, W_opt = fast_optimize(
                    U_init, V_init, W_init,
                    target_rank,
                    max_iter=200,
                    lr=0.03
                )

                # Try snapping
                ok, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if ok:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = target_rank
                    break

    # Step 3: Final validation and return
    # Ensure we have a valid decomposition
    if not validate(best_U, best_V, best_W, atol=1e-6):
        # Last resort: schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = 50

    # Convert to lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
