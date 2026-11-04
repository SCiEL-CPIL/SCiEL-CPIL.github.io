import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 4, 6
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[i,k,l] = 1 iff i = c_r*m+b, k = b*p+c_c, l = c_c*n+c_r
    target = np.zeros((nm, mp, pn), dtype=float)
    for c_r in range(n):
        for b in range(m):
            for c_c in range(p):
                i = c_r * m + b
                k = b * p + c_c
                l = c_c * n + c_r
                target[i, k, l] = 1.0

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, target, atol=1e-9)

    def get_strassen_66():
        rank = 66
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00, col] += 1; U[A11, col] += 1
            V[B00, col] += 1; V[B11, col] += 1
            W[C00, col] += 1; W[C11, col] += 1
            col += 1
            U[A10, col] += 1; U[A11, col] += 1
            V[B00, col] += 1
            W[C10, col] += 1; W[C11, col] -= 1
            col += 1
            U[A00, col] += 1
            V[B01, col] += 1; V[B11, col] -= 1
            W[C01, col] += 1; W[C11, col] += 1
            col += 1
            U[A11, col] += 1
            V[B10, col] += 1; V[B00, col] -= 1
            W[C00, col] += 1; W[C10, col] += 1
            col += 1
            U[A00, col] += 1; U[A01, col] += 1
            V[B11, col] += 1
            W[C00, col] -= 1; W[C01, col] += 1
            col += 1
            U[A10, col] += 1; U[A00, col] -= 1
            V[B00, col] += 1; V[B01, col] += 1
            W[C11, col] += 1
            col += 1
            U[A01, col] += 1; U[A11, col] -= 1
            V[B10, col] += 1; V[B11, col] += 1
            W[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_c

            U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
            col += 1
            U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
            col += 1
            U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
            col += 1
            U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)

        return U[:, :col], V[:, :col], W[:, :col]

    U66, V66, W66 = get_strassen_66()
    assert verify(U66, V66, W66), "Rank-66 failed"

    best_U, best_V, best_W = U66, V66, W66
    best_rank = 66

    # ---- Strategy 1: SVD-based tensor compression ----
    # Reshape the decomposition into a 2D matrix and compress
    def svd_compress(U, V, W, target_rank):
        """Compress the tensor network via SVD on reshaped factors."""
        r = U.shape[1]
        # Build the full tensor
        T = np.einsum('ic,kc,lc->ikl', U, V, W)
        # Reshape to (nm, mp*pn)
        T2 = T.reshape(nm, mp * pn)
        # SVD
        U_svd, s, Vt = np.linalg.svd(T2, full_matrices=False)
        if target_rank >= len(s):
            return None, None, None, None
        U_new = U_svd[:, :target_rank] * np.sqrt(s[:target_rank])
        V_new = (np.sqrt(s[:target_rank])[:, None] * Vt[:target_rank, :]).reshape(mp, pn, target_rank)
        V_new = V_new.transpose(1, 2, 0).reshape(mp * pn, target_rank).T  # (target_rank, mp*pn) -> need (mp, target_rank)
        # This approach doesn't directly give us the 3-factor decomposition
        # Let's use a different approach: compress one factor
        return None, None, None, None

    # ---- Strategy 2: Pairwise column merging ----
    # If two columns have similar structures, try to merge them
    def try_merge_columns(U, V, W, col1, col2):
        """Try to merge two columns into one by finding a linear combination."""
        r = U.shape[1]
        mask = [i for i in range(r) if i not in (col1, col2)]
        U_rest = U[:, mask]
        V_rest = V[:, mask]
        W_rest = W[:, mask]

        # We need: sum_{c in rest} u_c ⊗ v_c ⊗ w_c + a*u1⊗v1⊗w1 + b*u2⊗v2⊗w2 = target
        # Try a=1, b=-1 (or other small integer combinations)
        for a in [-2, -1, 0, 1, 2]:
            for b in [-2, -1, 0, 1, 2]:
                if a == 0 and b == 0:
                    continue
                # Compute the residual
                R_rest = np.einsum('ic,kc,lc->ikl', U_rest, V_rest, W_rest)
                R1 = np.einsum('i,k,l->ikl', U[:, col1], V[:, col1], W[:, col1])
                R2 = np.einsum('i,k,l->ikl', U[:, col2], V[:, col2], W[:, col2])
                R_test = R_rest + a * R1 + b * R2
                if np.allclose(R_test, target, atol=1e-9):
                    # We can replace these two columns with a single column
                    # if a*u1⊗v1⊗w1 + b*u2⊗v2⊗w2 can be written as a single rank-1 tensor
                    # This is not generally possible, so skip
                    pass
        return None

    # ---- Strategy 3: Aggressive greedy with multiple restarts and smarter ordering ----
    def greedy_prune_v2(U, V, W, max_iter=300, seed=0):
        rng = random.Random(seed)
        current_U = np.array(U, dtype=float).copy()
        current_V = np.array(V, dtype=float).copy()
        current_W = np.array(W, dtype=float).copy()
        current_rank = current_U.shape[1]

        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break

            # Generate a random order with bias toward columns with smaller norms
            norms = np.array([np.linalg.norm(current_U[:, c]) * np.linalg.norm(current_V[:, c]) * np.linalg.norm(current_W[:, c]) for c in range(current_rank)])
            # Sort by norm ascending (try removing small ones first) with some randomness
            order = list(range(current_rank))
            # Mix: 50% chance to sort by norm, 50% random
            if rng.random() < 0.5:
                order.sort(key=lambda c: norms[c])
            else:
                rng.shuffle(order)

            removed_any = False
            for col_idx in order:
                if time.time() > deadline - 5.0:
                    break
                if col_idx >= current_rank:
                    continue

                mask = [i for i in range(current_rank) if i != col_idx]
                U_new = current_U[:, mask]
                V_new = current_V[:, mask]
                W_new = current_W[:, mask]

                R = np.einsum('ic,kc,lc->ikl', U_new, V_new, W_new)

                if np.allclose(R, target, atol=1e-9):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break

            if not removed_any:
                if iteration > 30:
                    break

        return current_U, current_V, current_W, current_rank

    # Run multiple restarts
    for restart in range(40):
        if time.time() > deadline - 10.0:
            break
        seed_val = restart * 7919 + 12345
        U_p, V_p, W_p, r_p = greedy_prune_v2(U66, V66, W66, max_iter=150, seed=seed_val)
        if verify(U_p, V_p, W_p) and r_p < best_rank:
            best_rank = r_p
            best_U, best_V, best_W = U_p, V_p, W_p

    # If we improved, try more aggressive pruning on the result
    if best_rank < 66:
        for restart in range(30):
            if time.time() > deadline - 5.0:
                break
            seed_val = restart * 104729 + 99999
            U_p, V_p, W_p, r_p = greedy_prune_v2(best_U, best_V, best_W, max_iter=100, seed=seed_val)
            if verify(U_p, V_p, W_p) and r_p < best_rank:
                best_rank = r_p
                best_U, best_V, best_W = U_p, V_p, W_p

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
